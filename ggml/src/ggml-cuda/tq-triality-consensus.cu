#include "tq-triality-consensus.cuh"

#include "turbo-quant.cuh"

struct tq_triality_view {
    const char * data;
    int64_t ne[4];
    uint64_t nb[4];
    int type;
};

struct tq_triality_params {
    float weights[3];
    float bias[3];
    float scale[3];
};

static __device__ __forceinline__ float tq_triality_load_scalar(
        const tq_triality_view & view,
        const char * address) {
    return view.type == GGML_TYPE_F32
        ? *(const float *) address
        : __half2float(*(const half *) address);
}

static __device__ __forceinline__ float tq_triality_load_q(
        const tq_triality_view & q,
        int64_t dim,
        int64_t query,
        int64_t head,
        int64_t stream) {
    return tq_triality_load_scalar(
        q,
        q.data + dim * q.nb[0] + query * q.nb[1] + head * q.nb[2] + stream * q.nb[3]);
}

static __device__ __forceinline__ float tq_triality_rotate_q(
        const tq_triality_view & q,
        const tq_triality_view & rotation,
        int64_t dim,
        int64_t query,
        int64_t head,
        int64_t stream) {
    if (rotation.data == nullptr) {
        return tq_triality_load_q(q, dim, query, head, stream);
    }

    const int64_t block = dim / 8;
    const int64_t row = dim % 8;
    float value = 0.0f;
    for (int64_t col = 0; col < 8; ++col) {
        const int64_t q_dim = block * 8 + col;
        const char * coefficient_address = rotation.ne[0] == q.ne[0]
            ? rotation.data + q_dim * rotation.nb[0] + dim * rotation.nb[1]
            : rotation.data + col * rotation.nb[0] + row * rotation.nb[1] + block * rotation.nb[2];
        value += *(const float *) coefficient_address *
            tq_triality_load_q(q, q_dim, query, head, stream);
    }
    return value;
}

static __device__ __forceinline__ float tq_triality_load_key(
        const tq_triality_view & key,
        const char * row,
        int64_t dim) {
    switch (key.type) {
        case GGML_TYPE_F32:
            return *(const float *) (row + dim * key.nb[0]);
        case GGML_TYPE_F16:
            return __half2float(*(const half *) (row + dim * key.nb[0]));
        case GGML_TYPE_TURBO2_0: {
            const block_turbo2_0 * block = (const block_turbo2_0 *)
                (row + (dim / QK_TURBO2) * key.nb[0]);
            return turbo2_dequant_element(block, dim % QK_TURBO2, __half2float(block->norm));
        }
        case GGML_TYPE_TURBO3_0: {
            const block_turbo3_0 * block = (const block_turbo3_0 *)
                (row + (dim / QK_TURBO3) * key.nb[0]);
            return turbo3_dequant_element(block, dim % QK_TURBO3, __half2float(block->norm));
        }
        case GGML_TYPE_TURBO4_0: {
            const block_turbo4_0 * block = (const block_turbo4_0 *)
                (row + (dim / QK_TURBO4) * key.nb[0]);
            return turbo4_dequant_element(block, dim % QK_TURBO4, __half2float(block->norm));
        }
        default:
            return 0.0f;
    }
}

static __device__ __forceinline__ void tq_triality_wht(float * values, int tid) {
    values[tid] *= TURBO_WHT_SIGNS1[tid];
    __syncthreads();
    for (int stride = 1; stride < 128; stride *= 2) {
        if (tid % (2 * stride) < stride) {
            const float lhs = values[tid];
            const float rhs = values[tid + stride];
            values[tid] = lhs + rhs;
            values[tid + stride] = lhs - rhs;
        }
        __syncthreads();
    }
    values[tid] *= TURBO_WHT_SIGNS2[tid] * 0.08838834764831845f;
    __syncthreads();
}

static __global__ void k_tq_triality_kq_consensus(
        tq_triality_view dst,
        tq_triality_view q,
        tq_triality_view k0,
        tq_triality_view k1,
        tq_triality_view k2,
        tq_triality_view r0,
        tq_triality_view r1,
        tq_triality_view r2,
        tq_triality_params params,
        int64_t n_outputs) {
    const int64_t output_index = blockIdx.x;
    if (output_index >= n_outputs) {
        return;
    }

    const int tid = threadIdx.x;
    int64_t remaining = output_index;
    const int64_t kv = remaining % dst.ne[0];
    remaining /= dst.ne[0];
    const int64_t query = remaining % dst.ne[1];
    remaining /= dst.ne[1];
    const int64_t head = remaining % dst.ne[2];
    const int64_t stream = remaining / dst.ne[2];
    const tq_triality_view keys[3] = { k0, k1, k2 };
    const tq_triality_view rotations[3] = { r0, r1, r2 };
    const int64_t n_groups = (q.ne[0] + 127) / 128;
    __shared__ float values[128];
    float consensus = 0.0f;

    for (int branch = 0; branch < 3; ++branch) {
        const tq_triality_view key = keys[branch];
        const tq_triality_view rotation = rotations[branch];
        const int64_t key_head = head / (q.ne[2] / key.ne[2]);
        const int64_t key_stream = stream / (q.ne[3] / key.ne[3]);
        const char * key_row = key.data +
            kv * key.nb[1] + key_head * key.nb[2] + key_stream * key.nb[3];
        float dot = 0.0f;

        for (int64_t group = 0; group < n_groups; ++group) {
            const int64_t dim = group * 128 + tid;
            values[tid] = (dim < q.ne[0]
                ? tq_triality_rotate_q(q, rotation, dim, query, head, stream)
                : 0.0f);
            tq_triality_wht(values, tid);
            const float q_value = values[tid];
            const bool quantized =
                key.type == GGML_TYPE_TURBO2_0 ||
                key.type == GGML_TYPE_TURBO3_0 ||
                key.type == GGML_TYPE_TURBO4_0;
            float k_value;
            if (quantized) {
                k_value = tq_triality_load_key(key, key_row, dim);
            } else {
                values[tid] = dim < q.ne[0] ? tq_triality_load_key(key, key_row, dim) : 0.0f;
                tq_triality_wht(values, tid);
                k_value = values[tid];
            }
            values[tid] = q_value * k_value;
            __syncthreads();
            for (int stride = 64; stride > 0; stride /= 2) {
                if (tid < stride) {
                    values[tid] += values[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                dot += values[0];
            }
            __syncthreads();
        }

        if (tid == 0) {
            consensus += params.weights[branch] *
                ((dot - params.bias[branch]) / fmaxf(params.scale[branch], 1.0e-6f));
        }
        __syncthreads();
    }

    if (tid == 0) {
        *(float *) (const_cast<char *>(dst.data) +
            kv * dst.nb[0] + query * dst.nb[1] + head * dst.nb[2] + stream * dst.nb[3]) = consensus;
    }
}

static tq_triality_view tq_triality_make_view(const ggml_tensor * tensor) {
    tq_triality_view view = {};
    if (tensor == nullptr) {
        return view;
    }
    view.data = (const char *) tensor->data;
    view.type = tensor->type;
    for (int i = 0; i < 4; ++i) {
        view.ne[i] = tensor->ne[i];
        view.nb[i] = tensor->nb[i];
    }
    return view;
}

void ggml_cuda_tq_triality_kq_consensus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t n_outputs = ggml_nelements(dst);
    tq_triality_params params;
    const float * op_params = (const float *) dst->op_params;
    memcpy(params.weights, op_params + 0, 3 * sizeof(float));
    memcpy(params.bias, op_params + 3, 3 * sizeof(float));
    memcpy(params.scale, op_params + 6, 3 * sizeof(float));

    k_tq_triality_kq_consensus<<<dim3(n_outputs), dim3(128), 0, ctx.stream()>>>(
        tq_triality_make_view(dst),
        tq_triality_make_view(dst->src[0]),
        tq_triality_make_view(dst->src[1]),
        tq_triality_make_view(dst->src[2]),
        tq_triality_make_view(dst->src[3]),
        tq_triality_make_view(dst->src[4]),
        tq_triality_make_view(dst->src[5]),
        tq_triality_make_view(dst->src[6]),
        params,
        n_outputs);
    CUDA_CHECK(cudaGetLastError());
}
