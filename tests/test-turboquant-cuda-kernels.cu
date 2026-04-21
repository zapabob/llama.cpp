#include "testing.h"

#include "../ggml/include/ggml.h"
#include "../ggml/src/ggml-quants.h"

#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int k_tq4_1s_thread_count = 4;
constexpr int k_tq4_1s_values_per_thread = QK_TQ4_1S / k_tq4_1s_thread_count;

__device__ __forceinline__ float ggml_half_to_float_cuda(ggml_half value) {
    return __half2float(*reinterpret_cast<const __half *>(&value));
}

__device__ __forceinline__ ggml_half float_to_ggml_half_cuda(float value) {
    ggml_half out;
    *reinterpret_cast<__half *>(&out) = __float2half_rn(value);
    return out;
}

std::vector<float> make_wave_values(uint32_t count, float phase) {
    std::vector<float> values(count, 0.0f);
    for (uint32_t i = 0; i < count; ++i) {
        values[i] = 0.75f * std::sin(phase + 0.17f * static_cast<float>(i)) +
                    0.35f * std::cos(phase * 0.5f + 0.11f * static_cast<float>(i));
    }
    return values;
}

void cuda_check(cudaError_t status, const char * what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::vector<float> dequantize_q8_1_host(const std::vector<block_q8_1> & blocks) {
    std::vector<float> values(blocks.size() * QK8_1, 0.0f);
    for (size_t block = 0; block < blocks.size(); ++block) {
        const float d = ggml_fp16_to_fp32(blocks[block].d);
        for (int i = 0; i < QK8_1; ++i) {
            values[block * QK8_1 + i] = d * static_cast<float>(blocks[block].qs[i]);
        }
    }
    return values;
}

std::vector<float> dequantize_q8_0_host(const std::vector<block_q8_0> & blocks) {
    std::vector<float> values(blocks.size() * QK8_0, 0.0f);
    dequantize_row_q8_0(blocks.data(), values.data(), static_cast<int64_t>(values.size()));
    return values;
}

float dot_product(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    double sum = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        sum += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
    }
    return static_cast<float>(sum);
}

__device__ __forceinline__ float tq4_1s_centroid_cuda(const uint8_t idx) {
    switch (idx & 0x0F) {
        case 0x0: return -2.732590f;
        case 0x1: return -2.069017f;
        case 0x2: return -1.618046f;
        case 0x3: return -1.256231f;
        case 0x4: return -0.942340f;
        case 0x5: return -0.656759f;
        case 0x6: return -0.388048f;
        case 0x7: return -0.128395f;
        case 0x8: return  0.128395f;
        case 0x9: return  0.388048f;
        case 0xA: return  0.656759f;
        case 0xB: return  0.942340f;
        case 0xC: return  1.256231f;
        case 0xD: return  1.618046f;
        case 0xE: return  2.069017f;
        default:  return  2.732590f;
    }
}

__device__ __constant__ float k_tq4_1s_signs_cuda_local[QK_TQ4_1S] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

__device__ __forceinline__ float tq4_1s_sign_cuda(const int idx) {
    return k_tq4_1s_signs_cuda_local[idx];
}

__device__ __forceinline__ void tq4_1s_fwht32_cuda(float * values) {
    for (int step = 1; step < QK_TQ4_1S; step <<= 1) {
        for (int base = 0; base < QK_TQ4_1S; base += step << 1) {
#pragma unroll
            for (int j = 0; j < step; ++j) {
                const float a = values[base + j];
                const float b = values[base + j + step];
                values[base + j] = a + b;
                values[base + j + step] = a - b;
            }
        }
    }
}

__device__ __forceinline__ void tq4_1s_rht_inverse_cuda(float * values) {
    tq4_1s_fwht32_cuda(values);
#pragma unroll
    for (int i = 0; i < QK_TQ4_1S; ++i) {
        values[i] *= 0.17677669529663688f * tq4_1s_sign_cuda(i);
    }
}

__device__ __forceinline__ void quantize_f32_q8_0_block_local(const float * values, block_q8_0 * block) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        amax = fmaxf(amax, fabsf(values[j]));
    }

    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    block->d = float_to_ggml_half_cuda(d);
    for (int j = 0; j < QK8_0; ++j) {
        block->qs[j] = static_cast<int8_t>(roundf(values[j] * id));
    }
}

__device__ __forceinline__ void dequantize_block_tq4_1s_local(
    const block_tq4_1s & weights_block,
    float * weights) {

    const float d0 = ggml_half_to_float_cuda(weights_block.d0);
    const float d1 = ggml_half_to_float_cuda(weights_block.d1);

#pragma unroll
    for (int i = 0; i < QK_TQ4_1S; ++i) {
        const uint8_t packed = weights_block.qs[i / 2];
        const uint8_t idx = (packed >> ((i & 1) * 4)) & 0x0F;
        const float scale = i < QK_TQ4_1S / 2 ? d0 : d1;
        weights[i] = tq4_1s_centroid_cuda(idx) * scale;
    }

    tq4_1s_rht_inverse_cuda(weights);
}

__device__ __forceinline__ float vec_dot_tq4_1s_q8_1_local(
    const block_tq4_1s * weights_block,
    const block_q8_1 * activations_block,
    const int slice_idx) {

    float weights[QK_TQ4_1S];
    dequantize_block_tq4_1s_local(*weights_block, weights);

    const float act_scale = ggml_half_to_float_cuda(activations_block->d);
    const int offset = slice_idx * k_tq4_1s_values_per_thread;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < k_tq4_1s_values_per_thread; ++i) {
        sum += weights[offset + i] * (act_scale * float(activations_block->qs[offset + i]));
    }

    return sum;
}

__global__ void tq4_1s_to_q8_0_scratch_kernel(
    const block_tq4_1s * weights,
    block_q8_0 * scratch,
    int nblocks) {

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= nblocks) {
        return;
    }

    float values[QK_TQ4_1S];
    dequantize_block_tq4_1s_local(weights[block_idx], values);
    quantize_f32_q8_0_block_local(values, &scratch[block_idx]);
}

__global__ void tq4_1s_dequantize_kernel(
    const block_tq4_1s * weights,
    float * output,
    int nblocks) {

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= nblocks) {
        return;
    }

    float values[QK_TQ4_1S];
    dequantize_block_tq4_1s_local(weights[block_idx], values);
    for (int i = 0; i < QK_TQ4_1S; ++i) {
        output[block_idx * QK_TQ4_1S + i] = values[i];
    }
}

__global__ void tq4_1s_fused_dot_kernel(
    const block_tq4_1s * weights,
    const block_q8_1 * activations,
    int nblocks,
    float * output) {

    __shared__ float partials[k_tq4_1s_thread_count];
    float partial = 0.0f;

    if (threadIdx.x < k_tq4_1s_thread_count) {
        for (int block = 0; block < nblocks; ++block) {
            partial += vec_dot_tq4_1s_q8_1_local(weights + block, activations + block, threadIdx.x);
        }
        partials[threadIdx.x] = partial;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < k_tq4_1s_thread_count; ++i) {
            sum += partials[i];
        }
        *output = sum;
    }
}

} // namespace

int main() {
    testing t;

    t.test("tq4_1s_cuda_fused_dot_matches_cpu_reference_and_q8_0_scratch_path", [](testing & t) {
        constexpr int64_t n = 64;

        const std::vector<float> weights = make_wave_values(static_cast<uint32_t>(n), 0.4f);
        const std::vector<float> activation = make_wave_values(static_cast<uint32_t>(n), 1.2f);

        std::vector<block_tq4_1s> packed_weights(n / QK_TQ4_1S);
        quantize_row_tq4_1s_ref(weights.data(), packed_weights.data(), n);

        std::vector<block_q8_1> packed_activation(n / QK8_1);
        quantize_row_q8_1_ref(activation.data(), packed_activation.data(), n);

        std::vector<float> deq_weights(n, 0.0f);
        dequantize_row_tq4_1s(packed_weights.data(), deq_weights.data(), n);
        const std::vector<float> deq_activation = dequantize_q8_1_host(packed_activation);
        const float cpu_reference = dot_product(deq_weights, deq_activation);

        block_tq4_1s * d_weights = nullptr;
        block_q8_1 * d_activation = nullptr;
        block_q8_0 * d_q8_0 = nullptr;
        float * d_deq_weights = nullptr;
        float * d_fused_dot = nullptr;

        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_weights), packed_weights.size() * sizeof(block_tq4_1s)), "cudaMalloc weights");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_activation), packed_activation.size() * sizeof(block_q8_1)), "cudaMalloc activation");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_q8_0), packed_weights.size() * sizeof(block_q8_0)), "cudaMalloc q8_0 scratch");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_deq_weights), n * sizeof(float)), "cudaMalloc dequant weights");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_fused_dot), sizeof(float)), "cudaMalloc fused dot");

        cuda_check(cudaMemcpy(
            d_weights, packed_weights.data(), packed_weights.size() * sizeof(block_tq4_1s), cudaMemcpyHostToDevice),
            "cudaMemcpy weights");
        cuda_check(cudaMemcpy(
            d_activation, packed_activation.data(), packed_activation.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice),
            "cudaMemcpy activation");

        tq4_1s_fused_dot_kernel<<<1, k_tq4_1s_thread_count>>>(d_weights, d_activation, static_cast<int>(packed_weights.size()), d_fused_dot);
        cuda_check(cudaGetLastError(), "launch tq4_1s fused dot kernel");
        cuda_check(cudaDeviceSynchronize(), "sync fused dot kernel");

        float fused_dot = 0.0f;
        cuda_check(cudaMemcpy(&fused_dot, d_fused_dot, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy fused dot");

        tq4_1s_dequantize_kernel<<<1, static_cast<unsigned int>(packed_weights.size())>>>(
            d_weights,
            d_deq_weights,
            static_cast<int>(packed_weights.size()));
        cuda_check(cudaGetLastError(), "launch tq4_1s dequantize kernel");
        cuda_check(cudaDeviceSynchronize(), "sync tq4_1s dequantize kernel");

        std::vector<float> deq_weights_cuda(n, 0.0f);
        cuda_check(cudaMemcpy(
            deq_weights_cuda.data(), d_deq_weights, n * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy dequantized weights");
        const float cuda_dequant_dot = dot_product(deq_weights_cuda, deq_activation);

        tq4_1s_to_q8_0_scratch_kernel<<<1, static_cast<unsigned int>(packed_weights.size())>>>(
            d_weights,
            d_q8_0,
            static_cast<int>(packed_weights.size()));
        cuda_check(cudaGetLastError(), "launch tq4_1s q8_0 scratch kernel");
        cuda_check(cudaDeviceSynchronize(), "sync tq4_1s q8_0 scratch kernel");

        std::vector<block_q8_0> q8_0_weights(packed_weights.size());
        cuda_check(cudaMemcpy(
            q8_0_weights.data(), d_q8_0, q8_0_weights.size() * sizeof(block_q8_0), cudaMemcpyDeviceToHost),
            "cudaMemcpy q8_0 scratch weights");

        const std::vector<float> deq_q8_0_weights = dequantize_q8_0_host(q8_0_weights);
        const float scratch_dot = dot_product(deq_q8_0_weights, deq_activation);

        std::printf(
            "cpu_reference=%.8f cuda_dequant_dot=%.8f fused_dot=%.8f scratch_dot=%.8f deq_err=%.8f fused_err=%.8f scratch_err=%.8f\n",
            cpu_reference,
            cuda_dequant_dot,
            fused_dot,
            scratch_dot,
            std::fabs(cuda_dequant_dot - cpu_reference),
            std::fabs(fused_dot - cpu_reference),
            std::fabs(scratch_dot - cpu_reference));
        float max_deq_diff = 0.0f;
        int max_deq_index = 0;
        for (int i = 0; i < n; ++i) {
            const float diff = std::fabs(deq_weights_cuda[i] - deq_weights[i]);
            if (diff > max_deq_diff) {
                max_deq_diff = diff;
                max_deq_index = i;
            }
        }
        std::printf(
            "max_deq_diff[%d] cpu=%.8f cuda=%.8f diff=%.8f\n",
            max_deq_index,
            deq_weights[max_deq_index],
            deq_weights_cuda[max_deq_index],
            deq_weights_cuda[max_deq_index] - deq_weights[max_deq_index]);

        t.assert_true("fused tq4_1s dot matches cpu reference", std::fabs(fused_dot - cpu_reference) < 1.0e-4f);
        t.assert_true("q8_0 scratch dot stays close to cpu reference", std::fabs(scratch_dot - cpu_reference) < 2.5e-2f);
        t.assert_true("fused dot and q8_0 scratch stay close", std::fabs(fused_dot - scratch_dot) < 2.5e-2f);

        cudaFree(d_fused_dot);
        cudaFree(d_deq_weights);
        cudaFree(d_q8_0);
        cudaFree(d_activation);
        cudaFree(d_weights);
    });

    return t.summary();
}
