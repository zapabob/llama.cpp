#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

static constexpr uint32_t LLAMA_KV_STATE_K_VIEW_MARKER = 0x54514b56u;
static constexpr uint32_t LLAMA_KV_STATE_RESIDUAL_MARKER = 0x54515250u;
static constexpr uint32_t LLAMA_KV_STATE_RESIDUAL_VERSION = 2u;

static void tq_residual_rotate(
        const std::vector<float> & matrix,
        const float * input,
        float * output,
        uint32_t width,
        bool transpose) {
    for (uint32_t row = 0; row < width; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < width; ++col) {
            const size_t index = transpose ? static_cast<size_t>(col) * width + row : static_cast<size_t>(row) * width + col;
            sum += matrix[index] * input[col];
        }
        output[row] = sum;
    }
}

static uint8_t tq_residual_minus_bit(uint8_t main_code, uint8_t selector) {
    return static_cast<uint8_t>((selector ^ (main_code & 1u)) & 1u);
}

static uint8_t tq_residual_choose_selector(
        uint8_t main_code,
        float plus_value,
        float minus_value,
        const std::array<float, 3> & scales,
        const std::array<float, 2> & beta) {
    double best_error = std::numeric_limits<double>::infinity();
    uint8_t best_selector = 0;
    for (uint8_t selector = 0; selector < 2; ++selector) {
        const uint8_t minus_bit = tq_residual_minus_bit(main_code, selector);
        const double plus_error = static_cast<double>(plus_value) -
            (selector ? scales[1] : -scales[1]);
        const double minus_error = static_cast<double>(minus_value) -
            (minus_bit ? scales[2] : -scales[2]);
        const double error = static_cast<double>(beta[0]) * beta[0] * plus_error * plus_error +
            static_cast<double>(beta[1]) * beta[1] * minus_error * minus_error;
        if (error < best_error) {
            best_error = error;
            best_selector = selector;
        }
    }
    return best_selector;
}

static void tq_residual_pack4(uint8_t * row, uint32_t channel, uint8_t main_code, uint8_t selector) {
    const uint32_t bit = channel * 4u;
    const uint32_t byte = bit >> 3;
    const uint32_t shift = bit & 7u;
    const uint16_t code = static_cast<uint16_t>((main_code & 0x07u) | ((selector & 1u) << 3));
    const uint16_t value = code << shift;
    row[byte] |= static_cast<uint8_t>(value & 0xffu);
    if (shift > 4u) {
        row[byte + 1] |= static_cast<uint8_t>(value >> 8);
    }
}

static uint8_t tq_residual_unpack4(const uint8_t * row, uint32_t channel) {
    const uint32_t bit = channel * 4u;
    const uint32_t byte = bit >> 3;
    const uint32_t shift = bit & 7u;
    uint16_t value = row[byte];
    if (shift > 4u) {
        value |= static_cast<uint16_t>(row[byte + 1]) << 8;
    }
    return static_cast<uint8_t>((value >> shift) & 0x0fu);
}

static bool tq_residual_row_valid(const uint8_t * row, uint32_t logical_channels, uint32_t row_bytes) {
    for (uint32_t channel = 0; channel < logical_channels; ++channel) {
        if ((tq_residual_unpack4(row, channel) & 0x07u) == 0x04u) {
            return false;
        }
    }
    const uint32_t used_bits = logical_channels * 4u;
    const uint32_t padding_bits = row_bytes * 8u - used_bits;
    if (padding_bits != 0u) {
        const uint8_t used_mask = static_cast<uint8_t>((1u << (8u - padding_bits)) - 1u);
        if ((row[row_bytes - 1] & static_cast<uint8_t>(~used_mask)) != 0u) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache::residual_encode_op(
        ggml_tensor * dst,
  const ggml_tensor * seed,
  const ggml_tensor * src,
                int   ith,
                int   nth,
             void *   userdata) {
    GGML_UNUSED(seed);
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    GGML_ASSERT(dst->type == GGML_TYPE_I8 && src->type == GGML_TYPE_F32);
    auto & state = *static_cast<residual_layer_state *>(userdata);
    GGML_ASSERT(static_cast<uint32_t>(dst->ne[0]) == state.row_bytes);
    GGML_ASSERT(static_cast<uint32_t>(src->ne[0]) == state.head_dim);
    GGML_ASSERT(static_cast<uint32_t>(src->ne[1]) == state.n_head);

    std::vector<float> main_rot(state.head_dim);
    std::vector<float> main_dequant(state.head_dim);
    std::vector<float> main_recon(state.head_dim);
    std::vector<float> residual(state.head_dim);
    std::vector<float> plus_rot(state.head_dim);
    std::vector<float> minus_rot(state.head_dim);
    const uint64_t n_rows = ggml_nrows(src) / state.n_head;
    for (uint64_t row_index = 0; row_index < n_rows; ++row_index) {
        auto * packed = static_cast<uint8_t *>(dst->data) + row_index * dst->nb[1];
        std::memset(packed, 0, state.row_bytes);
        for (uint32_t head = 0; head < state.n_head; ++head) {
            const auto * input = reinterpret_cast<const float *>(
                static_cast<const uint8_t *>(src->data) + row_index * src->nb[2] + head * src->nb[1]);
            tq_residual_rotate(state.rotations[0], input, main_rot.data(), state.head_dim, false);
            for (uint32_t channel = 0; channel < state.head_dim; ++channel) {
                const int q = std::clamp(static_cast<int>(std::nearbyint(main_rot[channel] / state.scales[0])), -3, 3);
                main_dequant[channel] = q * state.scales[0];
            }
            tq_residual_rotate(state.rotations[0], main_dequant.data(), main_recon.data(), state.head_dim, true);
            for (uint32_t channel = 0; channel < state.head_dim; ++channel) {
                residual[channel] = input[channel] - main_recon[channel];
            }
            tq_residual_rotate(state.rotations[1], residual.data(), plus_rot.data(), state.head_dim, false);
            tq_residual_rotate(state.rotations[2], residual.data(), minus_rot.data(), state.head_dim, false);
            for (uint32_t channel = 0; channel < state.head_dim; ++channel) {
                const int q = std::clamp(static_cast<int>(std::nearbyint(main_rot[channel] / state.scales[0])), -3, 3);
                const uint8_t code = static_cast<uint8_t>(q) & 0x07u;
                const uint8_t selector = tq_residual_choose_selector(
                    code, plus_rot[channel], minus_rot[channel], state.scales, state.beta);
                tq_residual_pack4(packed, head * state.head_dim + channel, code, selector);
            }
        }
    }
}

void llama_kv_cache::residual_decode_op(
        ggml_tensor * dst,
  const ggml_tensor * seed,
  const ggml_tensor * src,
                int   ith,
                int   nth,
             void *   userdata) {
    GGML_UNUSED(seed);
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && src->type == GGML_TYPE_I8);
    auto & state = *static_cast<residual_layer_state *>(userdata);
    std::vector<float> main(state.head_dim);
    std::vector<float> plus(state.head_dim);
    std::vector<float> minus(state.head_dim);
    std::vector<float> main_inv(state.head_dim);
    std::vector<float> plus_inv(state.head_dim);
    std::vector<float> minus_inv(state.head_dim);
    const uint64_t n_kv = dst->ne[2];
    const uint64_t n_stream = dst->ne[3];
    for (uint64_t row_index = 0; row_index < n_kv * n_stream; ++row_index) {
        const uint64_t stream = row_index / n_kv;
        const uint64_t kv = row_index % n_kv;
        const auto * packed = static_cast<const uint8_t *>(src->data) + stream * src->nb[2] + kv * src->nb[1];
        for (uint32_t head = 0; head < state.n_head; ++head) {
            for (uint32_t channel = 0; channel < state.head_dim; ++channel) {
                const uint8_t code = tq_residual_unpack4(packed, head * state.head_dim + channel);
                const uint8_t main_code = code & 0x07u;
                GGML_ASSERT(main_code != 0x04u);
                const int q = main_code >= 4u ? static_cast<int>(main_code) - 8 : static_cast<int>(main_code);
                const uint8_t selector = (code >> 3) & 1u;
                const uint8_t minus_bit = tq_residual_minus_bit(main_code, selector);
                main[channel] = q * state.scales[0];
                plus[channel] = selector ? state.scales[1] : -state.scales[1];
                minus[channel] = minus_bit ? state.scales[2] : -state.scales[2];
            }
            tq_residual_rotate(state.rotations[0], main.data(), main_inv.data(), state.head_dim, true);
            tq_residual_rotate(state.rotations[1], plus.data(), plus_inv.data(), state.head_dim, true);
            tq_residual_rotate(state.rotations[2], minus.data(), minus_inv.data(), state.head_dim, true);
            auto * output = reinterpret_cast<float *>(
                static_cast<uint8_t *>(dst->data) + stream * dst->nb[3] + kv * dst->nb[2] + head * dst->nb[1]);
            for (uint32_t channel = 0; channel < state.head_dim; ++channel) {
                output[channel] = main_inv[channel] + state.beta[0] * plus_inv[channel] + state.beta[1] * minus_inv[channel];
            }
        }
    }
}

void llama_kv_cache::residual_store_rows_op(
        ggml_tensor * dst,
  const ggml_tensor * cache,
  const ggml_tensor * encoded,
  const ggml_tensor * indices,
                int   ith,
                int   nth,
             void *   userdata) {
    GGML_UNUSED(userdata);
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    GGML_ASSERT(dst->type == GGML_TYPE_I8 && cache->type == GGML_TYPE_I8 && encoded->type == GGML_TYPE_I8);
    GGML_ASSERT(indices->type == GGML_TYPE_I32 || indices->type == GGML_TYPE_I64);
    GGML_ASSERT(dst->data == cache->data);
    GGML_ASSERT(dst->ne[0] == cache->ne[0] && dst->ne[1] == cache->ne[1]);
    GGML_ASSERT(encoded->ne[0] == dst->ne[0]);
    GGML_ASSERT(indices->ne[0] == encoded->ne[1]);

    const int64_t n_rows = encoded->ne[1];
    for (int64_t row = 0; row < n_rows; ++row) {
        const int64_t index = indices->type == GGML_TYPE_I32
            ? static_cast<const int32_t *>(indices->data)[row]
            : static_cast<const int64_t *>(indices->data)[row];
        GGML_ASSERT(index >= 0 && index < dst->ne[1]);
        memcpy(
            static_cast<uint8_t *>(dst->data) + static_cast<size_t>(index) * dst->nb[1],
            static_cast<const uint8_t *>(encoded->data) + static_cast<size_t>(row) * encoded->nb[1],
            static_cast<size_t>(encoded->ne[0]));
    }
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

// InnerQ: cross-TU shared state for CUDA per-channel equalization.
// These are defined in ggml-cuda/turbo-innerq.cu (when CUDA is enabled).
// When CUDA is not available, we provide stub implementations.
#ifndef INNERQ_MAX_CHANNELS
#define INNERQ_MAX_CHANNELS 128
#endif

#ifdef GGML_USE_CUDA
#if defined(_WIN32) && !defined(__MINGW32__)
#  define TURBO_IQ_IMPORT __declspec(dllimport)
#else
#  define TURBO_IQ_IMPORT
#endif
extern TURBO_IQ_IMPORT bool  g_innerq_finalized;
extern TURBO_IQ_IMPORT float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];
TURBO_IQ_IMPORT bool turbo_innerq_needs_tensor_update(void);
TURBO_IQ_IMPORT void turbo_innerq_mark_tensor_updated(void);
#else
static bool  g_innerq_finalized = false;
static float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS] = {};
static bool turbo_innerq_needs_tensor_update(void) { return false; }
static void turbo_innerq_mark_tensor_updated(void) {}
#endif

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
                 uint32_t   tq_view_capacity,
       llama_tq_execution   tq_execution) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), tq_view_capacity(tq_view_capacity), tq_execution(tq_execution), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    if (tq_view_capacity != 1 && tq_view_capacity != 3) {
        throw std::invalid_argument("llama_kv_cache: tq_view_capacity must be 1 or 3");
    }

    if (other && tq_view_capacity != other->tq_view_capacity) {
        throw std::invalid_argument("llama_kv_cache: shared caches must use the same tq_view_capacity");
    }

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    // Auto-asymmetric: when symmetric turbo K+V is requested and the model has
    // high GQA ratio (few KV heads serving many Q heads), upgrade K to q8_0.
    // Turbo K quantization error gets amplified by the GQA broadcast factor.
    // Qwen2.5: 4 KV heads / 28 Q heads = 7:1 → turbo3 K PPL catastrophic (2887 vs 7.4 baseline)
    // Mistral:  8 KV heads / 32 Q heads = 4:1 → turbo3 K works fine (+4.4% PPL)
    // Threshold: GQA ratio >= 6 triggers auto-asymmetric.
    {
        const bool k_is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);
        if (k_is_turbo) {
            const uint32_t n_head    = hparams.n_head(0);
            const uint32_t n_head_kv = hparams.n_head_kv(0);
            const uint32_t gqa_ratio = (n_head_kv > 0) ? n_head / n_head_kv : 1;

            const char * env = getenv("TURBO_AUTO_ASYMMETRIC");
            const bool disabled = (env && env[0] == '0');

            if (!disabled && gqa_ratio >= 6 && type_k == type_v) {
                LLAMA_LOG_WARN("%s: auto-asymmetric: GQA ratio %u:1 (n_head=%u, n_head_kv=%u) — "
                               "upgrading K from %s to q8_0 to prevent quality degradation. "
                               "Disable with TURBO_AUTO_ASYMMETRIC=0\n",
                               __func__, gqa_ratio, n_head, n_head_kv, ggml_type_name(type_k));
                type_k = GGML_TYPE_Q8_0;
            }
        }
    }

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // +3 for turbo rotation matrices (turbo_rotation + turbo_rotation_inv + turbo_innerq_scale_inv)
                /*.mem_size   =*/ size_t(((tq_view_capacity + 1u)*(1 + n_stream)*n_layer +
                    (tq_execution == LLAMA_TQ_EXEC_RESIDUAL_PARITY ? n_layer : 0u) + 3)*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (n_embd_head_v_all == 0) {
            n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
        } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
            n_embd_head_v_all = -1;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        // TurboQuant zero-padding: for models with non-128-aligned head_dim (e.g. DeepSeek
        // head_dim_k=192), pad each head to the next multiple of 128. The padded zeros don't
        // affect dot products since WHT preserves inner products:
        //   <WHT(Q_padded), WHT(K_padded)> = <Q_padded, K_padded> = <Q, K> + <0, 0> = <Q, K>
        const uint32_t n_embd_head_k = hparams.n_embd_head_k(il);


        const bool has_k = true;
        const bool has_v = !is_mla;

        // Layer-adaptive: use higher precision for quality-sensitive layers
        // Config: TURBO_LAYER_ADAPTIVE env var controls the strategy
        //   0 = uniform (default)
        //   1 = q8_0 K+V for first+last 4 layers
        //   2 = q8_0 K+V for last 8 layers
        //   5 = Boundary V: first2+last2 V=turbo4, rest V=turbo2 (K unchanged)
        //   6 = V-only: last 8 V=turbo4, rest V=turbo2 (K unchanged)
        //   7 = Boundary V (recommended): first2+last2 V=q8_0, rest V=turbo2 (K unchanged)
        ggml_type layer_type_k = type_k;
        ggml_type layer_type_v = type_v;
        {
            static const int adaptive_mode = [&]() {
                const char * env = getenv("TURBO_LAYER_ADAPTIVE");
                if (env) {
                    int mode = atoi(env);
                    if (mode > 0) {
                        LLAMA_LOG_INFO("llama_kv_cache: layer-adaptive mode %d enabled (env)\n", mode);
                    }
                    return mode;
                }
                // Auto-enable Boundary V (mode 7) when V is turbo2
                if (type_v == GGML_TYPE_TURBO2_0 && hparams.n_layer() >= 8) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V auto-enabled for turbo2-V (opt-out: TURBO_LAYER_ADAPTIVE=0)\n");
                    return 7;
                }
                return 0;
            }();
            const bool is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);
            const bool v_is_turbo = (type_v == GGML_TYPE_TURBO3_0 || type_v == GGML_TYPE_TURBO4_0 || type_v == GGML_TYPE_TURBO2_0);
            const uint32_t n_layer = hparams.n_layer();
            if (adaptive_mode == 1 && is_turbo && n_layer >= 8) {
                if (il < 4 || il >= n_layer - 4) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 2 && is_turbo && n_layer >= 8) {
                if (il >= n_layer - 8) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 5 && v_is_turbo && n_layer >= 8) {
                // Boundary V (turbo4 boundaries): first2+last2 V=turbo4, rest V=turbo2
                const bool is_boundary = (il < 2 || il >= n_layer - 2);
                layer_type_v = is_boundary ? GGML_TYPE_TURBO4_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V mode 5: first2+last2 V=turbo4, rest V=turbo2\n");
                }
            } else if (adaptive_mode == 6 && v_is_turbo && n_layer >= 8) {
                // V-only: last 8 V=turbo4, rest V=turbo2
                layer_type_v = (il >= n_layer - 8) ? GGML_TYPE_TURBO4_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: V-only LA mode 6: last8 V=turbo4, rest V=turbo2\n");
                }
            } else if (adaptive_mode == 7 && v_is_turbo && n_layer >= 8) {
                // Boundary V (recommended): first2+last2 V=q8_0, rest V=turbo2
                const bool is_boundary = (il < 2 || il >= n_layer - 2);
                layer_type_v = is_boundary ? GGML_TYPE_Q8_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V mode 7: first2+last2 V=q8_0, rest V=turbo2\n");
                }
            }
        }
        std::shared_ptr<residual_layer_state> residual;
        if (tq_execution == LLAMA_TQ_EXEC_RESIDUAL_PARITY) {
            const auto & residual_metadata = model.turboquant_metadata.residual_parity.layers.at(il);
            residual = std::make_shared<residual_layer_state>();
            residual->logical_channels = n_embd_k_gqa;
            residual->row_bytes =
                (n_embd_k_gqa * TQ_RESIDUAL_PHYSICAL_BITS_PER_CHANNEL + 7u) / 8u;
            residual->head_dim = n_embd_head_k;
            residual->n_head = hparams.n_head_kv(il);
            residual->scales = residual_metadata.fixed_sector_scales;
            residual->beta = residual_metadata.beta;
            const uint64_t physical_rows = static_cast<uint64_t>(kv_size) * n_stream;
            const uint64_t actual_bits =
                (physical_rows * residual->row_bytes + TQ_RESIDUAL_CONTROLLER_BYTES) * 8u;
            const uint64_t target_bits = physical_rows * residual->logical_channels *
                TQ_RESIDUAL_LOGICAL_BITS_PER_CHANNEL;
            if (actual_bits > target_bits) {
                throw std::runtime_error(
                    "residual-parity physical K storage exceeds the controller-inclusive five-bit budget");
            }
            for (uint32_t view = 0; view < 3; ++view) {
                auto * rotation = model.turboquant_rotation(il, view);
                if (!rotation || rotation->type != GGML_TYPE_F32 ||
                    rotation->ne[0] != n_embd_head_k || rotation->ne[1] != n_embd_head_k || !rotation->buffer) {
                    throw std::runtime_error("residual-parity requires loaded F32 per-head rotations");
                }
                residual->rotations[view].resize(static_cast<size_t>(n_embd_head_k) * n_embd_head_k);
                ggml_backend_tensor_get(rotation, residual->rotations[view].data(), 0,
                    residual->rotations[view].size() * sizeof(float));
            }
            auto write_u32 = [&](size_t & offset, uint32_t value) {
                for (uint32_t byte = 0; byte < 4; ++byte) {
                    residual->controller[offset++] = static_cast<uint8_t>(value >> (byte * 8));
                }
            };
            auto write_f32 = [&](size_t & offset, float value) {
                uint32_t bits;
                static_assert(sizeof(bits) == sizeof(value));
                std::memcpy(&bits, &value, sizeof(bits));
                write_u32(offset, bits);
            };
            size_t controller_offset = 0;
            residual->controller[controller_offset++] = 'T';
            residual->controller[controller_offset++] = 'Q';
            residual->controller[controller_offset++] = 'R';
            residual->controller[controller_offset++] = 'P';
            write_u32(controller_offset, LLAMA_KV_STATE_RESIDUAL_VERSION);
            write_u32(controller_offset, residual->logical_channels);
            write_u32(controller_offset, residual->row_bytes);
            residual->controller[controller_offset++] = 3u;
            residual->controller[controller_offset++] = 1u;
            residual->controller[controller_offset++] = 1u;
            residual->controller[controller_offset++] = TQ_RESIDUAL_PARITY_COUPLED_LAYOUT;
            for (float value : residual->scales) {
                write_f32(controller_offset, value);
            }
            for (float value : residual->beta) {
                write_f32(controller_offset, value);
            }
            write_u32(controller_offset, TQ_RESIDUAL_LOGICAL_BITS_PER_CHANNEL * 1000u);
            GGML_ASSERT(controller_offset == 44u);
            layer_type_k = GGML_TYPE_I8;
        }
        // For turbo types, pad K head_dim to next multiple of 128 for full WHT groups
        uint32_t n_embd_k_gqa_eff = n_embd_k_gqa;
        const bool k_is_turbo = (layer_type_k == GGML_TYPE_TURBO3_0 || layer_type_k == GGML_TYPE_TURBO4_0 || layer_type_k == GGML_TYPE_TURBO2_0);
        if (k_is_turbo && n_embd_head_k % 128 != 0) {
            const uint32_t padded_head_k = ((n_embd_head_k + 127) / 128) * 128;
            const uint32_t n_head_kv = n_embd_k_gqa / n_embd_head_k;
            n_embd_k_gqa_eff = n_head_kv * padded_head_k;
            if (il == 0) {
                LLAMA_LOG_INFO("%s: turbo zero-padding K head_dim %u -> %u (cache %u -> %u)\n",
                               __func__, n_embd_head_k, padded_head_k, n_embd_k_gqa, n_embd_k_gqa_eff);
            }
        }
        if (residual) {
            n_embd_k_gqa_eff = residual->row_bytes;
        }

        // For turbo types, pad V head_dim to next multiple of 128 if needed
        const uint32_t n_embd_head_v = hparams.n_embd_head_v(il);
        uint32_t n_embd_v_gqa_eff = n_embd_v_gqa;
        const bool v_is_turbo = (layer_type_v == GGML_TYPE_TURBO3_0 || layer_type_v == GGML_TYPE_TURBO4_0 || layer_type_v == GGML_TYPE_TURBO2_0);
        if (v_is_turbo && !is_mla && n_embd_head_v % 128 != 0) {
            const uint32_t padded_head_v = ((n_embd_head_v + 127) / 128) * 128;
            const uint32_t n_head_kv = n_embd_v_gqa / n_embd_head_v;
            n_embd_v_gqa_eff = n_head_kv * padded_head_v;
            if (il == 0) {
                LLAMA_LOG_INFO("%s: turbo zero-padding V head_dim %u -> %u (cache %u -> %u)\n",
                               __func__, n_embd_head_v, padded_head_v, n_embd_v_gqa, n_embd_v_gqa_eff);
            }
        }

        std::array<ggml_tensor *, 3> k_views = {};
        for (uint32_t view = 0; view < tq_view_capacity; ++view) {
            k_views[view] = has_k ? ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa_eff, kv_size, n_stream) : nullptr;
            if (has_k) {
                if (view == 0) {
                    ggml_format_name(k_views[view], "cache_k_l%d", il);
                } else {
                    ggml_format_name(k_views[view], "cache_k_l%d_v%u", il, view);
                }
            }
        }
        ggml_tensor * k = k_views[0];
        ggml_tensor * residual_controller = residual ?
            ggml_new_tensor_1d(ctx, GGML_TYPE_I8, TQ_RESIDUAL_CONTROLLER_BYTES) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_gqa_eff, kv_size, n_stream) : nullptr;

        residual_controller && ggml_format_name(residual_controller, "cache_k_residual_controller_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
        std::array<std::vector<ggml_tensor *>, 3> k_stream_views;

        for (uint32_t s = 0; s < n_stream; ++s) {
            for (uint32_t view = 0; view < tq_view_capacity; ++view) {
                auto * k_view = k_views[view];
                k_stream_views[view].push_back(has_k ? ggml_view_2d(ctx, k_view, n_embd_k_gqa_eff, kv_size, k_view->nb[1], s*k_view->nb[2]) : nullptr);
            }
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa_eff, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        k_stream = k_stream_views[0];

        map_layer_ids[il] = layers.size();

        kv_layer layer;
        layer.il = il;
        layer.k = k;
        layer.v = v;
        layer.k_stream = std::move(k_stream);
        layer.v_stream = std::move(v_stream);
        layer.k_views = k_views;
        layer.k_stream_views = std::move(k_stream_views);
        layer.residual = std::move(residual);
        layer.residual_controller = residual_controller;
        layers.push_back(std::move(layer));

        // TurboQuant: create rotation matrix tensors (once, shared across layers)
        if (turbo_rotation == nullptr &&
            (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0)) {
            turbo_rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
            ggml_format_name(turbo_rotation, "turbo_rotation");  // R^T
            turbo_rotation_inv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
            ggml_format_name(turbo_rotation_inv, "turbo_rotation_inv");  // R

            // InnerQ: per-channel scale_inv tensor (128 floats, initialized to all 1.0)
            turbo_innerq_scale_inv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, INNERQ_MAX_CHANNELS);
            ggml_format_name(turbo_innerq_scale_inv, "turbo_innerq_scale_inv");
        }
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);

        for (const auto & layer : layers) {
            if (layer.residual && layer.residual_controller && layer.residual_controller->buffer == buf && !model.hparams.no_alloc) {
                ggml_backend_tensor_set(layer.residual_controller, layer.residual->controller.data(), 0, layer.residual->controller.size());
            }
        }

        // Fill turbo rotation matrices AFTER buffer clear (clear zeroes everything)
        if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr && !model.hparams.no_alloc) {
            #include "turbo-rotation-data.h"
            // ggml is column-major; C arrays are row-major. Storing a row-major matrix
            // into ggml implicitly transposes it. ggml_mul_mat(A, x) computes A^T @ x.
            // To get R @ q: store R^T → ggml sees (R^T)^T_col = R → mul_mat gives R @ q. Wait no —
            // store R so ggml col-major reads it as R^T, then mul_mat gives (R^T)^T = R. ✓
            // Store R for Q forward rotation, R^T for V inverse rotation
            // ggml_mul_mat(A,x) computes A@x for row-major stored A (verified by test)
            ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));

            // Initialize InnerQ scale_inv to all 1.0 (identity scaling)
            if (turbo_innerq_scale_inv != nullptr && turbo_innerq_scale_inv->buffer != nullptr) {
                float ones[INNERQ_MAX_CHANNELS];
                for (int i = 0; i < INNERQ_MAX_CHANNELS; i++) ones[i] = 1.0f;
                ggml_backend_tensor_set(turbo_innerq_scale_inv, ones, 0, INNERQ_MAX_CHANNELS * sizeof(float));
            }

            LLAMA_LOG_INFO("%s: TurboQuant rotation matrices initialized (128x128)\n", __func__);
        }
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TurboQuant keeps attention rotation enabled by default on supported K/V cache sides.
    // The official shared-cache path is preserved: views inherit the source cache rotation tensors.
    // LLAMA_ATTN_ROT_DISABLE remains a hard lock-out; per-side overrides can opt supported sides in or out.
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        const bool attn_rot_k_supported =
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        const bool attn_rot_k_deepseek_indexer =
            (model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4) &&
            hparams.n_embd_head_k_full == hparams.indexer_head_size;

        attn_rot_k =
            !attn_rot_disable &&
            (attn_rot_k_supported || attn_rot_k_deepseek_indexer);

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;

        const char * ROT_K_OV = getenv("LLAMA_ATTN_ROT_K_OVERRIDE");
        if (ROT_K_OV && !attn_rot_disable) {
            attn_rot_k = (atoi(ROT_K_OV) != 0) && (attn_rot_k_supported || attn_rot_k_deepseek_indexer);
        }

        const char * ROT_V_OV = getenv("LLAMA_ATTN_ROT_V_OVERRIDE");
        if (ROT_V_OV && !attn_rot_disable) {
            attn_rot_v = (atoi(ROT_V_OV) != 0) &&
                n_embd_head_v_all > 0 &&
                ggml_is_quantized(type_v) &&
                hparams.n_embd_head_v() % 64 == 0;
        }
    }
    if (tq_execution == LLAMA_TQ_EXEC_RESIDUAL_PARITY) {
        attn_rot_k = false;
    }
    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
    turboquant_cfg = llama_turboquant_runtime_from_env();
    if (turboquant_cfg.enabled) {
        LLAMA_LOG_INFO(
            "%s: TurboQuant enabled (mode=%s, so8=%d, so8_learned=%d, triality=%d, mix=%.3f, seed=%u)\n",
            __func__,
            turboquant_cfg.mode.c_str(),
            turboquant_cfg.so8_enabled ? 1 : 0,
            turboquant_cfg.so8_learned ? 1 : 0,
            turboquant_cfg.triality_enabled ? 1 : 0,
            turboquant_cfg.triality_mix,
            turboquant_cfg.rotation_seed);
    }
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }

        for (const auto & layer : layers) {
            if (layer.residual && layer.residual_controller && layer.residual_controller->buffer) {
                ggml_backend_tensor_set(layer.residual_controller, layer.residual->controller.data(), 0, layer.residual->controller.size());
            }
        }

        // Re-initialize turbo rotation matrices after buffer clear (clear zeroes everything)
        if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr) {
            #include "turbo-rotation-data.h"
            ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));

            // Re-initialize InnerQ scale_inv to all 1.0
            if (turbo_innerq_scale_inv != nullptr && turbo_innerq_scale_inv->buffer != nullptr) {
                float ones[INNERQ_MAX_CHANNELS];
                for (int i = 0; i < INNERQ_MAX_CHANNELS; i++) ones[i] = 1.0f;
                ggml_backend_tensor_set(turbo_innerq_scale_inv, ones, 0, INNERQ_MAX_CHANNELS * sizeof(float));
            }
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                for (uint32_t view = 0; view < tq_view_capacity; ++view) {
                    ggml_backend_tensor_copy(layer.k_stream_views[view][ssrc], layer.k_stream_views[view][sdst]);
                }

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> result;
    result.reserve(layers.size());
    for (const auto & layer : layers) {
        result.push_back(layer.il);
    }
    return result;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);
    return layers[ikv].k;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    return get_k(ctx, il, n_kv, sinfo, 0);
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo, uint32_t view) const {
    GGML_ASSERT(view < tq_view_capacity);

    const int32_t ikv = map_layer_ids.at(il);

    const auto & layer = layers[ikv];
    if (layer.residual) {
        GGML_ASSERT(view == 0);
        auto * k = layer.k_views[0];
        const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
        ggml_tensor * packed = ggml_view_3d(ctx, k,
            layer.residual->row_bytes, n_kv, ns,
            k->nb[1], k->nb[2], k->nb[2] * sinfo.s0);
        ggml_tensor * seed = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
            layer.residual->head_dim, layer.residual->n_head, n_kv, ns);
        return ggml_map_custom2(ctx, seed, packed, residual_decode_op, 1, layer.residual.get());
    }

    auto * k = layer.k_views[view];

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    // For turbo-padded caches, n_embd_k_gqa may be larger than hparams value
    const bool k_is_turbo = (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0);
    if (k_is_turbo) {
        assert(n_embd_k_gqa >= hparams.n_embd_k_gqa(il));
    } else {
        assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));
    }

    // Use padded head_dim for turbo types so the full padded data is returned
    const uint32_t head_k = hparams.n_embd_head_k(il);
    const uint32_t head_k_eff = (k_is_turbo && head_k % 128 != 0)
        ? ((head_k + 127) / 128) * 128 : head_k;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            head_k_eff, hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, head_k_eff),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE] — for turbo-padded V, cache may be larger
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    // Use padded head_dim for turbo types
    const bool v_is_turbo = (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0);
    const uint32_t head_v = hparams.n_embd_head_v(il);
    const uint32_t head_v_eff = (v_is_turbo && head_v % 128 != 0)
        ? ((head_v + 127) / 128) * 128 : head_v;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                head_v_eff, hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, head_v_eff),                      // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                    // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),            // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), head_v_eff, ns,
            ggml_row_size(v->type, kv_size*head_v_eff),              // v->nb[1]
            ggml_row_size(v->type, kv_size),                         // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),            // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    return cpy_k(ctx, k_cur, k_idxs, il, sinfo, 0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo, uint32_t view) const {
    GGML_UNUSED(sinfo);
    GGML_ASSERT(view < tq_view_capacity);

    const int32_t ikv = map_layer_ids.at(il);

    const auto & layer = layers[ikv];
    if (layer.residual) {
        GGML_ASSERT(view == 0 && k_cur->type == GGML_TYPE_F32);
        GGML_ASSERT(k_cur->ne[0] == layer.residual->head_dim && k_cur->ne[1] == layer.residual->n_head);
        ggml_tensor * seed = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, layer.residual->row_bytes, k_cur->ne[2]);
        ggml_tensor * encoded = ggml_map_custom2(ctx, seed, k_cur, residual_encode_op, 1, layer.residual.get());
        ggml_tensor * k = layer.k_views[0];
        if (k->ne[2] > 1) {
            k = ggml_reshape_2d(ctx, k, layer.residual->row_bytes, get_size() * k->ne[2]);
        }
        return ggml_map_custom3_inplace(ctx, k, encoded, k_idxs, residual_store_rows_op, 1, nullptr);
    }

    ggml_tensor * k = layer.k_views[view];

    int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    // Turbo zero-padding: pad each head to next multiple of 128 before merging dims.
    // k_cur shape here is (n_embd_head, n_head, n_tokens).
    // ggml_pad pads ne[0] with zeros — exactly what we need per-head.
    const bool k_is_turbo = (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0);
    const bool k_needs_pad = k_is_turbo && (n_embd_head % 128 != 0);
    if (k_needs_pad) {
        const int64_t pad_amount = ((n_embd_head + 127) / 128) * 128 - n_embd_head;
        k_cur = ggml_pad(ctx, k_cur, pad_amount, 0, 0, 0);
        n_embd_head = k_cur->ne[0];  // now 128-aligned
    }

    int64_t n_embd_gqa = n_embd_head * n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    const bool qwen_full_attention_layer = !hparams.is_recr(il);
    const bool turboquant_k_enabled = qwen_full_attention_layer && llama_turboquant_runtime_allows_k(turboquant_cfg);
    if (turboquant_k_enabled && !turboquant_logged_k) {
        LLAMA_LOG_INFO("%s: TurboQuant K-path active at layer %d (mode=%s)\n", __func__, il, turboquant_cfg.mode.c_str());
        turboquant_logged_k = true;
    }

    // store the current K values into the cache
    ggml_tensor * result = ggml_set_rows(ctx, k, k_cur, k_idxs);

    // For turbo: store WHT group size in op_params so the CUDA kernel knows.
    // With zero-padding, all groups are always full 128-element WHT groups.
    if (k_is_turbo) {
        int32_t wht_group = 128;  // always 128 with padding
        memcpy(result->op_params, &wht_group, sizeof(int32_t));
    }

    return result;
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    // Turbo zero-padding: pad V head_dim to next multiple of 128
    const bool v_is_turbo = (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0);
    const bool v_needs_pad = v_is_turbo && (n_embd_head % 128 != 0);
    if (v_needs_pad) {
        const int64_t pad_amount = ((n_embd_head + 127) / 128) * 128 - n_embd_head;
        v_cur = ggml_pad(ctx, v_cur, pad_amount, 0, 0, 0);
        n_embd_head = v_cur->ne[0];  // now 128-aligned
    }

    int64_t n_embd_gqa = n_embd_head * n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    const bool qwen_full_attention_layer = !hparams.is_recr(il);
    const bool turboquant_v_enabled = qwen_full_attention_layer && llama_turboquant_runtime_allows_v(turboquant_cfg);
    if (turboquant_v_enabled && !turboquant_logged_v) {
        LLAMA_LOG_INFO("%s: TurboQuant V-path active at layer %d (mode=%s)\n", __func__, il, turboquant_cfg.mode.c_str());
        turboquant_logged_v = true;
    }

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        ggml_tensor * result = ggml_set_rows(ctx, v, v_cur, v_idxs);
        // With zero-padding, all groups are always full 128-element WHT groups
        if (v_is_turbo) {
            int32_t wht_group = 128;  // always 128 with padding
            memcpy(result->op_params, &wht_group, sizeof(int32_t));
        }
        return result;
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        // EXPERIMENT (master TODO): force smallest rotation matrix (nrot=64)
        // for K, mirroring V's choice. Master defaults to the largest power-of-2
        // that divides head_dim, but the upstream comment hypothesizes smaller
        // tiles preserve more local structure → less PPL hit on sensitive models
        // (gemma-4 26B-A4B reportedly regresses with the largest tile).
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        const char * LLAMA_ATTN_ROT_K_NROT = getenv("LLAMA_ATTN_ROT_K_NROT");
        int nrot = LLAMA_ATTN_ROT_K_NROT ? atoi(LLAMA_ATTN_ROT_K_NROT) : 64;

        // Original master behavior (largest power-of-2): set LLAMA_ATTN_ROT_K_NROT=0
        if (nrot == 0) {
            nrot = 64;
            do {
                nrot *= 2;
            } while (n_embd_head_k_all % nrot == 0);
            nrot /= 2;
        }

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::tq_get_residual_storage_info(uint32_t il, tq_residual_storage_info & info) const {
    info = {};
    const auto it = std::find_if(layers.begin(), layers.end(),
        [il](const kv_layer & layer) { return layer.il == il; });
    if (it == layers.end() || !it->residual || !it->residual_controller || !it->k_views[0]) {
        return false;
    }
    for (const ggml_tensor * view : it->k_views) {
        info.physical_view_count += view != nullptr ? 1u : 0u;
    }
    info.logical_channels = it->residual->logical_channels;
    info.row_bytes = it->residual->row_bytes;
    info.physical_payload_bytes = ggml_nbytes(it->k_views[0]);
    info.controller_bytes = ggml_nbytes(it->residual_controller);
    const size_t row_count = ggml_nrows(it->k_views[0]);
    const double logical_values = static_cast<double>(row_count) * info.logical_channels;
    info.logical_payload_bits_per_channel = TQ_RESIDUAL_LOGICAL_BITS_PER_CHANNEL;
    info.physical_payload_bits_per_channel =
        static_cast<double>(info.physical_payload_bytes) * 8.0 / logical_values;
    info.actual_bits_per_channel =
        static_cast<double>(info.physical_payload_bytes + info.controller_bytes) * 8.0 / logical_values;
    info.five_bit_target_met = info.actual_bits_per_channel <= TQ_RESIDUAL_LOGICAL_BITS_PER_CHANNEL;

    std::array<uint8_t, TQ_RESIDUAL_CONTROLLER_BYTES> controller{};
    ggml_backend_tensor_get(it->residual_controller, controller.data(), 0, controller.size());
    info.controller_reserved_zero = std::all_of(controller.begin() + 44, controller.end(),
        [](uint8_t value) { return value == 0; });

    std::vector<uint8_t> row(info.row_bytes);
    info.rows_canonical = true;
    for (size_t index = 0; index < row_count; ++index) {
        ggml_backend_tensor_get(it->k_views[0], row.data(), index * info.row_bytes, row.size());
        if (!tq_residual_row_valid(row.data(), info.logical_channels, info.row_bytes)) {
            info.rows_canonical = false;
            break;
        }
    }
    return true;
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        for (uint32_t view = 0; view < tq_view_capacity; ++view) {
            size_k_bytes += ggml_nbytes(layer.k_views[view]);
        }
        if (layer.residual_controller) {
            size_k_bytes += ggml_nbytes(layer.residual_controller);
        }
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * tq_rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        const bool is_turbo = cur->type == GGML_TYPE_TURBO2_0 ||
            cur->type == GGML_TYPE_TURBO3_0 || cur->type == GGML_TYPE_TURBO4_0;
        if (is_turbo) {
            tmp = ggml_turbo_wht(ctx, tmp, 1, 128, turbo_innerq_scale_inv);
        } else {
            tmp = ggml_mul_mat_aux(ctx, tmp, rot);
        }

        const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
        const int64_t pad = cur->ne[0] - n_embd_head_k;
        if (pad > 0) {
            tmp = ggml_view_3d(ctx, tmp, n_embd_head_k, tmp->ne[1], tmp->ne[2], tmp->nb[1], tmp->nb[2], 0);
        }

        if (tq_rot) {
            tmp = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, tq_rot)), tmp);
        }

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        if (tq_rot) {
            tmp = ggml_mul_mat(ctx, tq_rot, tmp);
        }

        if (pad > 0) {
            tmp = ggml_pad(ctx, tmp, pad, 0, 0, 0);
        }

        if (is_turbo) {
            tmp = ggml_turbo_wht(ctx, tmp, 0, 128, turbo_innerq_scale_inv);
        } else {
            tmp = ggml_mul_mat_aux(ctx, tmp, rot);
        }

        tmp = ggml_cpy(ctx, tmp, cur);
    } else if (tq_rot) {
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);
        tmp = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, tq_rot)), tmp);
        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
        tmp = ggml_mul_mat(ctx, tq_rot, tmp);
        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();
    const auto * tq_config = lctx->get_turboquant_config();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv = hparams.n_head_kv(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        if (layer.residual) {
            const int64_t n_rows = static_cast<int64_t>(get_size()) * n_stream;
            ggml_tensor * packed = ggml_reshape_2d(ctx, layer.k_views[0], layer.residual->row_bytes, n_rows);
            ggml_tensor * decode_seed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                layer.residual->head_dim, layer.residual->n_head, n_rows);
            ggml_tensor * decoded = ggml_map_custom2(
                ctx, decode_seed, packed, residual_decode_op, 1, layer.residual.get());
            ggml_tensor * shifted = build_rope_shift(
                cparams, ctx, decoded, inp->k_shift, inp->k_rot, nullptr,
                rope_factors, freq_base_l, freq_scale_l, il);
            ggml_tensor * encode_seed = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, layer.residual->row_bytes, n_rows);
            ggml_tensor * encoded = ggml_map_custom2(
                ctx, encode_seed, shifted, residual_encode_op, 1, layer.residual.get());
            ggml_build_forward_expand(gf, ggml_cpy(ctx, encoded, packed));
            continue;
        }

        const bool tq_attention_consensus = tq_config &&
            tq_config->execution == LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS;
        const uint32_t views_to_shift = tq_attention_consensus ? tq_view_capacity : 1u;
        uint32_t selected_view = 0;
        if (tq_config && !tq_attention_consensus && layer.il < tq_config->layers.size()) {
            const auto & tq_layer = tq_config->layers[layer.il];
            float selected_error = std::numeric_limits<float>::infinity();
            for (uint32_t branch = 0; branch < 3; ++branch) {
                if ((tq_layer.active_mask & (uint8_t(1) << branch)) == 0) {
                    continue;
                }
                if (tq_config->execution == LLAMA_TQ_EXEC_SINGLE_VIEW ||
                    tq_layer.branches[branch].expected_error < selected_error) {
                    selected_view = static_cast<uint32_t>(tq_layer.branches[branch].view);
                    selected_error = tq_layer.branches[branch].expected_error;
                    if (tq_config->execution == LLAMA_TQ_EXEC_SINGLE_VIEW) {
                        break;
                    }
                }
            }
        }

        for (uint32_t view = 0; view < views_to_shift; ++view) {
            auto * k_storage = layer.k_views[view];
            const bool k_is_turbo = k_storage->type == GGML_TYPE_TURBO2_0 ||
                k_storage->type == GGML_TYPE_TURBO3_0 || k_storage->type == GGML_TYPE_TURBO4_0;
            const int64_t n_embd_head_eff = k_is_turbo && n_embd_head_k % 128 != 0
                ? ((n_embd_head_k + 127) / 128) * 128 : n_embd_head_k;
            const int64_t n_embd_k_gqa_eff = n_embd_head_eff * n_head_kv;
            ggml_tensor * k =
                ggml_view_3d(ctx, k_storage,
                    k_is_turbo ? n_embd_head_eff : n_rot, n_head_kv, get_size()*n_stream,
                    ggml_row_size(k_storage->type, n_embd_head_eff),
                    ggml_row_size(k_storage->type, n_embd_k_gqa_eff),
                    k_is_turbo ? 0 : ggml_row_size(k_storage->type, n_embd_nope));
            ggml_tensor * cur = build_rope_shift(
                cparams, ctx, k, inp->k_shift, inp->k_rot,
                tq_attention_consensus ? model.turboquant_rotation(il, view) :
                    (tq_config ? model.turboquant_rotation(il, selected_view) : nullptr),
                rope_factors, freq_base_l, freq_scale_l, il);

            ggml_build_forward_expand(gf, cur);
        }
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    if (tq_execution == LLAMA_TQ_EXEC_RESIDUAL_PARITY) {
        const uint32_t marker = LLAMA_KV_STATE_RESIDUAL_MARKER;
        const uint32_t version = LLAMA_KV_STATE_RESIDUAL_VERSION;
        const uint32_t controller_bytes = TQ_RESIDUAL_CONTROLLER_BYTES;
        io.write(&marker, sizeof(marker));
        io.write(&version, sizeof(version));
        io.write(&controller_bytes, sizeof(controller_bytes));
        for (const auto & layer : layers) {
            io.write(layer.residual->controller.data(), layer.residual->controller.size());
        }
    } else if (tq_view_capacity == 3) {
        const uint32_t k_view_marker = LLAMA_KV_STATE_K_VIEW_MARKER;
        io.write(&k_view_marker, sizeof(k_view_marker));
        io.write(&tq_view_capacity, sizeof(tq_view_capacity));
    }

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        for (uint32_t view = 0; view < tq_view_capacity; ++view) {
            auto * k = layer.k_stream_views[view][cr.strm];

            const uint32_t n_embd_k_gqa = (uint32_t) k->ne[0];
            const int32_t k_type_i = (int32_t) k->type;
            io.write(&k_type_i, sizeof(k_type_i));

            const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
            io.write(&k_size_row, sizeof(k_size_row));

            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * k_size_row;
                io.write_tensor(k, range.first * k_size_row, buf_size);
            }
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Use actual tensor width (may be padded for turbo types)
            const uint32_t n_embd_v_gqa = (uint32_t) v->ne[0];

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (tq_execution == LLAMA_TQ_EXEC_RESIDUAL_PARITY) {
        uint32_t marker;
        uint32_t version;
        uint32_t controller_bytes;
        io.read(&marker, sizeof(marker));
        io.read(&version, sizeof(version));
        io.read(&controller_bytes, sizeof(controller_bytes));
        if (marker != LLAMA_KV_STATE_RESIDUAL_MARKER || version != LLAMA_KV_STATE_RESIDUAL_VERSION ||
            controller_bytes != TQ_RESIDUAL_CONTROLLER_BYTES) {
            LLAMA_LOG_ERROR("%s: incompatible residual-parity state marker/version/controller size\n", __func__);
            return false;
        }
        for (const auto & layer : layers) {
            std::array<uint8_t, TQ_RESIDUAL_CONTROLLER_BYTES> controller{};
            io.read(controller.data(), controller.size());
            if (controller != layer.residual->controller) {
                LLAMA_LOG_ERROR("%s: residual-parity controller differs from model metadata at layer %u\n", __func__, layer.il);
                return false;
            }
            ggml_backend_tensor_set(layer.residual_controller, controller.data(), 0, controller.size());
        }
    } else if (tq_view_capacity == 3) {
        uint32_t k_view_marker;
        uint32_t stored_view_capacity;
        io.read(&k_view_marker, sizeof(k_view_marker));
        io.read(&stored_view_capacity, sizeof(stored_view_capacity));

        if (k_view_marker != LLAMA_KV_STATE_K_VIEW_MARKER || stored_view_capacity != tq_view_capacity) {
            LLAMA_LOG_ERROR("%s: mismatched K view capacity marker (marker 0x%08x, views %u instead of %u)\n",
                    __func__, k_view_marker, stored_view_capacity, tq_view_capacity);
            return false;
        }
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        for (uint32_t view = 0; view < tq_view_capacity; ++view) {
            auto * k = layer.k_stream_views[view][strm];

            const uint32_t n_embd_k_gqa = (uint32_t) k->ne[0];
            int32_t k_type_i_ref;
            io.read(&k_type_i_ref, sizeof(k_type_i_ref));

            if (tq_view_capacity == 1 && ((uint32_t) k_type_i_ref == LLAMA_KV_STATE_K_VIEW_MARKER ||
                (uint32_t) k_type_i_ref == LLAMA_KV_STATE_RESIDUAL_MARKER)) {
                uint32_t stored_view_capacity;
                io.read(&stored_view_capacity, sizeof(stored_view_capacity));
                LLAMA_LOG_ERROR("%s: mismatched K view capacity (%u instead of %u)\n", __func__, stored_view_capacity, tq_view_capacity);
                return false;
            }

            const int32_t k_type_i = (int32_t) k->type;
            if (k_type_i != k_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d, view %u)\n", __func__, k_type_i, k_type_i_ref, il, view);
                return false;
            }

            uint64_t k_size_row_ref;
            io.read(&k_size_row_ref, sizeof(k_size_row_ref));
            const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
            if (k_size_row != k_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d, view %u)\n", __func__, k_size_row, (size_t) k_size_row_ref, il, view);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
                } else {
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                        io.read_tensor(k, dst_offset, k_size_row);
                    }
                }
            }

            if (layer.residual && cell_count) {
                std::vector<uint8_t> row(layer.residual->row_bytes);
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const uint32_t cell = sinfo.is_contiguous() ? sinfo.head() + i : sinfo.idxs[0][i];
                    ggml_backend_tensor_get(k, row.data(), static_cast<size_t>(cell) * layer.residual->row_bytes, row.size());
                    if (!tq_residual_row_valid(row.data(), layer.residual->logical_channels, layer.residual->row_bytes)) {
                        LLAMA_LOG_ERROR("%s: residual-parity row has a reserved code or non-zero padding bits\n", __func__);
                        return false;
                    }
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Use actual tensor width (may be padded for turbo types)
            const uint32_t n_embd_v_gqa = (uint32_t) v->ne[0];

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    // InnerQ: check if CUDA calibration finalized and tensor needs update
    if (kv->get_turbo_innerq_scale_inv() != nullptr && turbo_innerq_needs_tensor_update()) {
        ggml_tensor * t = kv->get_turbo_innerq_scale_inv();
        if (t->buffer != nullptr) {
            ggml_backend_tensor_set(t, g_innerq_scale_inv_host, 0, INNERQ_MAX_CHANNELS * sizeof(float));
            turbo_innerq_mark_tensor_updated();
            LLAMA_LOG_INFO("%s: InnerQ scale_inv tensor updated\n", __func__);
        }
    }

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il, uint32_t view) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur], view);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation_inv() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rot_forward() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rot_inverse() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::get_turbo_innerq_scale_inv() const {
    return kv->get_turbo_innerq_scale_inv();
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, uint32_t view) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur], view);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}
