#include "testing.h"

#include "../ggml/include/ggml.h"
#include "../ggml/src/ggml-quants.h"
#include "../src/llama-turboquant.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

extern "C" void ggml_vec_dot_tq4_1s_q8_0(
    int n,
    float * s,
    size_t bs,
    const void * vx,
    size_t bx,
    const void * vy,
    size_t by,
    int nrc);

std::vector<float> make_wave_values(uint32_t count, float phase) {
    std::vector<float> values(count, 0.0f);
    for (uint32_t i = 0; i < count; ++i) {
        values[i] = 0.75f * std::sin(phase + 0.17f * static_cast<float>(i)) +
                    0.35f * std::cos(phase * 0.5f + 0.11f * static_cast<float>(i));
    }
    return values;
}

float rmse(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    double sum = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const double diff = static_cast<double>(lhs[i]) - rhs[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / static_cast<double>(lhs.size()));
}

std::vector<float> naive_matvec(
    const std::vector<float> & weights,
    uint32_t n_rows,
    uint32_t n_cols,
    const std::vector<float> & activation) {
    std::vector<float> output(n_rows, 0.0f);
    for (uint32_t row = 0; row < n_rows; ++row) {
        double sum = 0.0;
        for (uint32_t col = 0; col < n_cols; ++col) {
            sum += static_cast<double>(weights[static_cast<size_t>(row) * n_cols + col]) * activation[col];
        }
        output[row] = static_cast<float>(sum);
    }
    return output;
}

} // namespace

int main() {
    testing t;

    t.test("llama_turboquant_runtime_allows_key_only_block_so8_triality_vector", [](testing & t) {
        llama_turboquant_runtime_config cfg;
        cfg.enabled = true;
        cfg.mode = "key_only_block_so8_triality_vector";
        t.assert_true("k-side production mode is accepted", llama_turboquant_runtime_allows_k(cfg));
        t.assert_true("v-side remains disabled for key-only mode", !llama_turboquant_runtime_allows_v(cfg));
    });

    t.test("llama_turboquant_tq4_1s_reference_roundtrip_is_stable", [](testing & t) {
        const std::vector<float> source = make_wave_values(64, 0.25f);
        std::string error;
        const std::vector<uint8_t> packed = llama_turboquant_quantize_tq4_1s_reference(source, &error);
        t.assert_true("quantize succeeds", !packed.empty() && error.empty());
        t.assert_equal("packed size", size_t(40), packed.size());

        const std::vector<float> restored = llama_turboquant_dequantize_tq4_1s_reference(packed, &error);
        t.assert_true("dequantize succeeds", restored.size() == source.size() && error.empty());
        t.assert_true("roundtrip rmse bounded", rmse(source, restored) < 0.16f);
    });

    t.test("llama_turboquant_tq4_1s_reference_matches_golden_bytes", [](testing & t) {
        const std::vector<float> source = make_wave_values(64, 0.25f);
        const std::vector<uint8_t> expected = {
            27, 57, 47, 60, 153, 105, 211, 227, 122, 122,
            153, 56, 152, 152, 196, 4, 121, 185, 168, 169,
            26, 53, 164, 54, 85, 85, 192, 0, 115, 147,
            42, 42, 27, 171, 165, 55, 25, 72, 13, 254,
        };

        std::string error;
        const std::vector<uint8_t> packed = llama_turboquant_quantize_tq4_1s_reference(source, &error);
        t.assert_true("quantize succeeds", !packed.empty() && error.empty());
        t.assert_true("packed bytes stay byte exact", packed == expected);
    });

    t.test("ggml_tq4_1s_type_traits_and_quantization_match_reference_bytes", [](testing & t) {
        const std::vector<float> source = make_wave_values(64, 0.25f);
        const std::vector<uint8_t> expected = {
            27, 57, 47, 60, 153, 105, 211, 227, 122, 122,
            153, 56, 152, 152, 196, 4, 121, 185, 168, 169,
            26, 53, 164, 54, 85, 85, 192, 0, 115, 147,
            42, 42, 27, 171, 165, 55, 25, 72, 13, 254,
        };

        t.assert_equal("type name", std::string("tq4_1s"), std::string(ggml_type_name(GGML_TYPE_TQ4_1S)));
        t.assert_equal("block size", int64_t(32), ggml_blck_size(GGML_TYPE_TQ4_1S));
        t.assert_equal("row size", size_t(40), ggml_row_size(GGML_TYPE_TQ4_1S, 64));

        std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_TQ4_1S, 64), 0);
        quantize_row_tq4_1s_ref(source.data(), reinterpret_cast<block_tq4_1s *>(packed.data()), 64);
        t.assert_true("ggml ref bytes stay byte exact", packed == expected);
        t.assert_true("ggml row validation accepts tq4_1s payload", ggml_validate_row_data(GGML_TYPE_TQ4_1S, packed.data(), packed.size()));

        std::vector<float> restored(source.size(), 0.0f);
        dequantize_row_tq4_1s(reinterpret_cast<const block_tq4_1s *>(packed.data()), restored.data(), 64);
        t.assert_true("ggml dequant rmse bounded", rmse(source, restored) < 0.16f);
    });

    t.test("llama_turboquant_tq4_1s_reference_matvec_matches_dequantized_weights", [](testing & t) {
        constexpr uint32_t n_rows = 3;
        constexpr uint32_t n_cols = 64;
        const std::vector<float> weights = make_wave_values(n_rows * n_cols, 0.4f);
        const std::vector<float> activation = make_wave_values(n_cols, 1.2f);

        std::string error;
        const std::vector<uint8_t> packed = llama_turboquant_quantize_tq4_1s_reference(weights, &error);
        t.assert_true("quantize succeeds", !packed.empty() && error.empty());

        const std::vector<float> runtime = llama_turboquant_mul_mat_tq4_1s_reference(
            packed, n_rows, n_cols, activation, &error);
        t.assert_true("reference matvec succeeds", runtime.size() == n_rows && error.empty());

        const std::vector<float> dequantized = llama_turboquant_dequantize_tq4_1s_reference(packed, &error);
        t.assert_true("dequantize succeeds", dequantized.size() == weights.size() && error.empty());

        const std::vector<float> expected = naive_matvec(dequantized, n_rows, n_cols, activation);
        t.assert_true("runtime matches dequantized matvec", rmse(runtime, expected) < 1e-5f);
    });

    t.test("ggml_tq4_1s_q8_0_vec_dot_matches_dequantized_operands", [](testing & t) {
        const std::vector<float> weights = make_wave_values(64, 0.4f);
        const std::vector<float> activation = make_wave_values(64, 1.2f);

        std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_TQ4_1S, 64), 0);
        quantize_row_tq4_1s_ref(weights.data(), reinterpret_cast<block_tq4_1s *>(packed.data()), 64);

        std::vector<block_q8_0> act_blocks(64 / QK8_0);
        quantize_row_q8_0_ref(activation.data(), act_blocks.data(), 64);

        float dot = 0.0f;
        ggml_vec_dot_tq4_1s_q8_0(64, &dot, 0, packed.data(), 0, act_blocks.data(), 0, 1);

        std::vector<float> deq_weights(64, 0.0f);
        std::vector<float> deq_activation(64, 0.0f);
        dequantize_row_tq4_1s(reinterpret_cast<const block_tq4_1s *>(packed.data()), deq_weights.data(), 64);
        dequantize_row_q8_0(act_blocks.data(), deq_activation.data(), 64);

        float expected = 0.0f;
        for (size_t i = 0; i < deq_weights.size(); ++i) {
            expected += deq_weights[i] * deq_activation[i];
        }

        t.assert_true("cpu vec_dot matches dequantized operands", std::fabs(dot - expected) < 1e-3f);
    });

    t.test("llama_turboquant_validate_so8_rotation_accepts_identity_and_rejects_bad_rows", [](testing & t) {
        std::vector<float> identity(64, 0.0f);
        for (int i = 0; i < 8; ++i) {
            identity[static_cast<size_t>(i) * 8 + i] = 1.0f;
        }

        std::string error;
        t.assert_true("identity passes", llama_turboquant_validate_so8_rotation(identity, 1e-4f, &error));

        identity[1] = 0.5f;
        error.clear();
        t.assert_true("non-orthogonal fails", !llama_turboquant_validate_so8_rotation(identity, 1e-4f, &error));
        t.assert_true("error populated", !error.empty());
    });

    t.test("llama_turboquant_train_triality_codebook_learns_cluster_centers", [](testing & t) {
        constexpr uint32_t head_dim = 8;
        std::vector<float> values;
        values.reserve(9 * head_dim);

        const std::vector<float> cluster_a(head_dim, -1.0f);
        const std::vector<float> cluster_b(head_dim, 0.0f);
        const std::vector<float> cluster_c(head_dim, 1.0f);
        for (int i = 0; i < 3; ++i) {
            values.insert(values.end(), cluster_a.begin(), cluster_a.end());
            values.insert(values.end(), cluster_b.begin(), cluster_b.end());
            values.insert(values.end(), cluster_c.begin(), cluster_c.end());
        }

        const std::vector<float> codebook = llama_turboquant_train_triality_codebook(values, 9, head_dim);
        t.assert_equal("codebook size", size_t(3 * head_dim), codebook.size());

        const auto metrics = llama_turboquant_evaluate_triality(values, 9, head_dim, codebook);
        t.assert_true("triality mse is tiny on three clusters", metrics.triality_mse < 1e-6f);
        t.assert_true("relative reduction is strong", metrics.relative_mse_reduction > 0.99f);
    });

    return t.summary();
}
