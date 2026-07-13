#include "../src/llama-turboquant-consensus.h"
#include "turboquant-consensus-testing.h"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

int main() {
    tq_testing t;

    t.test("cpu_consensus_calibrates_before_one_softmax_and_value_matmul", [](tq_testing & t) {
        const std::vector<float>                query{ 1.0f, 2.0f };
        const std::array<std::vector<float>, 3> keys{
            {
             { 1.0f, 0.0f, 0.0f, 1.0f },
             { 2.0f, 0.0f, 0.0f, 2.0f },
             { -1.0f, 1.0f, 1.0f, -1.0f },
             }
        };
        const std::vector<float> values{ 10.0f, 0.0f, 0.0f, 20.0f };

        llama_tq_consensus_config config;
        config.weights = { 0.5f, 0.3f, 0.2f };
        config.bias    = { 0.0f, 0.1f, -0.2f };
        config.scale   = { 1.0f, 1.2f, 0.8f };

        llama_tq_consensus_result result;
        std::string               error;
        t.assert_true("reference attention succeeds",
                      llama_tq_consensus_attention_f32(query, keys, values, {}, 1, 2, 2, 2, config, result, &error));

        const float                inv_sqrt_two = 1.0f / std::sqrt(2.0f);
        const std::array<float, 3> expected_first{
            {
             inv_sqrt_two, 1.2f * 2.0f * inv_sqrt_two + 0.1f,
             0.8f * inv_sqrt_two - 0.2f,
             }
        };
        const std::array<float, 3> expected_second{
            {
             2.0f * inv_sqrt_two,
             1.2f * 4.0f * inv_sqrt_two + 0.1f,
             -0.8f * inv_sqrt_two - 0.2f,
             }
        };
        float consensus_first  = 0.0f;
        float consensus_second = 0.0f;
        for (size_t view = 0; view < 3; ++view) {
            t.assert_true("first calibrated branch matches",
                          std::fabs(result.branch_logits[view][0] - expected_first[view]) < 1e-6f);
            t.assert_true("second calibrated branch matches",
                          std::fabs(result.branch_logits[view][1] - expected_second[view]) < 1e-6f);
            consensus_first += config.weights[view] * expected_first[view];
            consensus_second += config.weights[view] * expected_second[view];
        }
        t.assert_true("first consensus logit matches", std::fabs(result.consensus_logits[0] - consensus_first) < 1e-6f);
        t.assert_true("second consensus logit matches",
                      std::fabs(result.consensus_logits[1] - consensus_second) < 1e-6f);
        t.assert_equal("one final softmax pass", uint32_t(1), result.softmax_passes);
        t.assert_equal("one final value matmul pass", uint32_t(1), result.value_matmul_passes);
        t.assert_true("probabilities normalize",
                      std::fabs(result.probabilities[0] + result.probabilities[1] - 1.0f) < 1e-6f);
        t.assert_true("value output uses final probabilities",
                      std::fabs(result.output[0] - 10.0f * result.probabilities[0]) < 1e-6f &&
                          std::fabs(result.output[1] - 20.0f * result.probabilities[1]) < 1e-6f);
    });

    t.test("cpu_consensus_applies_mask_before_its_only_softmax", [](tq_testing & t) {
        const std::vector<float>                query{ 1.0f, 0.0f };
        const std::array<std::vector<float>, 3> keys{
            {
             { 1.0f, 0.0f, 0.0f, 1.0f },
             { 1.0f, 0.0f, 0.0f, 1.0f },
             { 1.0f, 0.0f, 0.0f, 1.0f },
             }
        };
        const std::vector<float>  values{ 3.0f, 4.0f, 30.0f, 40.0f };
        const std::vector<float>  mask{ 0.0f, -std::numeric_limits<float>::infinity() };
        llama_tq_consensus_config config;
        config.weights = { 0.5f, 0.3f, 0.2f };

        llama_tq_consensus_result result;
        std::string               error;
        t.assert_true("masked attention succeeds",
                      llama_tq_consensus_attention_f32(query, keys, values, mask, 1, 2, 2, 2, config, result, &error));
        t.assert_equal("masked probability is zero", 0.0f, result.probabilities[1]);
        t.assert_equal("first value channel selected", 3.0f, result.output[0]);
        t.assert_equal("second value channel selected", 4.0f, result.output[1]);
        t.assert_equal("one masked softmax pass", uint32_t(1), result.softmax_passes);
    });

    t.test("cpu_consensus_rejects_invalid_controller_weights", [](tq_testing & t) {
        const std::vector<float>                query{ 1.0f };
        const std::array<std::vector<float>, 3> keys{
            { { 1.0f }, { 1.0f }, { 1.0f } }
        };
        const std::vector<float>  values{ 1.0f };
        llama_tq_consensus_config config;
        config.weights = { 0.5f, 0.6f, -0.1f };

        llama_tq_consensus_result result;
        std::string               error;
        t.assert_true("negative weight fails closed",
                      !llama_tq_consensus_attention_f32(query, keys, values, {}, 1, 1, 1, 1, config, result, &error));
        t.assert_true("invalid controller reports a reason", !error.empty());
    });

    t.test("cpu_consensus_rejects_finite_inputs_that_overflow_f32_logits", [](tq_testing & t) {
        const float                             maximum = std::numeric_limits<float>::max();
        const std::vector<float>                query{ maximum };
        const std::array<std::vector<float>, 3> keys{
            { { maximum }, { maximum }, { maximum } }
        };
        const std::vector<float>  values{ 1.0f };
        llama_tq_consensus_config config;
        config.weights = { 0.5f, 0.3f, 0.2f };

        llama_tq_consensus_result result;
        std::string               error;
        t.assert_true("overflow fails closed",
                      !llama_tq_consensus_attention_f32(query, keys, values, {}, 1, 1, 1, 1, config, result, &error));
        t.assert_true("overflow reports a reason", error.find("overflowed") != std::string::npos);
    });

    return t.summary();
}
