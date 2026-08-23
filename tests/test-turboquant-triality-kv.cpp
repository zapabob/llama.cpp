#include "../src/llama-turboquant-consensus.h"
#include "turboquant-consensus-testing.h"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

int main() {
    tq_testing t;

    t.test("full_triple_k_storage_copies_shifts_and_roundtrips", [](tq_testing & t) {
        llama_tq_full_k_storage storage;
        std::string             error;
        t.assert_true("initialize full triple", storage.initialize(llama_tq_k_storage_mode::full_triple, 4, 3, &error));

        for (size_t token = 0; token < 4; ++token) {
            std::array<std::vector<float>, 3> views;
            for (size_t view = 0; view < 3; ++view) {
                views[view] = {
                    static_cast<float>(100 * view + 10 * token),
                    static_cast<float>(100 * view + 10 * token + 1),
                    static_cast<float>(100 * view + 10 * token + 2),
                };
            }
            t.assert_true("set all three token views", storage.set_token(token, views, &error));
        }

        t.assert_true("overlapping copy succeeds", storage.copy_tokens(0, 2, 2, &error));
        t.assert_true("shift succeeds", storage.shift_left(1, &error));
        t.assert_equal("plus view shifted", 110.0f, storage.view(1)[0]);
        t.assert_equal("minus copied token retained", 200.0f, storage.view(2)[3]);
        t.assert_equal("tail is cleared", 0.0f, storage.view(0)[11]);

        const llama_tq_k_memory_breakdown breakdown = storage.memory_breakdown();
        t.assert_equal("vector bytes", size_t(48), breakdown.view_bytes[0]);
        t.assert_equal("plus bytes", size_t(48), breakdown.view_bytes[1]);
        t.assert_equal("minus bytes", size_t(48), breakdown.view_bytes[2]);

        const std::vector<uint8_t> bytes = storage.serialize();
        t.assert_equal("serialized size is measured", breakdown.total_bytes(), bytes.size());

        llama_tq_full_k_storage restored;
        t.assert_true("deserialize succeeds", llama_tq_full_k_storage::deserialize(bytes, restored, &error));
        t.assert_true("vector view roundtrips", restored.view(0) == storage.view(0));
        t.assert_true("plus view roundtrips", restored.view(1) == storage.view(1));
        t.assert_true("minus view roundtrips", restored.view(2) == storage.view(2));

        std::vector<uint8_t> malformed = bytes;
        malformed.push_back(0);
        t.assert_true("trailing bytes fail closed", !llama_tq_full_k_storage::deserialize(malformed, restored, &error));
        t.assert_true("failure reports a reason", !error.empty());

        malformed                = bytes;
        const auto overwrite_u64 = [](std::vector<uint8_t> & target, size_t offset, uint64_t value) {
            for (uint32_t shift = 0; shift < 64; shift += 8) {
                target[offset++] = static_cast<uint8_t>((value >> shift) & 0xffu);
            }
        };
        overwrite_u64(malformed, 16, std::numeric_limits<uint64_t>::max());
        t.assert_true("forged dimensions fail before allocation",
                      !llama_tq_full_k_storage::deserialize(malformed, restored, &error));
        t.assert_true("forged dimensions report a reason", error.find("payload length") != std::string::npos);
    });

    t.test("runtime_capability_gate_is_fail_closed", [](tq_testing & t) {
        llama_tq_runtime_capabilities capabilities;
        std::string                   error;
        t.assert_true(
            "single view remains available",
            llama_tq_require_runtime_capabilities(llama_tq_k_storage_mode::single_view, false, capabilities, &error));
        t.assert_true(
            "full triple is rejected before graph integration",
            !llama_tq_require_runtime_capabilities(llama_tq_k_storage_mode::full_triple, false, capabilities, &error));
        t.assert_true("full triple rejection is explicit", error.find("production graph") != std::string::npos);

        capabilities.full_triple_kv_graph = true;
        t.assert_true(
            "attention is independently rejected",
            !llama_tq_require_runtime_capabilities(llama_tq_k_storage_mode::full_triple, true, capabilities, &error));
        capabilities.cpu_attention_consensus_graph = true;
        t.assert_true(
            "single-view consensus is rejected even with a graph capability",
            !llama_tq_require_runtime_capabilities(llama_tq_k_storage_mode::single_view, true, capabilities, &error));
        t.assert_true("single-view consensus rejection is explicit", error.find("multi-view") != std::string::npos);
        t.assert_true(
            "complete capability set is accepted",
            llama_tq_require_runtime_capabilities(llama_tq_k_storage_mode::full_triple, true, capabilities, &error));
        t.assert_true("unknown storage mode fails closed",
                      !llama_tq_require_runtime_capabilities(static_cast<llama_tq_k_storage_mode>(99), false,
                                                             capabilities, &error));
    });

    t.test("view_shift_inverse_rotates_before_rope_and_rotates_forward", [](tq_testing & t) {
        using vector8 = std::array<float, 8>;
        using matrix8 = std::array<float, 64>;

        matrix8 rotation {};
        for (size_t i = 0; i < 8; ++i) {
            rotation[i * 8 + i] = 1.0f;
        }
        const float c = std::cos(0.37f);
        const float s = std::sin(0.37f);
        rotation[0] = c;
        rotation[1] = -s;
        rotation[8] = s;
        rotation[9] = c;
        rotation[36] = c;
        rotation[37] = -s;
        rotation[44] = s;
        rotation[45] = c;

        const auto multiply = [](const matrix8 & matrix, const vector8 & value, bool transpose) {
            vector8 result {};
            for (size_t row = 0; row < 8; ++row) {
                for (size_t col = 0; col < 8; ++col) {
                    result[row] += matrix[(transpose ? col * 8 + row : row * 8 + col)] * value[col];
                }
            }
            return result;
        };
        const auto rope_shift = [](vector8 value) {
            for (size_t pair = 0; pair < 4; ++pair) {
                const float angle = 0.19f * static_cast<float>(pair + 1);
                const float rc = std::cos(angle);
                const float rs = std::sin(angle);
                const float x = value[2 * pair];
                const float y = value[2 * pair + 1];
                value[2 * pair] = rc * x - rs * y;
                value[2 * pair + 1] = rs * x + rc * y;
            }
            return value;
        };

        const vector8 base = { 0.2f, -0.7f, 1.1f, 0.4f, -0.3f, 0.8f, 0.6f, -1.2f };
        const vector8 stored = multiply(rotation, base, false);
        const vector8 recovered = multiply(rotation, stored, true);
        const vector8 shifted_view = multiply(rotation, rope_shift(recovered), false);
        const vector8 expected = multiply(rotation, rope_shift(base), false);
        for (size_t i = 0; i < 8; ++i) {
            t.assert_true("inverse-R shift matches direct re-encoding", std::fabs(shifted_view[i] - expected[i]) < 1e-6f);
        }
    });

    return t.summary();
}
