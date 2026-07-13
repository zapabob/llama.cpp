#include "../src/llama-turboquant-consensus.h"
#include "turboquant-consensus-testing.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

llama_tq_rotation_bundle identity_rotations(size_t width) {
    llama_tq_rotation_bundle rotations;
    for (size_t view = 0; view < 3; ++view) {
        rotations.forward[view].assign(width * width, 0.0f);
        rotations.inverse[view].assign(width * width, 0.0f);
        for (size_t i = 0; i < width; ++i) {
            rotations.forward[view][i * width + i] = 1.0f;
            rotations.inverse[view][i * width + i] = 1.0f;
        }
    }
    return rotations;
}

double rmse(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    double sum = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const double difference = static_cast<double>(lhs[i]) - rhs[i];
        sum += difference * difference;
    }
    return std::sqrt(sum / lhs.size());
}

}  // namespace

int main() {
    tq_testing t;

    t.test("residual_parity_3_1_1_roundtrips_with_exact_payload_budget", [](tq_testing & t) {
        constexpr size_t   n_vectors = 128;
        constexpr size_t   width     = 8;
        std::vector<float> values(n_vectors * width, 0.0f);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] =
                0.75f * std::sin(0.071f * static_cast<float>(i)) + 0.2f * std::cos(0.037f * static_cast<float>(i));
        }

        llama_tq_residual_parity_storage storage;
        std::string                      error;
        t.assert_true("3+1+1 encode succeeds",
                      llama_tq_residual_parity_encode_f32(values, n_vectors, width, identity_rotations(width),
                                                          llama_tq_residual_parity_profile{}, storage, &error));

        const llama_tq_residual_parity_budget budget = llama_tq_residual_parity_measure(storage);
        t.assert_equal("main payload bits", uint64_t(values.size() * 3), budget.sector_payload_bits[0]);
        t.assert_equal("plus payload bits", uint64_t(values.size()), budget.sector_payload_bits[1]);
        t.assert_equal("minus payload bits", uint64_t(values.size()), budget.sector_payload_bits[2]);
        t.assert_true("payload meets five-bit target", std::fabs(budget.payload_bits_per_channel - 5.0) < 1e-12);
        t.assert_equal("packed payload bytes are exact", uint64_t(values.size() * 5 / 8), budget.payload_bytes);
        t.assert_true("controller overhead is reported", budget.controller_bytes > 0);
        t.assert_true("actual budget includes controller overhead", budget.actual_bits_per_channel > 5.0);

        std::vector<float> decoded;
        t.assert_true("3+1+1 decode succeeds",
                      llama_tq_residual_parity_decode_f32(storage, identity_rotations(width), decoded, &error));
        t.assert_equal("decoded shape", values.size(), decoded.size());
        t.assert_true("reference reconstruction error is bounded", rmse(values, decoded) < 0.22);
    });

    t.test("residual_parity_rejects_corrupt_reserved_codes", [](tq_testing & t) {
        constexpr size_t                 n_vectors = 2;
        constexpr size_t                 width     = 8;
        const std::vector<float>         values(n_vectors * width, 0.25f);
        llama_tq_residual_parity_storage storage;
        std::string                      error;
        t.assert_true("encode succeeds",
                      llama_tq_residual_parity_encode_f32(values, n_vectors, width, identity_rotations(width),
                                                          llama_tq_residual_parity_profile{}, storage, &error));

        storage.sectors[0].payload[0] = static_cast<uint8_t>((storage.sectors[0].payload[0] & 0xf8u) | 0x04u);
        std::vector<float> decoded;
        t.assert_true("reserved signed code fails closed",
                      !llama_tq_residual_parity_decode_f32(storage, identity_rotations(width), decoded, &error));
        t.assert_true("corruption reports a reason", error.find("reserved") != std::string::npos);
    });

    t.test("residual_parity_rejects_corrupt_zero_scale_codes", [](tq_testing & t) {
        constexpr size_t                 width = 8;
        const std::vector<float>         values(width, 0.0f);
        llama_tq_residual_parity_storage storage;
        std::string                      error;
        t.assert_true("zero input encode succeeds",
                      llama_tq_residual_parity_encode_f32(values, 1, width, identity_rotations(width),
                                                          llama_tq_residual_parity_profile{}, storage, &error));

        storage.sectors[0].payload[0] = 0x04u;
        std::vector<float> decoded;
        t.assert_true("zero-scale reserved code fails closed",
                      !llama_tq_residual_parity_decode_f32(storage, identity_rotations(width), decoded, &error));
        t.assert_true("zero-scale corruption reports a reason", !error.empty());
    });

    t.test("residual_parity_rejects_non_inverse_rotations", [](tq_testing & t) {
        constexpr size_t         width     = 4;
        llama_tq_rotation_bundle rotations = identity_rotations(width);
        rotations.inverse[1][0]            = 2.0f;
        const std::vector<float>         values(width, 1.0f);
        llama_tq_residual_parity_storage storage;
        std::string                      error;
        t.assert_true("invalid inverse fails closed",
                      !llama_tq_residual_parity_encode_f32(values, 1, width, rotations,
                                                           llama_tq_residual_parity_profile{}, storage, &error));
        t.assert_true("inverse failure reports a reason", error.find("inverse") != std::string::npos);
    });

    t.test("residual_parity_rejects_finite_inputs_that_overflow_rotations", [](tq_testing & t) {
        llama_tq_rotation_bundle rotations = identity_rotations(1);
        rotations.forward[0][0]            = std::numeric_limits<float>::max();
        rotations.inverse[0][0]            = 1.0f / std::numeric_limits<float>::max();
        const std::vector<float>         values{ std::numeric_limits<float>::max() };
        llama_tq_residual_parity_storage storage;
        std::string                      error;
        t.assert_true("rotation overflow fails closed",
                      !llama_tq_residual_parity_encode_f32(values, 1, 1, rotations, llama_tq_residual_parity_profile{},
                                                           storage, &error));
        t.assert_true("rotation overflow reports a reason", error.find("overflowed") != std::string::npos);
    });

    t.test("residual_parity_rejects_final_reconstruction_overflow", [](tq_testing & t) {
        llama_tq_residual_parity_storage storage;
        storage.n_vectors    = 1;
        storage.width        = 1;
        storage.profile.beta = { 1.0f, 1.0f };
        storage.sectors[0]   = { 3, 0.0f, { 0x00u } };
        storage.sectors[1]   = { 1, std::numeric_limits<float>::max(), { 0x01u } };
        storage.sectors[2]   = { 1, std::numeric_limits<float>::max(), { 0x01u } };

        std::vector<float> decoded;
        std::string        error;
        t.assert_true("final reconstruction overflow fails closed",
                      !llama_tq_residual_parity_decode_f32(storage, identity_rotations(1), decoded, &error));
        t.assert_true("final overflow reports a reason", error.find("final reconstruction") != std::string::npos);
        t.assert_true("failed decode does not publish values", decoded.empty());
    });

    t.test("residual_parity_validation_failures_are_diagnostic", [](tq_testing & t) {
        llama_tq_residual_parity_storage storage;
        std::string                      error;
        const std::vector<float>         non_finite{ std::numeric_limits<float>::infinity() };
        t.assert_true("non-finite encode fails closed",
                      !llama_tq_residual_parity_encode_f32(non_finite, 1, 1, identity_rotations(1),
                                                           llama_tq_residual_parity_profile{}, storage, &error));
        t.assert_true("non-finite encode reports a reason", error.find("non-finite") != std::string::npos);

        storage.n_vectors      = 1;
        storage.width          = 1;
        storage.profile.bits   = { 2, 2, 1 };
        std::vector<float> out = { 17.0f };
        error.clear();
        t.assert_true("unsupported stored profile fails closed",
                      !llama_tq_residual_parity_decode_f32(storage, identity_rotations(1), out, &error));
        t.assert_true("unsupported profile reports a reason", error.find("profile") != std::string::npos);
        t.assert_equal("failed profile decode preserves output", 17.0f, out[0]);
    });

    return t.summary();
}
