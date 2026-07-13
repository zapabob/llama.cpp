#include "../src/llama-turboquant-consensus.h"
#include "turboquant-consensus-testing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

uint8_t unpack_interleaved(const uint8_t * row, size_t channel) {
    const size_t bit_offset = channel * LLAMA_TQ_RESIDUAL_PARITY_PHYSICAL_BITS_PER_CHANNEL;
    const size_t byte_offset = bit_offset / 8;
    const size_t shift = bit_offset % 8;
    uint16_t value = row[byte_offset];
    if (shift > 4) {
        value |= static_cast<uint16_t>(row[byte_offset + 1]) << 8;
    }
    return static_cast<uint8_t>((value >> shift) & 0x0fu);
}

std::vector<float> apply_inverse(
        const std::vector<float> & matrix,
        const std::vector<float> & values,
        size_t width) {
    std::vector<float> result(width, 0.0f);
    for (size_t row = 0; row < width; ++row) {
        for (size_t col = 0; col < width; ++col) {
            result[row] += matrix[row * width + col] * values[col];
        }
    }
    return result;
}

double dot(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    double value = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        value += static_cast<double>(lhs[i]) * rhs[i];
    }
    return value;
}

}  // namespace

int main() {
    tq_testing t;

    t.test("residual_parity_3_1_1_logical_sectors_use_four_bit_physical_parity", [](tq_testing & t) {
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
        t.assert_equal("derived minus has no duplicate physical payload", size_t(0), storage.sectors[2].payload.size());
        t.assert_equal("packed physical payload is four bits per channel",
            uint64_t(values.size() * 4 / 8), budget.payload_bytes);
        t.assert_equal("measured controller bytes", uint64_t(51), budget.controller_bytes);
        t.assert_true("controller-inclusive total is exact",
            std::fabs(budget.actual_bits_per_channel - 4.3984375) < 1e-12);
        t.assert_true("controller-inclusive five-bit target is met", budget.five_bit_target_met);

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
        storage.sectors[2]   = { 1, std::numeric_limits<float>::max(), {} };

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

    t.test("production_fixed_codec_uses_one_parity_coupled_four_bit_row", [](tq_testing & t) {
        constexpr size_t n_vectors = 3;
        constexpr size_t width = 9;
        std::vector<float> values(n_vectors * width);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.65f * std::sin(0.31f * static_cast<float>(i)) -
                        0.15f * std::cos(0.17f * static_cast<float>(i));
        }
        llama_tq_residual_parity_fixed_config config;
        config.sector_scales = { 0.25f, 0.125f, 0.0625f };
        config.beta = { 0.5f, 0.25f };
        llama_tq_residual_parity_fixed_storage storage;
        std::string error;
        t.assert_true("fixed encode succeeds",
            llama_tq_residual_parity_encode_fixed_f32(
                values, n_vectors, width, identity_rotations(width), config, storage, &error));
        t.assert_equal("controller ABI is fixed", size_t(51), sizeof(storage.controller));
        t.assert_equal("one row is ceil channels times four over eight", size_t(5), storage.row_bytes);
        t.assert_equal("payload is rows times packed row bytes", size_t(15), storage.payload.size());
        t.assert_true("controller magic is canonical",
            storage.controller[0] == 'T' && storage.controller[1] == 'Q' &&
            storage.controller[2] == 'R' && storage.controller[3] == 'P');
        t.assert_true("controller reserved tail is zero",
            std::all_of(storage.controller.begin() + 44, storage.controller.end(),
                [](uint8_t value) { return value == 0; }));
        t.assert_equal("controller declares parity-coupled physical layout", uint8_t(1), storage.controller[19]);
        bool padding_is_zero = true;
        for (size_t row = 0; row < n_vectors; ++row) {
            padding_is_zero = padding_is_zero && (storage.payload[(row + 1) * storage.row_bytes - 1] & 0xf0u) == 0;
        }
        t.assert_true("every row has deterministic zero padding", padding_is_zero);
        std::vector<float> decoded;
        t.assert_true("fixed decode succeeds",
            llama_tq_residual_parity_decode_fixed_f32(storage, identity_rotations(width), decoded, &error));
        t.assert_equal("decoded fixed shape", values.size(), decoded.size());
        t.assert_true("fixed reconstruction is bounded", rmse(values, decoded) < 0.2);
    });

    t.test("production_fixed_validation_rejects_reserved_padding_and_controller_corruption", [](tq_testing & t) {
        constexpr size_t width = 9;
        const std::vector<float> values(width, 0.0f);
        llama_tq_residual_parity_fixed_config config;
        config.sector_scales = { 0.25f, 0.125f, 0.0625f };
        llama_tq_residual_parity_fixed_storage storage;
        std::string error;
        t.assert_true("fixture encode succeeds",
            llama_tq_residual_parity_encode_fixed_f32(
                values, 1, width, identity_rotations(width), config, storage, &error));

        auto reserved = storage;
        reserved.payload[0] = static_cast<uint8_t>((reserved.payload[0] & 0xf8u) | 0x04u);
        t.assert_true("main code four fails closed",
            !llama_tq_residual_parity_validate_fixed_storage(reserved, &error));
        t.assert_true("reserved code is diagnosed", error.find("code 4") != std::string::npos);

        auto valid_negative = storage;
        valid_negative.payload[0] = static_cast<uint8_t>((valid_negative.payload[0] & 0xf8u) | 0x05u);
        t.assert_true("main codes five through seven remain valid",
            llama_tq_residual_parity_validate_fixed_storage(valid_negative, &error));

        auto padded = storage;
        padded.payload.back() |= 0x80u;
        t.assert_true("non-zero row padding fails closed",
            !llama_tq_residual_parity_validate_fixed_storage(padded, &error));
        t.assert_true("padding corruption is diagnosed", error.find("padding") != std::string::npos);

        auto controller = storage;
        controller.controller[44] = 1u;
        t.assert_true("controller reserved tail fails closed",
            !llama_tq_residual_parity_validate_fixed_storage(controller, &error));
        t.assert_true("controller corruption is diagnosed", error.find("controller") != std::string::npos);
    });

    t.test("production_multistream_decode_honors_explicit_stream_stride", [](tq_testing & t) {
        constexpr size_t n_stream = 2;
        constexpr size_t n_kv = 2;
        constexpr size_t width = 8;
        std::vector<float> values(n_stream * n_kv * width);
        for (size_t row = 0; row < n_stream * n_kv; ++row) {
            for (size_t channel = 0; channel < width; ++channel) {
                values[row * width + channel] =
                    static_cast<float>(row + 1) * 0.3f + static_cast<float>(channel) * 0.04f;
            }
        }
        llama_tq_residual_parity_fixed_config config;
        config.sector_scales = { 0.25f, 0.125f, 0.0625f };
        llama_tq_residual_parity_fixed_storage storage;
        std::string error;
        t.assert_true("multistream fixture encode succeeds",
            llama_tq_residual_parity_encode_fixed_f32(
                values, n_stream * n_kv, width, identity_rotations(width), config, storage, &error));
        std::vector<float> contiguous;
        t.assert_true("contiguous reference decode succeeds",
            llama_tq_residual_parity_decode_fixed_f32(storage, identity_rotations(width), contiguous, &error));

        const size_t stream_payload = n_kv * storage.row_bytes;
        const size_t stream_stride = stream_payload + 7;
        std::vector<uint8_t> strided(n_stream * stream_stride, 0xa5u);
        for (size_t stream = 0; stream < n_stream; ++stream) {
            std::memcpy(strided.data() + stream * stream_stride,
                        storage.payload.data() + stream * stream_payload,
                        stream_payload);
        }
        std::vector<float> decoded;
        t.assert_true("strided decode succeeds",
            llama_tq_residual_parity_decode_fixed_strided_f32(
                strided, n_stream, n_kv, width, stream_stride,
                identity_rotations(width), config, decoded, &error));
        t.assert_equal("strided output shape", contiguous.size(), decoded.size());
        t.assert_true("explicit stream stride preserves all rows", rmse(contiguous, decoded) < 1e-7);
        t.assert_true("undersized stream stride fails closed",
            !llama_tq_residual_parity_decode_fixed_strided_f32(
                strided, n_stream, n_kv, width, stream_payload - 1,
                identity_rotations(width), config, decoded, &error));
    });

    t.test("decoded_standard_kq_matches_sector_sum_formula", [](tq_testing & t) {
        constexpr size_t width = 8;
        const std::vector<float> values{ 0.85f, -0.42f, 0.37f, -0.66f, 0.12f, 0.54f, -0.29f, 0.73f };
        const std::vector<float> query{ 0.2f, -0.4f, 0.7f, 0.1f, -0.3f, 0.6f, 0.5f, -0.2f };
        llama_tq_residual_parity_fixed_config config;
        config.sector_scales = { 0.25f, 0.125f, 0.0625f };
        config.beta = { 0.5f, 0.25f };
        const llama_tq_rotation_bundle rotations = identity_rotations(width);
        llama_tq_residual_parity_fixed_storage storage;
        std::string error;
        t.assert_true("formula fixture encode succeeds",
            llama_tq_residual_parity_encode_fixed_f32(
                values, 1, width, rotations, config, storage, &error));
        std::vector<float> decoded;
        t.assert_true("formula fixture decode succeeds",
            llama_tq_residual_parity_decode_fixed_f32(storage, rotations, decoded, &error));

        std::vector<float> main(width);
        std::vector<float> plus(width);
        std::vector<float> minus(width);
        for (size_t channel = 0; channel < width; ++channel) {
            const uint8_t code = unpack_interleaved(storage.payload.data(), channel);
            const uint8_t main_code = code & 0x07u;
            const int quantized = main_code >= 4 ? static_cast<int>(main_code) - 8 : main_code;
            const uint8_t plus_bit = (code >> 3) & 1u;
            const uint8_t minus_bit = plus_bit ^ (main_code & 1u);
            main[channel] = quantized * config.sector_scales[0];
            plus[channel] = plus_bit ? config.sector_scales[1] : -config.sector_scales[1];
            minus[channel] = minus_bit ? config.sector_scales[2] : -config.sector_scales[2];
        }
        const double sector_sum = dot(query, apply_inverse(rotations.inverse[0], main, width)) +
            config.beta[0] * dot(query, apply_inverse(rotations.inverse[1], plus, width)) +
            config.beta[1] * dot(query, apply_inverse(rotations.inverse[2], minus, width));
        t.assert_true("decoded KQ is formula-equivalent", std::fabs(dot(query, decoded) - sector_sum) < 1e-6);
    });

    t.test("production_budget_includes_controller_and_fails_closed_below_amortization_floor", [](tq_testing & t) {
        struct passing_shape {
            size_t layers;
            size_t rows_per_layer;
            size_t channels;
        };
        const std::array<passing_shape, 3> passing {{
            { 2, 32, 128 },
            { 3, 64, 64 },
            { 2, 96, 9 },
        }};
        for (const auto & shape : passing) {
            const auto budget = llama_tq_residual_parity_measure_production(
                shape.layers, shape.rows_per_layer, shape.channels);
            const uint64_t row_bytes =
                (shape.channels * LLAMA_TQ_RESIDUAL_PARITY_PHYSICAL_BITS_PER_CHANNEL + 7) / 8;
            t.assert_equal("physical payload uses one four-bit row per input",
                static_cast<uint64_t>(shape.layers * shape.rows_per_layer) * row_bytes,
                budget.physical_payload_bytes);
            t.assert_equal("controller remains 51 bytes per layer",
                static_cast<uint64_t>(shape.layers * LLAMA_TQ_RESIDUAL_PARITY_CONTROLLER_BYTES),
                budget.controller_bytes);
            t.assert_true("controller-inclusive actual budget is at most five BPC",
                budget.actual_bits_per_channel <= 5.0 && budget.five_bit_target_met);
        }

        const auto impossible = llama_tq_residual_parity_measure_production(2, 7, 9);
        t.assert_equal("small odd-width physical payload is exact", uint64_t(70), impossible.physical_payload_bytes);
        t.assert_equal("small-shape controller bytes are still charged", uint64_t(102), impossible.controller_bytes);
        t.assert_true("mathematically impossible small shape fails the five-bit gate",
            impossible.actual_bits_per_channel > 5.0 && !impossible.five_bit_target_met);
    });

    t.test("production_state_payload_copy_and_shift_roundtrip", [](tq_testing & t) {
        constexpr size_t n_vectors = 5;
        constexpr size_t width = 9;
        constexpr size_t shift = 2;
        std::vector<float> values(n_vectors * width);
        for (size_t row = 0; row < n_vectors; ++row) {
            for (size_t channel = 0; channel < width; ++channel) {
                values[row * width + channel] =
                    0.2f * static_cast<float>(row) - 0.05f * static_cast<float>(channel);
            }
        }
        llama_tq_residual_parity_fixed_config config;
        config.sector_scales = { 0.25f, 0.125f, 0.0625f };
        llama_tq_residual_parity_fixed_storage storage;
        std::string error;
        t.assert_true("state fixture encode succeeds",
            llama_tq_residual_parity_encode_fixed_f32(
                values, n_vectors, width, identity_rotations(width), config, storage, &error));
        std::vector<float> original;
        t.assert_true("state fixture decode succeeds",
            llama_tq_residual_parity_decode_fixed_f32(storage, identity_rotations(width), original, &error));

        llama_tq_residual_parity_fixed_storage restored;
        restored.n_vectors = storage.n_vectors;
        restored.width = storage.width;
        restored.row_bytes = storage.row_bytes;
        restored.config = storage.config;
        restored.controller = storage.controller;
        restored.payload = storage.payload;
        t.assert_true("restored state validates",
            llama_tq_residual_parity_validate_fixed_storage(restored, &error));
        std::vector<float> restored_values;
        t.assert_true("restored state decodes",
            llama_tq_residual_parity_decode_fixed_f32(
                restored, identity_rotations(width), restored_values, &error));
        t.assert_true("state payload roundtrip is exact", rmse(original, restored_values) < 1e-7);

        const size_t remaining = n_vectors - shift;
        std::memmove(restored.payload.data(),
                     restored.payload.data() + shift * restored.row_bytes,
                     remaining * restored.row_bytes);
        std::fill(restored.payload.begin() + remaining * restored.row_bytes, restored.payload.end(), 0);
        t.assert_true("shifted packed rows remain canonical",
            llama_tq_residual_parity_validate_fixed_storage(restored, &error));
        std::vector<float> shifted;
        t.assert_true("shifted packed rows decode",
            llama_tq_residual_parity_decode_fixed_f32(
                restored, identity_rotations(width), shifted, &error));
        bool prefix_matches = true;
        for (size_t i = 0; i < remaining * width; ++i) {
            prefix_matches = prefix_matches && std::fabs(shifted[i] - original[shift * width + i]) < 1e-7f;
        }
        t.assert_true("shift roundtrip preserves surviving rows", prefix_matches);
    });

    return t.summary();
}
