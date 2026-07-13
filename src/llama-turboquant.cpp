#include "llama-turboquant.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

namespace {
constexpr float TURBOQUANT_FLOAT_EPS = 1e-6f;

bool env_flag(const char * name, bool fallback) {
    const char * v = std::getenv(name);
    if (!v) {
        return fallback;
    }
    return std::atoi(v) != 0;
}

float env_float(const char * name, float fallback) {
    const char * v = std::getenv(name);
    if (!v) {
        return fallback;
    }
    return std::strtof(v, nullptr);
}

uint32_t env_u32(const char * name, uint32_t fallback) {
    const char * v = std::getenv(name);
    if (!v) {
        return fallback;
    }
    const int val = std::atoi(v);
    return val < 0 ? fallback : static_cast<uint32_t>(val);
}

std::string env_string(const char * name, const char * fallback) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return std::string(fallback);
    }
    return std::string(v);
}

float sqr(float x) {
    return x * x;
}

constexpr uint32_t TQ4_1S_BLOCK_SIZE = 32;
constexpr uint32_t TQ4_1S_BLOCK_BYTES = 20;
constexpr uint32_t TQ4_1S_HALF_BLOCK = 16;
constexpr float TQ4_1S_INV_SQRT32 = 0.17677669529663688f;
constexpr uint32_t TRIALITY_CODEBOOK_SIZE = 3;
constexpr uint32_t TRIALITY_ROTATION_BLOCK_SIZE = 8;
constexpr const char * TRIALITY_RUNTIME_MODE = "key_only_block_so8_triality_vector";
constexpr const char * TRIALITY_RUNTIME_MODE_PLUS = "key_only_block_so8_triality_plus";
constexpr const char * TRIALITY_RUNTIME_MODE_MINUS = "key_only_block_so8_triality_minus";
constexpr const char * TRIALITY_RUNTIME_MODE_BEST_PER_LAYER = "key_only_block_so8_triality_best_per_layer";
constexpr const char * TRIALITY_PUBLIC_CACHE_TYPE_K_VECTOR = "triality-vector";
constexpr const char * TRIALITY_PUBLIC_CACHE_TYPE_K_PLUS = "triality-plus";
constexpr const char * TRIALITY_PUBLIC_CACHE_TYPE_K_MINUS = "triality-minus";
constexpr const char * TRIALITY_PUBLIC_CACHE_TYPE_K_BEST_PER_LAYER = "best_per_layer";
constexpr const char * TURBOQUANT_PUBLIC_CACHE_TYPE_K_Q8_0 = "q8_0";
constexpr const char * TURBOQUANT_PUBLIC_CACHE_TYPE_V_Q8_0 = "q8_0";
constexpr const char * TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO4 = "turbo4";
constexpr const char * TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO3 = "turbo3";
constexpr const char * TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO2 = "turbo2";

constexpr float kTq4CentroidsWeight[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

constexpr float kTq4MidpointsWeight[15] = {
    -2.400804f, -1.843532f, -1.437139f, -1.099286f, -0.799550f,
    -0.522404f, -0.258222f,  0.000000f,  0.258222f,  0.522404f,
     0.799550f,  1.099286f,  1.437139f,  1.843532f,  2.400804f,
};

constexpr float kTqWeightSigns[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

bool set_error(std::string * error, const std::string & message);
std::string canonical_triality_view(std::string view);
std::string canonical_runtime_mode(std::string mode);
std::string canonical_public_cache_type_k(std::string cache_type_k);
std::string canonical_public_cache_type_v(std::string cache_type_v);
std::string public_cache_type_k_from_runtime_mode(const std::string & mode);
std::string public_cache_type_v_from_mode(const std::string & mode);
bool is_supported_public_cache_type_k(const std::string & cache_type_k);
bool is_supported_public_cache_type_v(const std::string & cache_type_v);
bool public_cache_types_match_runtime_mode(const std::string & mode, const std::string & cache_type_k);
bool is_supported_triality_view(const std::string & view);
bool is_supported_runtime_mode(const std::string & mode);
bool runtime_mode_matches_view(const std::string & mode, const std::string & view);

bool validate_tq4_reference_shape(
    const size_t value_count,
    const std::string & label,
    std::string * error) {
    if (value_count == 0 || value_count % TQ4_1S_BLOCK_SIZE != 0) {
        return set_error(
            error,
            label + " size must be a non-zero multiple of 32 values");
    }
    return true;
}

void fwht32(float * values) {
    for (uint32_t step = 1; step < TQ4_1S_BLOCK_SIZE; step <<= 1) {
        for (uint32_t base = 0; base < TQ4_1S_BLOCK_SIZE; base += step << 1) {
            for (uint32_t j = base; j < base + step; ++j) {
                const float a = values[j];
                const float b = values[j + step];
                values[j] = a + b;
                values[j + step] = a - b;
            }
        }
    }
}

void tq4_rht_forward(float * values) {
    for (uint32_t i = 0; i < TQ4_1S_BLOCK_SIZE; ++i) {
        values[i] *= kTqWeightSigns[i];
    }
    fwht32(values);
    for (uint32_t i = 0; i < TQ4_1S_BLOCK_SIZE; ++i) {
        values[i] *= TQ4_1S_INV_SQRT32;
    }
}

void tq4_rht_inverse(float * values) {
    fwht32(values);
    for (uint32_t i = 0; i < TQ4_1S_BLOCK_SIZE; ++i) {
        values[i] *= TQ4_1S_INV_SQRT32 * kTqWeightSigns[i];
    }
}

uint8_t tq4_choose_index(const float value) {
    for (uint8_t i = 0; i < 15; ++i) {
        if (value < kTq4MidpointsWeight[i]) {
            return i;
        }
    }
    return 15;
}

float squared_l2_distance(
    const std::vector<float> & lhs,
    size_t lhs_offset,
    const std::vector<float> & rhs,
    size_t rhs_offset,
    const uint32_t width) {
    double sum = 0.0;
    for (uint32_t i = 0; i < width; ++i) {
        const double diff = static_cast<double>(lhs[lhs_offset + i]) - rhs[rhs_offset + i];
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

bool set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

bool validate_runtime_bits(
    float stage1_effective_bits,
    uint32_t qjl_bits,
    float runtime_bits_per_channel,
    const std::string & label,
    std::string * error) {
    if (std::fabs((stage1_effective_bits + static_cast<float>(qjl_bits)) - runtime_bits_per_channel) > TURBOQUANT_FLOAT_EPS) {
        return set_error(
            error,
            label + " is inconsistent: tq_stage1_effective_bits + tq_qjl_bits != tq_runtime_bits_per_channel");
    }
    return true;
}

bool parse_u32(const std::map<std::string, std::string> & values, const char * key, uint32_t & out, std::string * error) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return set_error(error, std::string("missing artifact metadata key `") + key + "`");
    }
    try {
        out = static_cast<uint32_t>(std::stoul(it->second));
    } catch (const std::exception &) {
        return set_error(error, std::string("invalid uint32 artifact metadata for `") + key + "`");
    }
    return true;
}

bool parse_f32(const std::map<std::string, std::string> & values, const char * key, float & out, std::string * error) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return set_error(error, std::string("missing artifact metadata key `") + key + "`");
    }
    try {
        out = std::stof(it->second);
    } catch (const std::exception &) {
        return set_error(error, std::string("invalid float artifact metadata for `") + key + "`");
    }
    return true;
}

bool parse_string(const std::map<std::string, std::string> & values, const char * key, std::string & out, std::string * error) {
    const auto it = values.find(key);
    if (it == values.end()) {
        return set_error(error, std::string("missing artifact metadata key `") + key + "`");
    }
    out = it->second;
    return true;
}

std::vector<std::string> serialize_metadata_lines(const llama_turboquant_artifact_metadata & metadata) {
    std::vector<std::string> lines;
    lines.reserve(15);
    lines.push_back("tq_schema_version=" + std::to_string(metadata.schema_version));
    lines.push_back("tq_total_bits=" + std::to_string(metadata.total_bits));
    lines.push_back("tq_runtime_bits_per_channel=" + std::to_string(metadata.runtime_bits_per_channel));
    lines.push_back("tq_stage1_effective_bits=" + std::to_string(metadata.stage1_effective_bits));
    lines.push_back("tq_qjl_bits=" + std::to_string(metadata.qjl_bits));
    lines.push_back("tq_qjl_dim=" + std::to_string(metadata.qjl_dim));
    lines.push_back("tq_rotation_policy=" + metadata.rotation_policy);
    lines.push_back("tq_rotation_seed=" + std::to_string(metadata.rotation_seed));
    lines.push_back("tq_qjl_seed=" + std::to_string(metadata.qjl_seed));
    lines.push_back("tq_triality_mode=" + metadata.triality_mode);
    lines.push_back("tq_triality_view=" + metadata.triality_view);
    lines.push_back("tq_stage1_allocation_scheme=" + metadata.stage1_allocation_scheme);
    lines.push_back("tq_stage1_bitwidth_payload_dtype=" + metadata.stage1_bitwidth_payload_dtype);
    lines.push_back("tq_norm_dtype=" + metadata.norm_dtype);
    lines.push_back("tq_sign_pack_format=" + metadata.sign_pack_format);
    return lines;
}

bool parse_artifact_metadata(
    const std::vector<std::string> & metadata_lines,
    llama_turboquant_artifact_metadata & metadata,
    std::string * error) {
    std::map<std::string, std::string> values;
    for (const std::string & line : metadata_lines) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            return set_error(error, "invalid artifact metadata line without `=`");
        }
        values.emplace(line.substr(0, eq), line.substr(eq + 1));
    }

    if (!parse_u32(values, "tq_schema_version", metadata.schema_version, error) ||
        !parse_f32(values, "tq_total_bits", metadata.total_bits, error) ||
        !parse_f32(values, "tq_runtime_bits_per_channel", metadata.runtime_bits_per_channel, error) ||
        !parse_f32(values, "tq_stage1_effective_bits", metadata.stage1_effective_bits, error) ||
        !parse_u32(values, "tq_qjl_bits", metadata.qjl_bits, error) ||
        !parse_u32(values, "tq_qjl_dim", metadata.qjl_dim, error) ||
        !parse_string(values, "tq_rotation_policy", metadata.rotation_policy, error) ||
        !parse_u32(values, "tq_rotation_seed", metadata.rotation_seed, error) ||
        !parse_u32(values, "tq_qjl_seed", metadata.qjl_seed, error) ||
        !parse_string(values, "tq_triality_mode", metadata.triality_mode, error) ||
        !parse_string(values, "tq_triality_view", metadata.triality_view, error) ||
        !parse_string(values, "tq_stage1_allocation_scheme", metadata.stage1_allocation_scheme, error) ||
        !parse_string(values, "tq_stage1_bitwidth_payload_dtype", metadata.stage1_bitwidth_payload_dtype, error) ||
        !parse_string(values, "tq_norm_dtype", metadata.norm_dtype, error) ||
        !parse_string(values, "tq_sign_pack_format", metadata.sign_pack_format, error)) {
        return false;
    }

    metadata.triality_mode = canonical_runtime_mode(metadata.triality_mode);
    metadata.triality_view = canonical_triality_view(metadata.triality_view);
    if (!is_supported_runtime_mode(metadata.triality_mode) ||
        metadata.triality_mode == TRIALITY_RUNTIME_MODE_BEST_PER_LAYER) {
        return set_error(error, "artifact metadata uses unsupported triality mode");
    }
    if (!is_supported_triality_view(metadata.triality_view) ||
        !runtime_mode_matches_view(metadata.triality_mode, metadata.triality_view)) {
        return set_error(error, "artifact metadata uses unsupported triality view");
    }

    return validate_runtime_bits(
        metadata.stage1_effective_bits,
        metadata.qjl_bits,
        metadata.runtime_bits_per_channel,
        "artifact metadata",
        error);
}

bool read_required_f32_array(
    const gguf_context * ctx,
    const char * key,
    size_t expected_len,
    std::vector<float> & out,
    std::string * error) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return set_error(error, std::string("missing GGUF TurboQuant metadata key `") + key + "`");
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_FLOAT32) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must be a FLOAT32 array");
    }
    const size_t count = gguf_get_arr_n(ctx, key_id);
    if (count != expected_len) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must have length " + std::to_string(expected_len));
    }
    const float * data = static_cast<const float *>(gguf_get_arr_data(ctx, key_id));
    out.assign(data, data + count);
    return true;
}

bool read_required_u32_array(
    const gguf_context * ctx,
    const char * key,
    size_t expected_len,
    std::vector<uint32_t> & out,
    std::string * error) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return set_error(error, std::string("missing GGUF TurboQuant metadata key `") + key + "`");
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_UINT32) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must be a UINT32 array");
    }
    const size_t count = gguf_get_arr_n(ctx, key_id);
    if (count != expected_len) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must have length " + std::to_string(expected_len));
    }
    const uint32_t * data = static_cast<const uint32_t *>(gguf_get_arr_data(ctx, key_id));
    out.assign(data, data + count);
    return true;
}

bool read_required_str_array(
    const gguf_context * ctx,
    const char * key,
    size_t expected_len,
    std::vector<std::string> & out,
    std::string * error) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return set_error(error, std::string("missing GGUF TurboQuant metadata key `") + key + "`");
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_STRING) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must be a STRING array");
    }
    const size_t count = gguf_get_arr_n(ctx, key_id);
    if (count != expected_len) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must have length " + std::to_string(expected_len));
    }
    out.clear();
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, key_id, i));
    }
    return true;
}

bool read_optional_bool(
    const gguf_context * ctx,
    const char * key,
    bool & out) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return false;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_BOOL) {
        return false;
    }
    out = gguf_get_val_bool(ctx, key_id);
    return true;
}

bool read_optional_u64(
    const gguf_context * ctx,
    const char * key,
    uint64_t & out) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return false;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_UINT64) {
        return false;
    }
    out = gguf_get_val_u64(ctx, key_id);
    return true;
}

bool read_optional_u32(
    const gguf_context * ctx,
    const char * key,
    uint32_t & out) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return false;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_UINT32) {
        return false;
    }
    out = gguf_get_val_u32(ctx, key_id);
    return true;
}

bool read_optional_f32(
    const gguf_context * ctx,
    const char * key,
    float & out) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return false;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_FLOAT32) {
        return false;
    }
    out = gguf_get_val_f32(ctx, key_id);
    return true;
}

bool read_optional_string(
    const gguf_context * ctx,
    const char * key,
    std::string & out) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return false;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_STRING) {
        return false;
    }
    out = gguf_get_val_str(ctx, key_id);
    return true;
}

bool read_required_str_array_any(
    const gguf_context * ctx,
    const char * key,
    std::vector<std::string> & out,
    std::string * error) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0 || gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_STRING) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must be a STRING array");
    }
    const size_t count = gguf_get_arr_n(ctx, key_id);
    out.clear();
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, key_id, i));
    }
    return true;
}

bool read_required_bool(const gguf_context * ctx, const char * key, bool & out, std::string * error) {
    if (!read_optional_bool(ctx, key, out)) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must be BOOL");
    }
    return true;
}

bool read_required_u32(const gguf_context * ctx, const char * key, uint32_t & out, std::string * error) {
    if (!read_optional_u32(ctx, key, out)) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must be UINT32");
    }
    return true;
}

bool read_required_f32(const gguf_context * ctx, const char * key, float & out, std::string * error) {
    if (!read_optional_f32(ctx, key, out)) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must be FLOAT32");
    }
    return true;
}

bool read_required_string(const gguf_context * ctx, const char * key, std::string & out, std::string * error) {
    if (!read_optional_string(ctx, key, out)) {
        return set_error(error, std::string("GGUF TurboQuant key `") + key + "` must be STRING");
    }
    return true;
}

bool is_lower_sha256(const std::string & value) {
    if (value.size() != 64) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

uint32_t sha256_rotr(uint32_t value, uint32_t shift) {
    return (value >> shift) | (value << (32 - shift));
}

std::string sha256_hex(const std::string & input) {
    static constexpr uint32_t constants[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    std::vector<uint8_t> bytes(input.begin(), input.end());
    const uint64_t bit_count = static_cast<uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while ((bytes.size() % 64) != 56) {
        bytes.push_back(0);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        bytes.push_back(static_cast<uint8_t>(bit_count >> shift));
    }

    std::array<uint32_t, 8> state = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    for (size_t offset = 0; offset < bytes.size(); offset += 64) {
        uint32_t words[64] = {};
        for (uint32_t i = 0; i < 16; ++i) {
            const size_t index = offset + i * 4;
            words[i] = (uint32_t(bytes[index]) << 24) | (uint32_t(bytes[index + 1]) << 16) |
                (uint32_t(bytes[index + 2]) << 8) | uint32_t(bytes[index + 3]);
        }
        for (uint32_t i = 16; i < 64; ++i) {
            const uint32_t s0 = sha256_rotr(words[i - 15], 7) ^ sha256_rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
            const uint32_t s1 = sha256_rotr(words[i - 2], 17) ^ sha256_rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (uint32_t i = 0; i < 64; ++i) {
            const uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + s1 + choice + constants[i] + words[i];
            const uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < state.size(); ++i) {
        for (uint32_t byte = 0; byte < 4; ++byte) {
            const uint8_t value = static_cast<uint8_t>(state[i] >> (24 - byte * 8));
            result[i * 8 + byte * 2] = hex[value >> 4];
            result[i * 8 + byte * 2 + 1] = hex[value & 0x0f];
        }
    }
    return result;
}

bool require_f32_tensor(
        const gguf_context * ctx,
        const std::string & name,
        const std::vector<int64_t> & expected_shape,
        std::string * error) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name.c_str());
    if (tensor_id < 0) {
        return set_error(error, "missing GGUF TurboQuant tensor `" + name + "`");
    }
    if (gguf_get_tensor_type(ctx, tensor_id) != GGML_TYPE_F32) {
        return set_error(error, "GGUF TurboQuant tensor `" + name + "` must be F32");
    }
    if (gguf_get_tensor_n_dims(ctx, tensor_id) != static_cast<int>(expected_shape.size())) {
        return set_error(error, "GGUF TurboQuant tensor `" + name + "` has an invalid rank");
    }
    for (size_t dim = 0; dim < expected_shape.size(); ++dim) {
        if (gguf_get_tensor_ne(ctx, tensor_id, static_cast<int>(dim)) != expected_shape[dim]) {
            return set_error(error, "GGUF TurboQuant tensor `" + name + "` has an invalid shape");
        }
    }
    return true;
}

bool require_rotation_tensor(
        const gguf_context * ctx,
        const std::string & name,
        std::string * error) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name.c_str());
    if (tensor_id < 0) {
        return set_error(error, "missing GGUF TurboQuant tensor `" + name + "`");
    }
    if (gguf_get_tensor_type(ctx, tensor_id) != GGML_TYPE_F32) {
        return set_error(error, "GGUF TurboQuant tensor `" + name + "` must be F32");
    }
    if (gguf_get_tensor_n_dims(ctx, tensor_id) != 2) {
        return set_error(error, "GGUF TurboQuant rotation tensor `" + name + "` must be a matrix");
    }
    const int64_t rows = gguf_get_tensor_ne(ctx, tensor_id, 0);
    const int64_t columns = gguf_get_tensor_ne(ctx, tensor_id, 1);
    if (rows <= 0 || rows % 8 != 0 || rows != columns) {
        return set_error(error, "GGUF TurboQuant rotation tensor `" + name + "` must be a square multiple-of-8 matrix");
    }
    return true;
}

bool parse_execution(const std::string & value, llama_tq_execution & execution) {
    if (value == "single_view") {
        execution = LLAMA_TQ_EXEC_SINGLE_VIEW;
    } else if (value == "best_per_layer") {
        execution = LLAMA_TQ_EXEC_BEST_PER_LAYER;
    } else if (value == "attention_logit_consensus") {
        execution = LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS;
    } else if (value == "residual_parity") {
        execution = LLAMA_TQ_EXEC_RESIDUAL_PARITY;
    } else {
        return false;
    }
    return true;
}

bool parse_triality_schema_v2(
        const gguf_context * ctx,
        uint32_t n_layers,
        llama_turboquant_gguf_metadata & metadata,
        std::string * error) {
    uint32_t public_schema = 0;
    uint32_t view_count = 0;
    std::string execution_name;
    std::vector<std::string> views;
    std::vector<float> weights;
    std::vector<float> bias;
    std::vector<float> scale;
    std::vector<float> temperature;
    if (!read_required_u32(ctx, "hypura.turboquant.schema_version", public_schema, error) || public_schema != 2 ||
        !read_required_string(ctx, "hypura.turboquant.triality.profile_id", metadata.profile, error) ||
        !read_required_string(ctx, "hypura.turboquant.triality.execution", execution_name, error) ||
        !read_required_u32(ctx, "hypura.turboquant.triality.view_count", view_count, error) ||
        !read_required_str_array(ctx, "hypura.turboquant.triality.views", 3, views, error) ||
        !read_required_f32_array(ctx, "hypura.turboquant.triality.weights", static_cast<size_t>(n_layers) * 3, weights, error) ||
        !read_required_f32_array(ctx, "hypura.turboquant.triality.bias", static_cast<size_t>(n_layers) * 3, bias, error) ||
        !read_required_f32_array(ctx, "hypura.turboquant.triality.scale", static_cast<size_t>(n_layers) * 3, scale, error) ||
        !read_required_f32_array(ctx, "hypura.turboquant.triality.temperature", static_cast<size_t>(n_layers) * 3, temperature, error) ||
        !read_required_f32(ctx, "hypura.turboquant.triality.js_fallback_threshold", metadata.js_fallback_threshold, error)) {
        return false;
    }
    if (view_count != 3 || views != std::vector<std::string>{"vector", "spinor_plus_proxy", "spinor_minus_proxy"}) {
        return set_error(error, "Triality schema-v2 views must use the canonical three-view order");
    }
    if (metadata.profile.empty() || metadata.profile.size() >= 64 ||
        !std::all_of(metadata.profile.begin(), metadata.profile.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '-' || c == '.';
        })) {
        return set_error(error, "Triality schema-v2 profile_id is invalid");
    }
    if (!parse_execution(execution_name, metadata.execution)) {
        return set_error(error, "Triality schema-v2 execution mode is unsupported");
    }
    if (!std::isfinite(metadata.js_fallback_threshold) || metadata.js_fallback_threshold < 0.0f) {
        return set_error(error, "Triality schema-v2 js_fallback_threshold must be finite and non-negative");
    }

    metadata.consensus_layers.resize(n_layers);
    for (uint32_t layer_index = 0; layer_index < n_layers; ++layer_index) {
        auto & layer = metadata.consensus_layers[layer_index];
        layer.active_mask = metadata.execution == LLAMA_TQ_EXEC_SINGLE_VIEW ? 0x01 : 0x07;
        float row_sum = 0.0f;
        for (uint32_t branch_index = 0; branch_index < 3; ++branch_index) {
            const size_t index = static_cast<size_t>(layer_index) * 3 + branch_index;
            auto & branch = layer.branches[branch_index];
            branch.view = static_cast<llama_tq_view>(branch_index);
            branch.weight = weights[index];
            branch.bias = bias[index];
            branch.scale = scale[index];
            branch.temperature = temperature[index];
            if (!std::isfinite(branch.weight) || branch.weight < 0.0f || !std::isfinite(branch.bias) ||
                !std::isfinite(branch.scale) || branch.scale <= 0.0f || !std::isfinite(branch.temperature) ||
                branch.temperature <= 0.0f) {
                return set_error(error, "Triality schema-v2 consensus row contains invalid values");
            }
            row_sum += branch.weight;
            const std::string rotation_name = "turboquant.profile." + metadata.profile + ".layer." +
                std::to_string(layer_index) + ".rotation." + views[branch_index];
            if (!require_rotation_tensor(ctx, rotation_name, error)) {
                return false;
            }
        }
        if (std::fabs(row_sum - 1.0f) > 1e-6f) {
            return set_error(error, "Triality schema-v2 consensus weights must sum to one per layer");
        }
    }
    for (const char * field : {"weights", "bias", "scale", "temperature"}) {
        if (!require_f32_tensor(
                ctx,
                "turboquant.profile." + metadata.profile + ".consensus." + field,
                {3, n_layers},
                error)) {
            return false;
        }
    }
    metadata.three_view_bundle = true;

    if (!read_required_bool(ctx, "hypura.turboquant.ncka.enabled", metadata.ncka.enabled, error) ||
        !read_required_bool(ctx, "hypura.turboquant.ncka.required", metadata.ncka.required, error) ||
        !read_required_u32(ctx, "hypura.turboquant.ncka.schema_version", metadata.ncka.schema_version, error) ||
        !read_required_string(ctx, "hypura.turboquant.ncka.controller_type", metadata.ncka.controller_type, error) ||
        !read_required_str_array_any(ctx, "hypura.turboquant.ncka.coordinate_names", metadata.ncka.coordinate_names, error) ||
        !read_required_u32(ctx, "hypura.turboquant.ncka.outer_count", metadata.ncka.outer_count, error) ||
        !read_required_u32(ctx, "hypura.turboquant.ncka.knot_count", metadata.ncka.knot_count, error) ||
        !read_required_bool(ctx, "hypura.turboquant.ncka.s3_equivariant", metadata.ncka.s3_equivariant, error) ||
        !read_required_string(ctx, "hypura.turboquant.ncka.controller_sha256", metadata.ncka.controller_sha256, error) ||
        !read_required_string(ctx, "hypura.turboquant.ncka.normalisation_sha256", metadata.ncka.normalisation_sha256, error)) {
        return false;
    }
    if (metadata.ncka.enabled) {
        const bool supported = metadata.ncka.schema_version == 1 && metadata.ncka.controller_type == "finite_moment_ka_v1";
        if (!supported && metadata.ncka.required) {
            return set_error(error, "required NC-KA controller is unsupported");
        }
        if (metadata.ncka.coordinate_names.empty() || metadata.ncka.outer_count == 0 || metadata.ncka.knot_count < 2 ||
            !is_lower_sha256(metadata.ncka.controller_sha256) || !is_lower_sha256(metadata.ncka.normalisation_sha256)) {
            return set_error(error, "enabled NC-KA metadata is incomplete or invalid");
        }
        const size_t coordinate_count = metadata.ncka.coordinate_names.size();
        const std::string prefix = "turboquant.profile." + metadata.profile + ".ncka.";
        if (!require_f32_tensor(ctx, prefix + "fallback_weights", {3}, error)) {
            return false;
        }
        if (!supported) {
            metadata.ncka.static_fallback_selected = true;
        } else if (!require_f32_tensor(ctx, prefix + "coordinate_min", {static_cast<int64_t>(coordinate_count)}, error) ||
            !require_f32_tensor(ctx, prefix + "coordinate_max", {static_cast<int64_t>(coordinate_count)}, error) ||
            !require_f32_tensor(ctx, prefix + "inner_knots", {metadata.ncka.knot_count, static_cast<int64_t>(coordinate_count), metadata.ncka.outer_count, 3}, error) ||
            !require_f32_tensor(ctx, prefix + "inner_values", {metadata.ncka.knot_count, static_cast<int64_t>(coordinate_count), metadata.ncka.outer_count, 3}, error) ||
            !require_f32_tensor(ctx, prefix + "outer_knots", {metadata.ncka.knot_count, metadata.ncka.outer_count, 3}, error) ||
            !require_f32_tensor(ctx, prefix + "outer_values", {metadata.ncka.knot_count, metadata.ncka.outer_count, 3}, error)) {
            return false;
        }
    } else if (metadata.ncka.required) {
        return set_error(error, "NC-KA cannot be required when disabled");
    }

    if (!read_required_bool(ctx, "hypura.turboquant.urt.enabled", metadata.urt.enabled, error) ||
        !read_required_u32(ctx, "hypura.turboquant.urt.schema_version", metadata.urt.schema_version, error) ||
        !read_required_string(ctx, "hypura.turboquant.urt.abstract_algebra_id", metadata.urt.abstract_algebra_id, error) ||
        !read_required_string(ctx, "hypura.turboquant.urt.operator_word_manifest", metadata.urt.operator_word_manifest, error) ||
        !read_required_string(ctx, "hypura.turboquant.urt.operator_word_sha256", metadata.urt.operator_word_sha256, error) ||
        !read_required_string(ctx, "hypura.turboquant.urt.reference_representation", metadata.urt.reference_representation, error) ||
        !read_required_str_array_any(ctx, "hypura.turboquant.urt.supported_representations", metadata.urt.supported_representations, error) ||
        !read_required_f32(ctx, "hypura.turboquant.urt.consistency_tolerance", metadata.urt.consistency_tolerance, error) ||
        !read_required_u32(ctx, "hypura.turboquant.urt.moment_degree", metadata.urt.moment_degree, error) ||
        !read_required_string(ctx, "hypura.turboquant.urt.moment_manifest_sha256", metadata.urt.moment_manifest_sha256, error)) {
        return false;
    }
    if (metadata.urt.enabled) {
        if (metadata.urt.schema_version != 1 || metadata.urt.abstract_algebra_id.empty() ||
            !is_lower_sha256(metadata.urt.operator_word_sha256) ||
            sha256_hex(metadata.urt.operator_word_manifest) != metadata.urt.operator_word_sha256 ||
            metadata.urt.supported_representations.empty() ||
            std::find(metadata.urt.supported_representations.begin(), metadata.urt.supported_representations.end(), metadata.urt.reference_representation) == metadata.urt.supported_representations.end() ||
            !std::isfinite(metadata.urt.consistency_tolerance) || metadata.urt.consistency_tolerance <= 0.0f ||
            metadata.urt.moment_degree == 0 || !is_lower_sha256(metadata.urt.moment_manifest_sha256)) {
            return set_error(error, "enabled URT metadata is incomplete, unsupported, or hash-invalid");
        }
    }
    return true;
}
} // namespace

llama_turboquant_runtime_config llama_turboquant_runtime_from_env() {
    llama_turboquant_runtime_config cfg;
    cfg.enabled = env_flag("LLAMA_TURBOQUANT", false);
    cfg.mode = canonical_runtime_mode(env_string("LLAMA_TURBOQUANT_MODE", TRIALITY_RUNTIME_MODE));
    cfg.triality_view = canonical_triality_view(env_string("LLAMA_TURBOQUANT_TRIALITY_VIEW", "vector"));
    {
        const std::string default_cache_type_k = public_cache_type_k_from_runtime_mode(cfg.mode);
        const std::string default_cache_type_v = public_cache_type_v_from_mode(cfg.mode);
        cfg.cache_type_k = canonical_public_cache_type_k(
            env_string("LLAMA_TURBOQUANT_CACHE_TYPE_K", default_cache_type_k.c_str()));
        cfg.cache_type_v = canonical_public_cache_type_v(
            env_string("LLAMA_TURBOQUANT_CACHE_TYPE_V", default_cache_type_v.c_str()));
        if (cfg.cache_type_k != TURBOQUANT_PUBLIC_CACHE_TYPE_K_Q8_0 &&
            is_supported_public_cache_type_k(cfg.cache_type_k)) {
            cfg.mode = canonical_runtime_mode(cfg.cache_type_k);
        }
    }
    cfg.so8_enabled = env_flag("LLAMA_TURBOQUANT_SO8", true);
    cfg.so8_learned = env_flag("LLAMA_TURBOQUANT_SO8_LEARNED", false);
    cfg.triality_enabled = env_flag("LLAMA_TURBOQUANT_TRIALITY", true);
    cfg.require_artifact = env_flag("LLAMA_TURBOQUANT_REQUIRE_ARTIFACT", false);
    cfg.triality_mix = std::clamp(env_float("LLAMA_TURBOQUANT_TRIALITY_MIX", 0.5f), 0.0f, 1.0f);
    cfg.rotation_seed = env_u32("LLAMA_TURBOQUANT_ROTATION_SEED", 0);
    return cfg;
}

bool llama_turboquant_runtime_allows_k(const llama_turboquant_runtime_config & cfg) {
    if (!cfg.enabled) {
        return false;
    }
    const std::string canonical_mode = canonical_runtime_mode(cfg.mode);
    std::string cache_type_k = canonical_public_cache_type_k(cfg.cache_type_k);
    if (cache_type_k.empty()) {
        cache_type_k = public_cache_type_k_from_runtime_mode(canonical_mode);
    }
    return cache_type_k != TURBOQUANT_PUBLIC_CACHE_TYPE_K_Q8_0 &&
        is_supported_public_cache_type_k(cache_type_k) &&
        public_cache_types_match_runtime_mode(canonical_mode, cache_type_k);
}

namespace {
std::string canonical_triality_view(std::string view) {
    std::transform(view.begin(), view.end(), view.begin(), [](unsigned char c) {
        return c == '-' ? '_' : static_cast<char>(std::tolower(c));
    });
    if (view == "vector") {
        return "vector";
    }
    if (view == "plus" || view == "spinor_plus_proxy") {
        return "spinor_plus_proxy";
    }
    if (view == "minus" || view == "spinor_minus_proxy") {
        return "spinor_minus_proxy";
    }
    return view;
}

std::string canonical_runtime_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return c == '-' ? '_' : static_cast<char>(std::tolower(c));
    });
    if (mode == "triality_vector" || mode == "research_kv_split" || mode == TRIALITY_RUNTIME_MODE) {
        return TRIALITY_RUNTIME_MODE;
    }
    if (mode == "triality_plus" || mode == "spinor_plus_proxy" || mode == TRIALITY_RUNTIME_MODE_PLUS) {
        return TRIALITY_RUNTIME_MODE_PLUS;
    }
    if (mode == "triality_minus" || mode == "spinor_minus_proxy" || mode == TRIALITY_RUNTIME_MODE_MINUS) {
        return TRIALITY_RUNTIME_MODE_MINUS;
    }
    if (mode == "triality_best_per_layer" || mode == "best_per_layer" || mode == TRIALITY_RUNTIME_MODE_BEST_PER_LAYER) {
        return TRIALITY_RUNTIME_MODE_BEST_PER_LAYER;
    }
    return mode;
}

std::string canonical_public_cache_type_k(std::string cache_type_k) {
    std::transform(cache_type_k.begin(), cache_type_k.end(), cache_type_k.begin(), [](unsigned char c) {
        return c == '-' ? '_' : static_cast<char>(std::tolower(c));
    });
    if (cache_type_k == "triality_vector" || cache_type_k == "vector" || cache_type_k == "research_kv_split" || cache_type_k == TRIALITY_RUNTIME_MODE) {
        return TRIALITY_PUBLIC_CACHE_TYPE_K_VECTOR;
    }
    if (cache_type_k == "triality_plus" || cache_type_k == "spinor_plus_proxy" || cache_type_k == "plus" || cache_type_k == TRIALITY_RUNTIME_MODE_PLUS) {
        return TRIALITY_PUBLIC_CACHE_TYPE_K_PLUS;
    }
    if (cache_type_k == "triality_minus" || cache_type_k == "spinor_minus_proxy" || cache_type_k == "minus" || cache_type_k == TRIALITY_RUNTIME_MODE_MINUS) {
        return TRIALITY_PUBLIC_CACHE_TYPE_K_MINUS;
    }
    if (cache_type_k == "triality_best_per_layer" || cache_type_k == "best_per_layer" || cache_type_k == TRIALITY_RUNTIME_MODE_BEST_PER_LAYER) {
        return TRIALITY_PUBLIC_CACHE_TYPE_K_BEST_PER_LAYER;
    }
    if (cache_type_k == TURBOQUANT_PUBLIC_CACHE_TYPE_K_Q8_0) {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_K_Q8_0;
    }
    return cache_type_k;
}

std::string canonical_public_cache_type_v(std::string cache_type_v) {
    std::transform(cache_type_v.begin(), cache_type_v.end(), cache_type_v.begin(), [](unsigned char c) {
        return c == '-' ? '_' : static_cast<char>(std::tolower(c));
    });
    if (cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_Q8_0) {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_V_Q8_0;
    }
    if (cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO4) {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO4;
    }
    if (cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO3) {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO3;
    }
    if (cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO2) {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO2;
    }
    return cache_type_v;
}

std::string public_cache_type_k_from_runtime_mode(const std::string & mode) {
    const std::string canonical_mode = canonical_runtime_mode(mode);
    if (canonical_mode == TRIALITY_RUNTIME_MODE) {
        return TRIALITY_PUBLIC_CACHE_TYPE_K_VECTOR;
    }
    if (canonical_mode == TRIALITY_RUNTIME_MODE_PLUS) {
        return TRIALITY_PUBLIC_CACHE_TYPE_K_PLUS;
    }
    if (canonical_mode == TRIALITY_RUNTIME_MODE_MINUS) {
        return TRIALITY_PUBLIC_CACHE_TYPE_K_MINUS;
    }
    if (canonical_mode == TRIALITY_RUNTIME_MODE_BEST_PER_LAYER) {
        return TRIALITY_PUBLIC_CACHE_TYPE_K_BEST_PER_LAYER;
    }
    if (canonical_mode == "asym_q8_turbo4" || canonical_mode == "asym_q8_turbo3" || canonical_mode == "asym_q8_turbo2") {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_K_Q8_0;
    }
    return canonical_public_cache_type_k(canonical_mode);
}

std::string public_cache_type_v_from_mode(const std::string & mode) {
    const std::string canonical_mode = canonical_runtime_mode(mode);
    if (canonical_mode == "asym_q8_turbo4") {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO4;
    }
    if (canonical_mode == "asym_q8_turbo3") {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO3;
    }
    if (canonical_mode == "asym_q8_turbo2") {
        return TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO2;
    }
    return TURBOQUANT_PUBLIC_CACHE_TYPE_V_Q8_0;
}

bool is_supported_public_cache_type_k(const std::string & cache_type_k) {
    return cache_type_k == TRIALITY_PUBLIC_CACHE_TYPE_K_VECTOR ||
        cache_type_k == TRIALITY_PUBLIC_CACHE_TYPE_K_PLUS ||
        cache_type_k == TRIALITY_PUBLIC_CACHE_TYPE_K_MINUS ||
        cache_type_k == TRIALITY_PUBLIC_CACHE_TYPE_K_BEST_PER_LAYER ||
        cache_type_k == TURBOQUANT_PUBLIC_CACHE_TYPE_K_Q8_0;
}

bool is_supported_public_cache_type_v(const std::string & cache_type_v) {
    return cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_Q8_0 ||
        cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO4 ||
        cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO3 ||
        cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO2;
}

bool public_cache_types_match_runtime_mode(const std::string & mode, const std::string & cache_type_k) {
    const std::string canonical_mode = canonical_runtime_mode(mode);
    const std::string canonical_cache_type_k = canonical_public_cache_type_k(cache_type_k);
    if (canonical_mode == "asym_q8_turbo4" || canonical_mode == "asym_q8_turbo3" || canonical_mode == "asym_q8_turbo2") {
        return canonical_cache_type_k == TURBOQUANT_PUBLIC_CACHE_TYPE_K_Q8_0;
    }
    if (is_supported_runtime_mode(canonical_mode)) {
        return public_cache_type_k_from_runtime_mode(canonical_mode) == canonical_cache_type_k;
    }
    return false;
}

bool is_supported_triality_view(const std::string & view) {
    return view == "vector" || view == "spinor_plus_proxy" || view == "spinor_minus_proxy";
}

bool is_supported_runtime_mode(const std::string & mode) {
    return mode == TRIALITY_RUNTIME_MODE ||
        mode == TRIALITY_RUNTIME_MODE_PLUS ||
        mode == TRIALITY_RUNTIME_MODE_MINUS ||
        mode == TRIALITY_RUNTIME_MODE_BEST_PER_LAYER;
}

bool runtime_mode_matches_view(const std::string & mode, const std::string & view) {
    if (mode == TRIALITY_RUNTIME_MODE_BEST_PER_LAYER) {
        return is_supported_triality_view(view);
    }
    if (mode == TRIALITY_RUNTIME_MODE) {
        return view == "vector";
    }
    if (mode == TRIALITY_RUNTIME_MODE_PLUS) {
        return view == "spinor_plus_proxy";
    }
    if (mode == TRIALITY_RUNTIME_MODE_MINUS) {
        return view == "spinor_minus_proxy";
    }
    return false;
}
} // namespace

bool llama_turboquant_runtime_allows_v(const llama_turboquant_runtime_config & cfg) {
    if (!cfg.enabled) {
        return false;
    }
    std::string cache_type_v = canonical_public_cache_type_v(cfg.cache_type_v);
    if (cache_type_v.empty()) {
        cache_type_v = public_cache_type_v_from_mode(cfg.mode);
    }
    return cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO4 ||
        cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO3 ||
        cache_type_v == TURBOQUANT_PUBLIC_CACHE_TYPE_V_TURBO2 ||
        cfg.mode == "asym_q8_turbo4" ||
        cfg.mode == "asym_q8_turbo3" ||
        cfg.mode == "asym_q8_turbo2";
}

bool llama_turboquant_load_gguf_metadata(
    const gguf_context * ctx,
    uint32_t n_layers,
    llama_turboquant_gguf_metadata & metadata,
    std::string * error) {
    metadata = {};

    const int64_t schema_key = gguf_find_key(ctx, "tq_schema_version");
    if (schema_key < 0) {
        return true;
    }
    metadata.present = true;
    metadata.schema_version = gguf_get_val_u32(ctx, schema_key);
    if (metadata.schema_version != 1 && metadata.schema_version != 2) {
        metadata.present = false;
        return set_error(error, "unsupported GGUF TurboQuant schema version");
    }

    std::vector<float> total_bits;
    std::vector<float> runtime_bits_per_channel;
    std::vector<float> stage1_effective_bits;
    std::vector<uint32_t> qjl_bits;
    std::vector<uint32_t> qjl_dim;
    std::vector<std::string> rotation_policy;
    std::vector<uint32_t> rotation_seed;
    std::vector<uint32_t> qjl_seed;
    std::vector<std::string> triality_mode;
    std::vector<std::string> triality_view;
    std::vector<std::string> stage1_allocation_scheme;
    std::vector<std::string> stage1_bitwidth_payload_dtype;
    std::vector<std::string> norm_dtype;
    std::vector<std::string> sign_pack_format;

    const size_t expected_len = static_cast<size_t>(n_layers);
    if (!read_required_f32_array(ctx, "tq_total_bits", expected_len, total_bits, error) ||
        !read_required_f32_array(ctx, "tq_runtime_bits_per_channel", expected_len, runtime_bits_per_channel, error) ||
        !read_required_f32_array(ctx, "tq_stage1_effective_bits", expected_len, stage1_effective_bits, error) ||
        !read_required_u32_array(ctx, "tq_qjl_bits", expected_len, qjl_bits, error) ||
        !read_required_u32_array(ctx, "tq_qjl_dim", expected_len, qjl_dim, error) ||
        !read_required_str_array(ctx, "tq_rotation_policy", expected_len, rotation_policy, error) ||
        !read_required_u32_array(ctx, "tq_rotation_seed", expected_len, rotation_seed, error) ||
        !read_required_u32_array(ctx, "tq_qjl_seed", expected_len, qjl_seed, error) ||
        !read_required_str_array(ctx, "tq_triality_mode", expected_len, triality_mode, error) ||
        !read_required_str_array(ctx, "tq_triality_view", expected_len, triality_view, error) ||
        !read_required_str_array(ctx, "tq_stage1_allocation_scheme", expected_len, stage1_allocation_scheme, error) ||
        !read_required_str_array(ctx, "tq_stage1_bitwidth_payload_dtype", expected_len, stage1_bitwidth_payload_dtype, error) ||
        !read_required_str_array(ctx, "tq_norm_dtype", expected_len, norm_dtype, error) ||
        !read_required_str_array(ctx, "tq_sign_pack_format", expected_len, sign_pack_format, error)) {
        metadata.present = false;
        metadata.layers.clear();
        return false;
    }

    metadata.layers.resize(expected_len);
    for (size_t i = 0; i < expected_len; ++i) {
        auto & layer = metadata.layers[i];
        layer.total_bits = total_bits[i];
        layer.runtime_bits_per_channel = runtime_bits_per_channel[i];
        layer.stage1_effective_bits = stage1_effective_bits[i];
        layer.qjl_bits = qjl_bits[i];
        layer.qjl_dim = qjl_dim[i];
        layer.rotation_policy = rotation_policy[i];
        layer.rotation_seed = rotation_seed[i];
        layer.qjl_seed = qjl_seed[i];
        layer.triality_mode = canonical_runtime_mode(triality_mode[i]);
        layer.triality_view = canonical_triality_view(triality_view[i]);
        layer.stage1_allocation_scheme = stage1_allocation_scheme[i];
        layer.stage1_bitwidth_payload_dtype = stage1_bitwidth_payload_dtype[i];
        layer.norm_dtype = norm_dtype[i];
        layer.sign_pack_format = sign_pack_format[i];

        if (!validate_runtime_bits(
                layer.stage1_effective_bits,
                layer.qjl_bits,
                layer.runtime_bits_per_channel,
                std::string("GGUF TurboQuant metadata layer ") + std::to_string(i),
                error)) {
            metadata.present = false;
            metadata.layers.clear();
            return false;
        }
        if (!is_supported_runtime_mode(layer.triality_mode)) {
            metadata.present = false;
            metadata.layers.clear();
            return set_error(error, std::string("GGUF TurboQuant metadata layer ") + std::to_string(i) + " uses unsupported triality mode");
        }
        if (!is_supported_triality_view(layer.triality_view) ||
            !runtime_mode_matches_view(layer.triality_mode, layer.triality_view)) {
            metadata.present = false;
            metadata.layers.clear();
            return set_error(error, std::string("GGUF TurboQuant metadata layer ") + std::to_string(i) + " uses unsupported triality view");
        }
    }

    read_optional_bool(ctx, "hypura.turboquant.weight.enabled", metadata.weight.enabled);
    read_optional_string(ctx, "hypura.turboquant.weight.source_ftype", metadata.weight.source_ftype);
    read_optional_string(ctx, "hypura.turboquant.weight.policy", metadata.weight.policy);
    read_optional_string(ctx, "hypura.turboquant.weight.protected_roles", metadata.weight.protected_roles);
    read_optional_string(ctx, "hypura.turboquant.weight.protected_layers", metadata.weight.protected_layers);
    read_optional_string(ctx, "hypura.turboquant.weight.modality_scope", metadata.weight.modality_scope);
    read_optional_string(ctx, "hypura.turboquant.weight.payload_format", metadata.weight.payload_format);
    read_optional_u64(ctx, "hypura.turboquant.weight.payload_bytes", metadata.weight.payload_bytes);
    read_optional_string(ctx, "hypura.turboquant.weight.payload_json", metadata.weight.payload_json);

    std::string codec;
    uint32_t rotation_block_size = 0;
    float orthogonality_error = 0.0f;
    float determinant_error_max = 0.0f;
    std::string runtime_mode;
    std::string public_triality_view;
    std::string public_cache_type_k;
    std::string public_cache_type_v;
    bool view_bundle_complete = false;

    if (!read_optional_string(ctx, "hypura.turboquant.codec", codec) ||
        !read_optional_u32(ctx, "hypura.turboquant.rotation_block_size", rotation_block_size) ||
        !read_optional_f32(ctx, "hypura.turboquant.orthogonality_error", orthogonality_error) ||
        !read_optional_f32(ctx, "hypura.turboquant.determinant_error_max", determinant_error_max) ||
        !read_optional_string(ctx, "hypura.turboquant.runtime_mode", runtime_mode) ||
        !read_optional_string(ctx, "hypura.turboquant.triality_view", public_triality_view) ||
        !read_optional_string(ctx, "hypura.turboquant.cache_type_k", public_cache_type_k) ||
        !read_optional_string(ctx, "hypura.turboquant.cache_type_v", public_cache_type_v) ||
        !read_optional_bool(ctx, "hypura.turboquant.view_bundle_complete", view_bundle_complete)) {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "missing required hypura.turboquant shared ABI metadata");
    }

    if (codec != "tq4_1s") {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "hypura.turboquant.codec must be 'tq4_1s'");
    }
    if (rotation_block_size != TRIALITY_ROTATION_BLOCK_SIZE) {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "hypura.turboquant.rotation_block_size must equal 8");
    }
    if (orthogonality_error < 0.0f || determinant_error_max < 0.0f) {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "hypura.turboquant orthogonality/determinant metrics must be non-negative");
    }
    if (!view_bundle_complete) {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "hypura.turboquant.view_bundle_complete must be true");
    }

    runtime_mode = canonical_runtime_mode(runtime_mode);
    public_triality_view = canonical_triality_view(public_triality_view);
    public_cache_type_k = canonical_public_cache_type_k(public_cache_type_k);
    public_cache_type_v = canonical_public_cache_type_v(public_cache_type_v);
    if (!is_supported_runtime_mode(runtime_mode)) {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "unsupported hypura.turboquant.runtime_mode");
    }
    if (!is_supported_triality_view(public_triality_view) ||
        !runtime_mode_matches_view(runtime_mode, public_triality_view)) {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "hypura.turboquant triality mode/view pair is inconsistent");
    }
    if (!is_supported_public_cache_type_k(public_cache_type_k) ||
        !public_cache_types_match_runtime_mode(runtime_mode, public_cache_type_k)) {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "hypura.turboquant cache_type_k is inconsistent with runtime_mode");
    }
    if (!is_supported_public_cache_type_v(public_cache_type_v)) {
        metadata.present = false;
        metadata.layers.clear();
        return set_error(error, "hypura.turboquant cache_type_v is unsupported");
    }

    metadata.public_runtime_mode = runtime_mode;
    metadata.public_triality_view = public_triality_view;
    metadata.public_cache_type_k = public_cache_type_k;
    metadata.public_cache_type_v = public_cache_type_v;

    const int64_t public_schema_key = gguf_find_key(ctx, "hypura.turboquant.schema_version");
    if (metadata.schema_version == 2) {
        if (!parse_triality_schema_v2(ctx, n_layers, metadata, error)) {
            metadata = {};
            return false;
        }
    } else if (public_schema_key >= 0 &&
        (gguf_get_kv_type(ctx, public_schema_key) != GGUF_TYPE_UINT32 || gguf_get_val_u32(ctx, public_schema_key) != 1)) {
        metadata = {};
        return set_error(error, "legacy and public TurboQuant schema versions disagree");
    }

    return true;
}

bool llama_turboquant_validate_so8_rotation(
    const std::vector<float> & rotation_matrix,
    float atol,
    std::string * error) {
    if (rotation_matrix.size() != 64 || rotation_matrix.size() % 64 != 0) {
        return set_error(error, "SO(8) rotation must contain exactly one 8x8 block (64 coefficients)");
    }
    if (!(atol > 0.0f)) {
        return set_error(error, "SO(8) validation tolerance must be positive");
    }

    for (size_t i = 0; i < rotation_matrix.size(); ++i) {
        if (!std::isfinite(rotation_matrix[i])) {
            return set_error(error, "SO(8) rotation contains non-finite coefficients");
        }
    }

    for (uint32_t row = 0; row < 8; ++row) {
        double norm = 0.0;
        for (uint32_t col = 0; col < 8; ++col) {
            const double value = rotation_matrix[row * 8 + col];
            norm += value * value;
        }
        if (std::fabs(norm - 1.0) > atol) {
            return set_error(error, "SO(8) rotation row norm check failed");
        }
    }

    for (uint32_t col = 0; col < 8; ++col) {
        double norm = 0.0;
        for (uint32_t row = 0; row < 8; ++row) {
            const double value = rotation_matrix[row * 8 + col];
            norm += value * value;
        }
        if (std::fabs(norm - 1.0) > atol) {
            return set_error(error, "SO(8) rotation column norm check failed");
        }
    }

    for (uint32_t lhs = 0; lhs < 8; ++lhs) {
        for (uint32_t rhs = lhs + 1; rhs < 8; ++rhs) {
            double dot = 0.0;
            for (uint32_t col = 0; col < 8; ++col) {
                dot += static_cast<double>(rotation_matrix[lhs * 8 + col]) *
                       static_cast<double>(rotation_matrix[rhs * 8 + col]);
            }
            if (std::fabs(dot) > atol) {
                return set_error(error, "SO(8) rotation row orthogonality check failed");
            }
        }
    }

    double det_matrix[8][8];
    for (uint32_t row = 0; row < 8; ++row) {
        for (uint32_t col = 0; col < 8; ++col) {
            det_matrix[row][col] = static_cast<double>(rotation_matrix[row * 8 + col]);
        }
    }
    double det = 1.0;
    for (uint32_t pivot = 0; pivot < 8; ++pivot) {
        uint32_t best_row = pivot;
        double best_value = std::fabs(det_matrix[pivot][pivot]);
        for (uint32_t row = pivot + 1; row < 8; ++row) {
            const double candidate = std::fabs(det_matrix[row][pivot]);
            if (candidate > best_value) {
                best_value = candidate;
                best_row = row;
            }
        }
        if (best_value <= atol) {
            return set_error(error, "SO(8) rotation determinant check failed");
        }
        if (best_row != pivot) {
            for (uint32_t col = 0; col < 8; ++col) {
                std::swap(det_matrix[pivot][col], det_matrix[best_row][col]);
            }
            det = -det;
        }
        const double pivot_value = det_matrix[pivot][pivot];
        det *= pivot_value;
        for (uint32_t row = pivot + 1; row < 8; ++row) {
            const double factor = det_matrix[row][pivot] / pivot_value;
            for (uint32_t col = pivot; col < 8; ++col) {
                det_matrix[row][col] -= factor * det_matrix[pivot][col];
            }
        }
    }
    if (!std::isfinite(det) || std::fabs(det - 1.0) > atol) {
        return set_error(error, "SO(8) rotation determinant check failed");
    }

    return true;
}

void llama_turboquant_apply_so8_rotation(
    std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & rotation_matrix) {
    if (head_dim == 0 || n_vec == 0 || values.empty()) {
        return;
    }
    const uint32_t n_blocks = head_dim / 8;
    if (n_blocks == 0 || rotation_matrix.size() < 64) {
        return;
    }

    for (uint32_t i = 0; i < n_vec; ++i) {
        for (uint32_t b = 0; b < n_blocks; ++b) {
            float in[8];
            float out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const uint32_t off = i * head_dim + b * 8;
            for (uint32_t r = 0; r < 8; ++r) {
                in[r] = values[off + r];
            }
            for (uint32_t r = 0; r < 8; ++r) {
                for (uint32_t c = 0; c < 8; ++c) {
                    out[r] += rotation_matrix[r * 8 + c] * in[c];
                }
            }
            for (uint32_t r = 0; r < 8; ++r) {
                values[off + r] = out[r];
            }
        }
    }
}

std::vector<float> llama_turboquant_train_triality_codebook(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim) {
    if (n_vec == 0 || head_dim == 0 || values.size() < static_cast<size_t>(n_vec) * head_dim) {
        return {};
    }

    const uint32_t n_centroids = TRIALITY_CODEBOOK_SIZE;
    std::vector<float> codebook(static_cast<size_t>(n_centroids) * head_dim, 0.0f);
    for (uint32_t c = 0; c < n_centroids; ++c) {
        const uint32_t src = n_centroids > 1
            ? static_cast<uint32_t>((static_cast<uint64_t>(c) * (n_vec - 1)) / (n_centroids - 1))
            : 0;
        const size_t src_off = static_cast<size_t>(src) * head_dim;
        const size_t dst_off = static_cast<size_t>(c) * head_dim;
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(src_off), head_dim, codebook.begin() + static_cast<std::ptrdiff_t>(dst_off));
    }

    std::vector<uint32_t> assign(n_vec, 0);
    std::vector<float> next(codebook.size(), 0.0f);
    std::vector<uint32_t> counts(n_centroids, 0);

    for (uint32_t iter = 0; iter < 12; ++iter) {
        std::fill(next.begin(), next.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (uint32_t i = 0; i < n_vec; ++i) {
            const size_t off = static_cast<size_t>(i) * head_dim;
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_idx = 0;
            for (uint32_t c = 0; c < n_centroids; ++c) {
                const float dist = squared_l2_distance(values, off, codebook, static_cast<size_t>(c) * head_dim, head_dim);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = c;
                }
            }
            assign[i] = best_idx;
            counts[best_idx] += 1;
            const size_t dst_off = static_cast<size_t>(best_idx) * head_dim;
            for (uint32_t j = 0; j < head_dim; ++j) {
                next[dst_off + j] += values[off + j];
            }
        }

        for (uint32_t c = 0; c < n_centroids; ++c) {
            const size_t dst_off = static_cast<size_t>(c) * head_dim;
            if (counts[c] == 0) {
                continue;
            }
            const float inv_count = 1.0f / static_cast<float>(counts[c]);
            for (uint32_t j = 0; j < head_dim; ++j) {
                codebook[dst_off + j] = next[dst_off + j] * inv_count;
            }
        }
    }

    return codebook;
}

llama_turboquant_triality_metrics llama_turboquant_evaluate_triality(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & codebook) {
    llama_turboquant_triality_metrics m;
    if (n_vec == 0 || head_dim == 0 || codebook.size() < static_cast<size_t>(TRIALITY_CODEBOOK_SIZE) * head_dim) {
        return m;
    }

    double signal_power = 0.0;
    double mse_triality = 0.0;
    for (uint32_t i = 0; i < n_vec; ++i) {
        const size_t off = static_cast<size_t>(i) * head_dim;
        float best_dist = std::numeric_limits<float>::max();
        for (uint32_t j = 0; j < head_dim; ++j) {
            signal_power += sqr(values[off + j]);
        }
        for (uint32_t c = 0; c < TRIALITY_CODEBOOK_SIZE; ++c) {
            const float dist = squared_l2_distance(values, off, codebook, static_cast<size_t>(c) * head_dim, head_dim);
            best_dist = std::min(best_dist, dist);
        }
        mse_triality += best_dist;
    }
    const double denom = static_cast<double>(n_vec) * head_dim;
    m.exact_mse = static_cast<float>(signal_power / denom);
    m.triality_mse = static_cast<float>(mse_triality / denom);
    m.relative_mse_reduction = m.exact_mse > 0.0f ? (m.exact_mse - m.triality_mse) / m.exact_mse : 0.0f;
    return m;
}

std::vector<uint8_t> llama_turboquant_quantize_tq4_1s_reference(
    const std::vector<float> & values,
    std::string * error) {
    if (!validate_tq4_reference_shape(values.size(), "TQ4_1S input", error)) {
        return {};
    }

    const size_t n_blocks = values.size() / TQ4_1S_BLOCK_SIZE;
    std::vector<uint8_t> packed(n_blocks * TQ4_1S_BLOCK_BYTES, 0);

    static const float scale_candidates[9] = {0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f};

    for (size_t block = 0; block < n_blocks; ++block) {
        float buf[TQ4_1S_BLOCK_SIZE];
        const size_t block_offset = block * TQ4_1S_BLOCK_SIZE;
        for (uint32_t i = 0; i < TQ4_1S_BLOCK_SIZE; ++i) {
            buf[i] = values[block_offset + i];
        }
        tq4_rht_forward(buf);

        float rms0 = 0.0f;
        float rms1 = 0.0f;
        for (uint32_t i = 0; i < TQ4_1S_HALF_BLOCK; ++i) {
            rms0 += buf[i] * buf[i];
            rms1 += buf[i + TQ4_1S_HALF_BLOCK] * buf[i + TQ4_1S_HALF_BLOCK];
        }
        rms0 = std::sqrt(rms0 / static_cast<float>(TQ4_1S_HALF_BLOCK));
        rms1 = std::sqrt(rms1 / static_cast<float>(TQ4_1S_HALF_BLOCK));

        float best_d0 = rms0;
        float best_d1 = rms1;
        float best_err = std::numeric_limits<float>::max();

        for (float candidate : scale_candidates) {
            const float d0 = rms0 * candidate;
            const float d1 = rms1 * candidate;
            const float inv0 = d0 > TURBOQUANT_FLOAT_EPS ? 1.0f / d0 : 0.0f;
            const float inv1 = d1 > TURBOQUANT_FLOAT_EPS ? 1.0f / d1 : 0.0f;
            float err = 0.0f;
            for (uint32_t i = 0; i < TQ4_1S_HALF_BLOCK; ++i) {
                const uint8_t idx0 = tq4_choose_index(buf[i] * inv0);
                const uint8_t idx1 = tq4_choose_index(buf[i + TQ4_1S_HALF_BLOCK] * inv1);
                err += sqr(buf[i] - kTq4CentroidsWeight[idx0] * d0);
                err += sqr(buf[i + TQ4_1S_HALF_BLOCK] - kTq4CentroidsWeight[idx1] * d1);
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        for (uint32_t iter = 0; iter < 6; ++iter) {
            const float inv0 = best_d0 > TURBOQUANT_FLOAT_EPS ? 1.0f / best_d0 : 0.0f;
            const float inv1 = best_d1 > TURBOQUANT_FLOAT_EPS ? 1.0f / best_d1 : 0.0f;
            float num0 = 0.0f;
            float den0 = 0.0f;
            float num1 = 0.0f;
            float den1 = 0.0f;
            for (uint32_t i = 0; i < TQ4_1S_HALF_BLOCK; ++i) {
                const uint8_t idx0 = tq4_choose_index(buf[i] * inv0);
                const uint8_t idx1 = tq4_choose_index(buf[i + TQ4_1S_HALF_BLOCK] * inv1);
                const float c0 = kTq4CentroidsWeight[idx0];
                const float c1 = kTq4CentroidsWeight[idx1];
                num0 += buf[i] * c0;
                den0 += c0 * c0;
                num1 += buf[i + TQ4_1S_HALF_BLOCK] * c1;
                den1 += c1 * c1;
            }
            if (den0 > TURBOQUANT_FLOAT_EPS) {
                best_d0 = num0 / den0;
            }
            if (den1 > TURBOQUANT_FLOAT_EPS) {
                best_d1 = num1 / den1;
            }
        }

        const size_t byte_offset = block * TQ4_1S_BLOCK_BYTES;
        const ggml_fp16_t d0 = ggml_fp32_to_fp16(best_d0);
        const ggml_fp16_t d1 = ggml_fp32_to_fp16(best_d1);
        packed[byte_offset + 0] = static_cast<uint8_t>(d0 & 0xff);
        packed[byte_offset + 1] = static_cast<uint8_t>((d0 >> 8) & 0xff);
        packed[byte_offset + 2] = static_cast<uint8_t>(d1 & 0xff);
        packed[byte_offset + 3] = static_cast<uint8_t>((d1 >> 8) & 0xff);

        const float inv0 = best_d0 > TURBOQUANT_FLOAT_EPS ? 1.0f / best_d0 : 0.0f;
        const float inv1 = best_d1 > TURBOQUANT_FLOAT_EPS ? 1.0f / best_d1 : 0.0f;
        for (uint32_t i = 0; i < TQ4_1S_BLOCK_SIZE; ++i) {
            const float inv = i < TQ4_1S_HALF_BLOCK ? inv0 : inv1;
            const uint8_t idx = tq4_choose_index(buf[i] * inv);
            packed[byte_offset + 4 + (i / 2)] |= static_cast<uint8_t>((idx & 0x0F) << ((i & 1) * 4));
        }
    }

    return packed;
}

std::vector<float> llama_turboquant_dequantize_tq4_1s_reference(
    const std::vector<uint8_t> & packed_values,
    std::string * error) {
    if (packed_values.empty() || packed_values.size() % TQ4_1S_BLOCK_BYTES != 0) {
        set_error(error, "TQ4_1S packed buffer must be a non-zero multiple of 20 bytes");
        return {};
    }

    const size_t n_blocks = packed_values.size() / TQ4_1S_BLOCK_BYTES;
    std::vector<float> values(n_blocks * TQ4_1S_BLOCK_SIZE, 0.0f);

    for (size_t block = 0; block < n_blocks; ++block) {
        const size_t byte_offset = block * TQ4_1S_BLOCK_BYTES;
        const ggml_fp16_t d0_bits = static_cast<ggml_fp16_t>(
            static_cast<uint16_t>(packed_values[byte_offset + 0]) |
            (static_cast<uint16_t>(packed_values[byte_offset + 1]) << 8));
        const ggml_fp16_t d1_bits = static_cast<ggml_fp16_t>(
            static_cast<uint16_t>(packed_values[byte_offset + 2]) |
            (static_cast<uint16_t>(packed_values[byte_offset + 3]) << 8));
        const float d0 = ggml_fp16_to_fp32(d0_bits);
        const float d1 = ggml_fp16_to_fp32(d1_bits);

        float buf[TQ4_1S_BLOCK_SIZE];
        for (uint32_t i = 0; i < TQ4_1S_BLOCK_SIZE; ++i) {
            const uint8_t packed = packed_values[byte_offset + 4 + (i / 2)];
            const uint8_t idx = static_cast<uint8_t>((packed >> ((i & 1) * 4)) & 0x0F);
            const float d = i < TQ4_1S_HALF_BLOCK ? d0 : d1;
            buf[i] = kTq4CentroidsWeight[idx] * d;
        }
        tq4_rht_inverse(buf);

        const size_t value_offset = block * TQ4_1S_BLOCK_SIZE;
        for (uint32_t i = 0; i < TQ4_1S_BLOCK_SIZE; ++i) {
            values[value_offset + i] = buf[i];
        }
    }

    return values;
}

std::vector<float> llama_turboquant_mul_mat_tq4_1s_reference(
    const std::vector<uint8_t> & packed_weights,
    uint32_t n_rows,
    uint32_t n_cols,
    const std::vector<float> & activation,
    std::string * error) {
    if (n_rows == 0 || n_cols == 0 || n_cols % TQ4_1S_BLOCK_SIZE != 0) {
        set_error(error, "TQ4_1S matmul expects non-zero rows and cols divisible by 32");
        return {};
    }
    if (activation.size() != static_cast<size_t>(n_cols)) {
        set_error(error, "TQ4_1S matmul activation size must match n_cols");
        return {};
    }
    const size_t expected_bytes = static_cast<size_t>(n_rows) * (static_cast<size_t>(n_cols) / TQ4_1S_BLOCK_SIZE) * TQ4_1S_BLOCK_BYTES;
    if (packed_weights.size() != expected_bytes) {
        set_error(error, "TQ4_1S matmul packed weight size does not match n_rows*n_cols");
        return {};
    }

    std::vector<float> rotated = activation;
    for (uint32_t block = 0; block < n_cols / TQ4_1S_BLOCK_SIZE; ++block) {
        float * block_ptr = rotated.data() + static_cast<size_t>(block) * TQ4_1S_BLOCK_SIZE;
        tq4_rht_forward(block_ptr);
    }

    std::vector<float> output(n_rows, 0.0f);
    const size_t blocks_per_row = n_cols / TQ4_1S_BLOCK_SIZE;

    for (uint32_t row = 0; row < n_rows; ++row) {
        double sum = 0.0;
        for (size_t block = 0; block < blocks_per_row; ++block) {
            const size_t byte_offset = (static_cast<size_t>(row) * blocks_per_row + block) * TQ4_1S_BLOCK_BYTES;
            const ggml_fp16_t d0_bits = static_cast<ggml_fp16_t>(
                static_cast<uint16_t>(packed_weights[byte_offset + 0]) |
                (static_cast<uint16_t>(packed_weights[byte_offset + 1]) << 8));
            const ggml_fp16_t d1_bits = static_cast<ggml_fp16_t>(
                static_cast<uint16_t>(packed_weights[byte_offset + 2]) |
                (static_cast<uint16_t>(packed_weights[byte_offset + 3]) << 8));
            const float d0 = ggml_fp16_to_fp32(d0_bits);
            const float d1 = ggml_fp16_to_fp32(d1_bits);

            const size_t act_offset = block * TQ4_1S_BLOCK_SIZE;
            for (uint32_t i = 0; i < TQ4_1S_BLOCK_SIZE; ++i) {
                const uint8_t packed = packed_weights[byte_offset + 4 + (i / 2)];
                const uint8_t idx = static_cast<uint8_t>((packed >> ((i & 1) * 4)) & 0x0F);
                const float d = i < TQ4_1S_HALF_BLOCK ? d0 : d1;
                sum += static_cast<double>(kTq4CentroidsWeight[idx] * d) * rotated[act_offset + i];
            }
        }
        output[row] = static_cast<float>(sum);
    }

    return output;
}

bool llama_turboquant_save_artifact(
    const std::string & path,
    const llama_turboquant_artifact & artifact,
    std::string * error) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) {
            *error = "failed to open artifact for write";
        }
        return false;
    }

    out << "TQCUDA2\n";
    out << artifact.head_dim << "\n";
    out << (artifact.so8_learned ? 1 : 0) << "\n";
    out << (artifact.triality_enabled ? 1 : 0) << "\n";
    const std::vector<std::string> metadata_lines = serialize_metadata_lines(artifact.metadata);
    out << metadata_lines.size() << "\n";
    for (const std::string & line : metadata_lines) {
        out << line << "\n";
    }
    out << artifact.so8_rotation.size() << "\n";
    for (float v : artifact.so8_rotation) {
        out << v << "\n";
    }
    out << artifact.triality_codebook.size() << "\n";
    for (float v : artifact.triality_codebook) {
        out << v << "\n";
    }
    return true;
}

bool llama_turboquant_load_artifact(
    const std::string & path,
    llama_turboquant_artifact & artifact,
    std::string * error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) {
            *error = "failed to open artifact for read";
        }
        return false;
    }

    std::string magic;
    std::getline(in, magic);
    if (magic == "TQCUDA1") {
        return set_error(error, "legacy artifact metadata is missing; TQCUDA2 metadata is required");
    }
    if (magic != "TQCUDA2") {
        if (error) {
            *error = "invalid artifact magic";
        }
        return false;
    }

    uint32_t so8 = 0;
    uint32_t tri = 0;
    size_t metadata_count = 0;
    size_t rot_size = 0;
    size_t cb_size = 0;
    in >> artifact.head_dim;
    in >> so8;
    in >> tri;
    in >> metadata_count;
    artifact.so8_learned = so8 != 0;
    artifact.triality_enabled = tri != 0;
    std::vector<std::string> metadata_lines;
    metadata_lines.reserve(metadata_count);
    std::string line;
    std::getline(in, line);
    for (size_t i = 0; i < metadata_count; ++i) {
        if (!std::getline(in, line)) {
            return set_error(error, "artifact metadata parse failed");
        }
        metadata_lines.push_back(line);
    }
    if (!parse_artifact_metadata(metadata_lines, artifact.metadata, error)) {
        return false;
    }
    in >> rot_size;
    artifact.so8_rotation.resize(rot_size);
    for (size_t i = 0; i < rot_size; ++i) {
        in >> artifact.so8_rotation[i];
    }
    in >> cb_size;
    artifact.triality_codebook.resize(cb_size);
    for (size_t i = 0; i < cb_size; ++i) {
        in >> artifact.triality_codebook[i];
    }
    if (!in.good() && !in.eof()) {
        if (error) {
            *error = "artifact parse failed";
        }
        return false;
    }
    if (artifact.so8_learned) {
        if (!llama_turboquant_validate_so8_rotation(artifact.so8_rotation, 1e-3f, error)) {
            return false;
        }
    }
    if (artifact.triality_enabled) {
        if (artifact.head_dim == 0 || artifact.triality_codebook.size() != static_cast<size_t>(TRIALITY_CODEBOOK_SIZE) * artifact.head_dim) {
            return set_error(error, "triality codebook size must equal 3 * head_dim");
        }
    }
    return true;
}

namespace {

bool tq_set_api_error(llama_tq_error * err, llama_tq_error_code code, const std::string & message) {
    if (err) {
        err->code = code;
        std::snprintf(err->message, sizeof(err->message), "%s", message.c_str());
    }
    return false;
}

void tq_clear_api_error(llama_tq_error * err) {
    if (err) {
        err->code = LLAMA_TQ_ERROR_NONE;
        err->message[0] = '\0';
    }
}

bool tq_execution_is_valid(llama_tq_execution execution) {
    return execution >= LLAMA_TQ_EXEC_SINGLE_VIEW && execution <= LLAMA_TQ_EXEC_RESIDUAL_PARITY;
}

bool tq_view_is_valid(llama_tq_view view) {
    return view >= LLAMA_TQ_VIEW_VECTOR && view <= LLAMA_TQ_VIEW_SPINOR_MINUS_PROXY;
}

bool tq_validate_layer(const llama_tq_layer_config & layer, llama_tq_execution execution, size_t layer_index, llama_tq_error * err) {
    if (layer.active_mask == 0 || (layer.active_mask & ~uint8_t(0x07)) != 0) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "layer " + std::to_string(layer_index) + " has an invalid active mask");
    }

    uint32_t active_count = 0;
    float weight_sum = 0.0f;
    uint8_t seen_views = 0;
    for (uint32_t branch_index = 0; branch_index < 3; ++branch_index) {
        const llama_tq_branch_config & branch = layer.branches[branch_index];
        if (!tq_view_is_valid(branch.view)) {
            return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "layer " + std::to_string(layer_index) + " has an invalid view");
        }
        if (!std::isfinite(branch.weight) || !std::isfinite(branch.bias) || !std::isfinite(branch.scale) ||
            !std::isfinite(branch.temperature) || !std::isfinite(branch.expected_error) || branch.weight < 0.0f ||
            branch.scale <= 0.0f || branch.temperature <= 0.0f || branch.expected_error < 0.0f) {
            return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "layer " + std::to_string(layer_index) + " contains invalid branch values");
        }
        if ((layer.active_mask & (uint8_t(1) << branch_index)) == 0) {
            continue;
        }
        const uint8_t view_bit = uint8_t(1) << static_cast<uint8_t>(branch.view);
        if ((seen_views & view_bit) != 0) {
            return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "layer " + std::to_string(layer_index) + " repeats an active view");
        }
        seen_views |= view_bit;
        ++active_count;
        weight_sum += branch.weight;
    }

    if (std::fabs(weight_sum - 1.0f) > 1e-5f) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "layer " + std::to_string(layer_index) + " active weights must sum to one");
    }
    if (execution == LLAMA_TQ_EXEC_SINGLE_VIEW && active_count != 1) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "single-view execution requires exactly one active branch");
    }
    if ((execution == LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS || execution == LLAMA_TQ_EXEC_RESIDUAL_PARITY) && active_count != 3) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "consensus and residual-parity execution require all three branches");
    }
    return true;
}

}

llama_tq_context_state::llama_tq_context_state(size_t model_layer_count) : model_layer_count_(model_layer_count) {
}

bool llama_tq_context_state::configure(
        const llama_tq_context_config & cfg,
        bool initialization,
        llama_tq_error * err) {
    tq_clear_api_error(err);
    if (started_) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_CONTEXT_STARTED, "TurboQuant configuration is locked after encode or decode starts");
    }
    if (cfg.schema_version != 2) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "TurboQuant context schema_version must be 2");
    }
    if (!tq_execution_is_valid(cfg.execution)) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "TurboQuant execution mode is invalid");
    }
    if (cfg.n_layers == 0 || cfg.layers == nullptr) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "TurboQuant context requires at least one layer configuration");
    }
    if (model_layer_count_ != 0 && cfg.n_layers != model_layer_count_) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "TurboQuant layer count does not match the model");
    }
    if (!std::isfinite(cfg.js_fallback_threshold) || cfg.js_fallback_threshold < 0.0f) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_INVALID_CONFIG, "js_fallback_threshold must be finite and non-negative");
    }
    for (size_t i = 0; i < cfg.n_layers; ++i) {
        if (!tq_validate_layer(cfg.layers[i], cfg.execution, i, err)) {
            return false;
        }
    }

    const uint8_t required_view_capacity =
        cfg.execution == LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS || cfg.execution == LLAMA_TQ_EXEC_RESIDUAL_PARITY ? 3 : 1;
    if (!initialization && required_view_capacity > storage_view_capacity_) {
        return tq_set_api_error(
            err,
            LLAMA_TQ_ERROR_INVALID_CONFIG,
            "execution mode requires three-view storage reserved by llama_tq_init_from_model");
    }
    if (initialization) {
        storage_view_capacity_ = required_view_capacity;
    }

    llama_tq_owned_context_config next;
    next.schema_version = cfg.schema_version;
    next.execution = cfg.execution;
    next.layers.assign(cfg.layers, cfg.layers + cfg.n_layers);
    next.required = cfg.required;
    next.trace_enabled = cfg.trace_enabled;
    next.js_fallback_threshold = cfg.js_fallback_threshold;
    config_ = std::move(next);
    configured_ = true;
    return true;
}

bool llama_tq_context_state::get_config(
        llama_tq_context_config & out,
        llama_tq_layer_config * layer_storage,
        size_t layer_capacity,
        size_t & n_layers_required,
        llama_tq_error * err) const {
    tq_clear_api_error(err);
    if (!configured_) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_UNAVAILABLE, "TurboQuant context is not configured");
    }
    n_layers_required = config_.layers.size();
    if (layer_capacity < n_layers_required || (n_layers_required != 0 && layer_storage == nullptr)) {
        return tq_set_api_error(err, LLAMA_TQ_ERROR_BUFFER_TOO_SMALL, "layer_storage is smaller than the configured layer count");
    }
    std::copy(config_.layers.begin(), config_.layers.end(), layer_storage);
    out.schema_version = config_.schema_version;
    out.execution = config_.execution;
    out.layers = layer_storage;
    out.n_layers = config_.layers.size();
    out.required = config_.required;
    out.trace_enabled = config_.trace_enabled;
    out.js_fallback_threshold = config_.js_fallback_threshold;
    return true;
}

void llama_tq_context_state::mark_started() {
    started_ = true;
}

bool llama_tq_context_state::started() const {
    return started_;
}

bool llama_tq_context_state::trace_enabled() const {
    return configured_ && config_.trace_enabled;
}
