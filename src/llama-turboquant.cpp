#include "llama-turboquant.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
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
} // namespace

llama_turboquant_runtime_config llama_turboquant_runtime_from_env() {
    llama_turboquant_runtime_config cfg;
    cfg.enabled = env_flag("LLAMA_TURBOQUANT", false);
    cfg.mode = canonical_runtime_mode(env_string("LLAMA_TURBOQUANT_MODE", TRIALITY_RUNTIME_MODE));
    cfg.triality_view = canonical_triality_view(env_string("LLAMA_TURBOQUANT_TRIALITY_VIEW", "vector"));
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
    return is_supported_runtime_mode(canonical_runtime_mode(cfg.mode));
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
    return cfg.mode == "asym_q8_turbo4" || cfg.mode == "asym_q8_turbo3";
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
    bool view_bundle_complete = false;

    if (!read_optional_string(ctx, "hypura.turboquant.codec", codec) ||
        !read_optional_u32(ctx, "hypura.turboquant.rotation_block_size", rotation_block_size) ||
        !read_optional_f32(ctx, "hypura.turboquant.orthogonality_error", orthogonality_error) ||
        !read_optional_f32(ctx, "hypura.turboquant.determinant_error_max", determinant_error_max) ||
        !read_optional_string(ctx, "hypura.turboquant.runtime_mode", runtime_mode) ||
        !read_optional_string(ctx, "hypura.turboquant.triality_view", public_triality_view) ||
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
