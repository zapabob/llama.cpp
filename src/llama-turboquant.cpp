#include "llama-turboquant.h"
#include "gguf.h"

#include <algorithm>
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
} // namespace

llama_turboquant_runtime_config llama_turboquant_runtime_from_env() {
    llama_turboquant_runtime_config cfg;
    cfg.enabled = env_flag("LLAMA_TURBOQUANT", false);
    cfg.mode = env_string("LLAMA_TURBOQUANT_MODE", "asym_q8_turbo4");
    cfg.so8_enabled = env_flag("LLAMA_TURBOQUANT_SO8", true);
    cfg.so8_learned = env_flag("LLAMA_TURBOQUANT_SO8_LEARNED", false);
    cfg.triality_enabled = env_flag("LLAMA_TURBOQUANT_TRIALITY", true);
    cfg.triality_mix = std::clamp(env_float("LLAMA_TURBOQUANT_TRIALITY_MIX", 0.5f), 0.0f, 1.0f);
    cfg.rotation_seed = env_u32("LLAMA_TURBOQUANT_ROTATION_SEED", 0);
    return cfg;
}

bool llama_turboquant_runtime_allows_k(const llama_turboquant_runtime_config & cfg) {
    if (!cfg.enabled) {
        return false;
    }
    return cfg.mode == "triality_vector";
}

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
        layer.triality_mode = triality_mode[i];
        layer.triality_view = triality_view[i];
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

    const uint32_t n_centroids = 3;
    std::vector<float> codebook(static_cast<size_t>(n_centroids) * head_dim, 0.0f);
    const uint32_t stride = std::max(1u, n_vec / n_centroids);
    for (uint32_t c = 0; c < n_centroids; ++c) {
        const uint32_t src = std::min(n_vec - 1, c * stride);
        const size_t src_off = static_cast<size_t>(src) * head_dim;
        const size_t dst_off = static_cast<size_t>(c) * head_dim;
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(src_off), head_dim, codebook.begin() + static_cast<std::ptrdiff_t>(dst_off));
    }
    return codebook;
}

llama_turboquant_triality_metrics llama_turboquant_evaluate_triality(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & codebook) {
    llama_turboquant_triality_metrics m;
    if (n_vec == 0 || head_dim == 0 || codebook.size() < static_cast<size_t>(3) * head_dim) {
        return m;
    }

    double mse_exact = 0.0;
    double mse_triality = 0.0;
    for (uint32_t i = 0; i < n_vec; ++i) {
        const size_t off = static_cast<size_t>(i) * head_dim;
        for (uint32_t j = 0; j < head_dim; ++j) {
            const float v = values[off + j];
            mse_exact += sqr(v);
            float best = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < 3; ++c) {
                const float d = std::fabs(v - codebook[static_cast<size_t>(c) * head_dim + j]);
                best = std::min(best, d);
            }
            mse_triality += sqr(best);
        }
    }
    const double denom = static_cast<double>(n_vec) * head_dim;
    m.exact_mse = static_cast<float>(mse_exact / denom);
    m.triality_mse = static_cast<float>(mse_triality / denom);
    m.relative_mse_reduction = m.exact_mse > 0.0f ? (m.exact_mse - m.triality_mse) / m.exact_mse : 0.0f;
    return m;
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
    return true;
}
