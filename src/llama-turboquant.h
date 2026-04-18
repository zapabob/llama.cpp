#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

struct llama_turboquant_runtime_config {
    bool enabled = false;
    std::string mode = "asym_q8_turbo4";
    bool so8_enabled = true;
    bool so8_learned = false;
    bool triality_enabled = true;
    float triality_mix = 0.5f;
    uint32_t rotation_seed = 0;
};

struct llama_turboquant_training_config {
    uint32_t head_dim = 128;
    uint32_t steps = 500;
    float lr = 1e-3f;
    uint32_t seed = 0;
    bool so8_learned = true;
    bool triality_enabled = true;
};

struct llama_turboquant_triality_metrics {
    float exact_mse = 0.0f;
    float triality_mse = 0.0f;
    float relative_mse_reduction = 0.0f;
};

struct llama_turboquant_artifact_metadata {
    uint32_t schema_version = 0;
    float total_bits = 0.0f;
    float runtime_bits_per_channel = 0.0f;
    float stage1_effective_bits = 0.0f;
    uint32_t qjl_bits = 0;
    uint32_t qjl_dim = 0;
    std::string rotation_policy;
    uint32_t rotation_seed = 0;
    uint32_t qjl_seed = 0;
    std::string triality_mode;
    std::string triality_view;
    std::string stage1_allocation_scheme;
    std::string stage1_bitwidth_payload_dtype;
    std::string norm_dtype;
    std::string sign_pack_format;
};

struct llama_turboquant_gguf_layer_metadata {
    float total_bits = 0.0f;
    float runtime_bits_per_channel = 0.0f;
    float stage1_effective_bits = 0.0f;
    uint32_t qjl_bits = 0;
    uint32_t qjl_dim = 0;
    std::string rotation_policy;
    uint32_t rotation_seed = 0;
    uint32_t qjl_seed = 0;
    std::string triality_mode;
    std::string triality_view;
    std::string stage1_allocation_scheme;
    std::string stage1_bitwidth_payload_dtype;
    std::string norm_dtype;
    std::string sign_pack_format;
};

struct llama_turboquant_weight_gguf_metadata {
    bool enabled = false;
    std::string source_ftype;
    std::string policy;
    std::string protected_roles;
    std::string protected_layers;
    std::string modality_scope;
    std::string payload_format;
    uint64_t payload_bytes = 0;
    std::string payload_json;
};

struct llama_turboquant_gguf_metadata {
    bool present = false;
    uint32_t schema_version = 0;
    std::vector<llama_turboquant_gguf_layer_metadata> layers;
    llama_turboquant_weight_gguf_metadata weight;
};

struct llama_turboquant_artifact {
    uint32_t head_dim = 0;
    bool so8_learned = false;
    bool triality_enabled = false;
    llama_turboquant_artifact_metadata metadata;
    std::vector<float> so8_rotation;
    std::vector<float> triality_codebook;
};

llama_turboquant_runtime_config llama_turboquant_runtime_from_env();
bool llama_turboquant_runtime_allows_k(const llama_turboquant_runtime_config & cfg);
bool llama_turboquant_runtime_allows_v(const llama_turboquant_runtime_config & cfg);
bool llama_turboquant_load_gguf_metadata(
    const struct gguf_context * ctx,
    uint32_t n_layers,
    llama_turboquant_gguf_metadata & metadata,
    std::string * error);

// Applies an in-place block-SO(8) rotation for vectors laid out as [n_vec, head_dim].
// If head_dim is not a multiple of 8, this function leaves the tail unchanged.
void llama_turboquant_apply_so8_rotation(
    std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & rotation_matrix);

// Produces a compact proxy codebook from training vectors using simple centroid slicing.
std::vector<float> llama_turboquant_train_triality_codebook(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim);

llama_turboquant_triality_metrics llama_turboquant_evaluate_triality(
    const std::vector<float> & values,
    uint32_t n_vec,
    uint32_t head_dim,
    const std::vector<float> & codebook);

bool llama_turboquant_save_artifact(
    const std::string & path,
    const llama_turboquant_artifact & artifact,
    std::string * error);

bool llama_turboquant_load_artifact(
    const std::string & path,
    llama_turboquant_artifact & artifact,
    std::string * error);
