#pragma once

#include <array>

#include "../include/llama-turboquant.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;
struct gguf_context;

// Compatibility note: legacy bridge surfaces still accept zapabob/TheTom-style
// aliases such as asym_q8_turbo4 even though the current production K-side
// canonical mode is key_only_block_so8_triality_vector.

struct llama_turboquant_runtime_config {
    bool enabled = false;
    std::string mode = "key_only_block_so8_triality_vector";
    std::string triality_view = "vector";
    std::string cache_type_k;
    std::string cache_type_v;
    bool so8_enabled = true;
    bool so8_learned = false;
    bool triality_enabled = true;
    bool require_artifact = false;
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

struct llama_tq_ncka_metadata {
    bool enabled = false;
    bool required = false;
    uint32_t schema_version = 0;
    std::string controller_type;
    std::vector<std::string> coordinate_names;
    uint32_t outer_count = 0;
    uint32_t knot_count = 0;
    bool s3_equivariant = false;
    std::string controller_sha256;
    std::string normalisation_sha256;
    bool static_fallback_selected = false;
};

struct llama_tq_urt_metadata {
    bool enabled = false;
    uint32_t schema_version = 0;
    std::string abstract_algebra_id;
    std::string operator_word_manifest;
    std::string operator_word_sha256;
    std::string reference_representation;
    std::vector<std::string> supported_representations;
    float consistency_tolerance = 0.0f;
    uint32_t moment_degree = 0;
    std::string moment_manifest_sha256;
};

struct llama_tq_residual_parity_layer_metadata {
    std::array<uint8_t, 3> sector_bits{ 0, 0, 0 };
    std::array<float, 3> fixed_sector_scales{ 0.0f, 0.0f, 0.0f };
    std::array<float, 2> beta{ 0.0f, 0.0f };
};

struct llama_tq_residual_parity_metadata {
    bool present = false;
    std::string mode;
    std::vector<llama_tq_residual_parity_layer_metadata> layers;
    uint32_t payload_bits_per_channel_milli = 0;
    uint32_t controller_bytes = 0;
};

struct llama_turboquant_gguf_metadata {
    bool present = false;
    uint32_t schema_version = 0;
    uint32_t public_schema_version = 0;
    std::vector<llama_turboquant_gguf_layer_metadata> layers;
    std::string public_runtime_mode;
    std::string public_triality_view;
    std::string public_cache_type_k;
    std::string public_cache_type_v;
    llama_turboquant_weight_gguf_metadata weight;
    std::string profile;
    bool three_view_bundle = false;
    llama_tq_execution execution = LLAMA_TQ_EXEC_SINGLE_VIEW;
    std::vector<llama_tq_layer_config> consensus_layers;
    float js_fallback_threshold = 0.0f;
    llama_tq_ncka_metadata ncka;
    llama_tq_urt_metadata urt;
    llama_tq_residual_parity_metadata residual_parity;
};

struct llama_tq_owned_context_config {
    uint32_t schema_version = 0;
    llama_tq_execution execution = LLAMA_TQ_EXEC_SINGLE_VIEW;
    std::vector<llama_tq_layer_config> layers;
    bool required = false;
    bool trace_enabled = false;
    float js_fallback_threshold = 0.0f;
    bool allow_identity_view_fallback = false;
};

class llama_tq_context_state {
public:
    explicit llama_tq_context_state(size_t model_layer_count = 0);

    bool configure(const llama_tq_context_config & cfg, bool initialization, llama_tq_error * err);
    bool get_config(
        llama_tq_context_config & out,
        llama_tq_layer_config * layer_storage,
        size_t layer_capacity,
        size_t & n_layers_required,
        llama_tq_error * err) const;
    void mark_started();
    bool started() const;
    bool trace_enabled() const;
    bool configured() const;
    uint8_t storage_view_capacity() const;
    uint64_t revision() const;
    const llama_tq_owned_context_config * config() const;

private:
    size_t model_layer_count_ = 0;
    bool started_ = false;
    bool configured_ = false;
    uint8_t storage_view_capacity_ = 1;
    llama_tq_execution storage_execution_ = LLAMA_TQ_EXEC_SINGLE_VIEW;
    uint64_t revision_ = 0;
    llama_tq_owned_context_config config_;
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

bool llama_turboquant_validate_so8_rotation(
    const std::vector<float> & rotation_matrix,
    float atol,
    std::string * error);

bool llama_turboquant_validate_so8_rotation_tensor(
    const ggml_tensor * tensor,
    float atol,
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

// Research-faithful reference codec for TQ4_1S weights.
std::vector<uint8_t> llama_turboquant_quantize_tq4_1s_reference(
    const std::vector<float> & values,
    std::string * error);

std::vector<float> llama_turboquant_dequantize_tq4_1s_reference(
    const std::vector<uint8_t> & packed_values,
    std::string * error);

// Native reference matvec: activation is pre-rotated once, weights remain packed.
std::vector<float> llama_turboquant_mul_mat_tq4_1s_reference(
    const std::vector<uint8_t> & packed_weights,
    uint32_t n_rows,
    uint32_t n_cols,
    const std::vector<float> & activation,
    std::string * error);

bool llama_turboquant_save_artifact(
    const std::string & path,
    const llama_turboquant_artifact & artifact,
    std::string * error);

bool llama_turboquant_load_artifact(
    const std::string & path,
    llama_turboquant_artifact & artifact,
    std::string * error);
