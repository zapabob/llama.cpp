#ifndef LLAMA_TURBOQUANT_H
#define LLAMA_TURBOQUANT_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

enum llama_tq_execution {
    LLAMA_TQ_EXEC_SINGLE_VIEW = 0,
    LLAMA_TQ_EXEC_BEST_PER_LAYER = 1,
    LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS = 2,
    LLAMA_TQ_EXEC_RESIDUAL_PARITY = 3,
};

enum llama_tq_view {
    LLAMA_TQ_VIEW_VECTOR = 0,
    LLAMA_TQ_VIEW_SPINOR_PLUS_PROXY = 1,
    LLAMA_TQ_VIEW_SPINOR_MINUS_PROXY = 2,
};

enum llama_tq_error_code {
    LLAMA_TQ_ERROR_NONE = 0,
    LLAMA_TQ_ERROR_INVALID_ARGUMENT = 1,
    LLAMA_TQ_ERROR_INVALID_CONFIG = 2,
    LLAMA_TQ_ERROR_CONTEXT_STARTED = 3,
    LLAMA_TQ_ERROR_UNAVAILABLE = 4,
    LLAMA_TQ_ERROR_BUFFER_TOO_SMALL = 5,
};

enum llama_tq_execution_capability {
    LLAMA_TQ_CAP_SINGLE_VIEW = 1u << LLAMA_TQ_EXEC_SINGLE_VIEW,
    LLAMA_TQ_CAP_BEST_PER_LAYER = 1u << LLAMA_TQ_EXEC_BEST_PER_LAYER,
    LLAMA_TQ_CAP_ATTENTION_LOGIT_CONSENSUS = 1u << LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS,
    LLAMA_TQ_CAP_RESIDUAL_PARITY = 1u << LLAMA_TQ_EXEC_RESIDUAL_PARITY,
};

struct llama_tq_branch_config {
    enum llama_tq_view view;
    float weight;
    float bias;
    float scale;
    float temperature;
    float expected_error;
    uint32_t bits_per_channel_milli;
};

struct llama_tq_layer_config {
    struct llama_tq_branch_config branches[3];
    uint8_t active_mask;
};

struct llama_tq_context_config {
    uint32_t schema_version;
    enum llama_tq_execution execution;
    const struct llama_tq_layer_config * layers;
    size_t n_layers;
    bool required;
    bool trace_enabled;
    float js_fallback_threshold;
};

struct llama_tq_error {
    int32_t code;
    char message[512];
};

struct llama_tq_model_capabilities {
    bool metadata_present;
    bool three_view_bundle;
    bool ncka_available;
    bool ncka_static_fallback_selected;
    bool urt_available;
    uint32_t schema_version;
    uint32_t n_layers;
    uint32_t supported_execution_mask;
    enum llama_tq_execution selected_execution;
    char profile_id[64];
};

// Creates a context after validating and deep-copying cfg. This is the only
// entry point that can reserve storage for three-view execution modes.
LLAMA_API struct llama_context * llama_tq_init_from_model(
        struct llama_model * model,
        struct llama_context_params params,
        const struct llama_tq_context_config * cfg,
        struct llama_tq_error * err);

// Updates a context before its first successful encode/decode. The context
// deep-copies cfg. The call fails if cfg needs more view storage than was
// reserved by llama_tq_init_from_model.
LLAMA_API bool llama_tq_context_configure(
        struct llama_context * ctx,
        const struct llama_tq_context_config * cfg,
        struct llama_tq_error * err);

// Copies the current configuration into caller-owned layer_storage. On a
// buffer-too-small result, n_layers_required receives the exact required size
// and no layer entries are copied.
LLAMA_API bool llama_tq_context_get_config(
        const struct llama_context * ctx,
        struct llama_tq_context_config * out,
        struct llama_tq_layer_config * layer_storage,
        size_t layer_capacity,
        size_t * n_layers_required,
        struct llama_tq_error * err);

// Returns a fixed-size, caller-owned summary of the parsed GGUF contract.
// Models without TurboQuant metadata return success with metadata_present=false.
LLAMA_API bool llama_tq_model_get_capabilities(
        const struct llama_model * model,
        struct llama_tq_model_capabilities * out,
        struct llama_tq_error * err);

#ifdef __cplusplus
}
#endif

#endif
