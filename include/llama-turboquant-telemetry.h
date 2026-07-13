#ifndef LLAMA_TURBOQUANT_TELEMETRY_H
#define LLAMA_TURBOQUANT_TELEMETRY_H

#include "llama-turboquant.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_tq_branch_metrics {
    float logit_mean;
    float logit_variance;
    float logit_l2;
    float probability_entropy;
    float top1_probability;
    float orthogonality_error;
    float determinant_error;
    float expected_quantisation_error;
    uint64_t bytes_read;
    uint64_t duration_us;
};

struct llama_tq_consensus_metrics {
    struct llama_tq_branch_metrics branches[3];
    float pairwise_js[3];
    float mean_pairwise_js;
    float max_pairwise_js;
    float numerical_rank;
    float effective_rank;
    bool ka_fallback_used;
    uint64_t operator_word_hash_hi;
    uint64_t operator_word_hash_lo;
};

// Metrics contain aggregate numerical diagnostics only. Prompt, token and
// generated-text payloads are deliberately excluded from this ABI.
LLAMA_API bool llama_tq_context_get_last_metrics(
        const struct llama_context * ctx,
        struct llama_tq_consensus_metrics * out,
        struct llama_tq_error * err);

LLAMA_API void llama_tq_context_reset_metrics(struct llama_context * ctx);

#ifdef __cplusplus
}
#endif

#endif
