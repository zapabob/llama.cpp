#include "llama-turboquant.h"
#include "llama-turboquant-telemetry.h"

#include <stddef.h>

int main(void) {
    struct llama_tq_error error = {0};
    struct llama_tq_context_config config = {0};
    struct llama_tq_model_capabilities capabilities = {0};
    struct llama_tq_consensus_metrics metrics = {0};
    size_t required = 0;

    if (llama_tq_context_configure(NULL, &config, &error)) {
        return 1;
    }
    if (llama_tq_context_get_config(NULL, &config, NULL, 0, &required, &error)) {
        return 2;
    }
    if (llama_tq_model_get_capabilities(NULL, &capabilities, &error)) {
        return 3;
    }
    if (llama_tq_context_get_last_metrics(NULL, &metrics, &error)) {
        return 4;
    }
    return 0;
}
