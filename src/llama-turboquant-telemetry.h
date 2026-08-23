#pragma once

#include "../include/llama-turboquant-telemetry.h"

struct ggml_tensor;
struct llama_tq_owned_context_config;

class llama_tq_telemetry_state {
public:
    void set_trace_enabled(bool enabled);
    void record(const llama_tq_consensus_metrics & metrics);
    bool trace_enabled() const;
    bool wants_graph_tensor(const ggml_tensor * tensor) const;
    bool record_graph_tensor(
        const ggml_tensor * tensor,
        const llama_tq_owned_context_config * config);
    bool get(llama_tq_consensus_metrics & out, llama_tq_error * err) const;
    void reset();

private:
    bool trace_enabled_ = false;
    bool has_metrics_ = false;
    llama_tq_consensus_metrics metrics_ = {};
};
