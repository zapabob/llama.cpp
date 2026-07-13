#pragma once

#include "../include/llama-turboquant-telemetry.h"

class llama_tq_telemetry_state {
public:
    void set_trace_enabled(bool enabled);
    void record(const llama_tq_consensus_metrics & metrics);
    bool get(llama_tq_consensus_metrics & out, llama_tq_error * err) const;
    void reset();

private:
    bool trace_enabled_ = false;
    bool has_metrics_ = false;
    llama_tq_consensus_metrics metrics_ = {};
};
