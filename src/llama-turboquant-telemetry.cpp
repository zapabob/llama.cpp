#include "llama-turboquant-telemetry.h"

#include <cstdio>

namespace {

bool tq_metrics_error(llama_tq_error * err, const char * message) {
    if (err) {
        err->code = LLAMA_TQ_ERROR_UNAVAILABLE;
        std::snprintf(err->message, sizeof(err->message), "%s", message);
    }
    return false;
}

void tq_metrics_clear_error(llama_tq_error * err) {
    if (err) {
        err->code = LLAMA_TQ_ERROR_NONE;
        err->message[0] = '\0';
    }
}

}

void llama_tq_telemetry_state::set_trace_enabled(bool enabled) {
    trace_enabled_ = enabled;
    if (!enabled) {
        reset();
    }
}

void llama_tq_telemetry_state::record(const llama_tq_consensus_metrics & metrics) {
    if (!trace_enabled_) {
        return;
    }
    metrics_ = metrics;
    has_metrics_ = true;
}

bool llama_tq_telemetry_state::get(llama_tq_consensus_metrics & out, llama_tq_error * err) const {
    tq_metrics_clear_error(err);
    if (!trace_enabled_) {
        return tq_metrics_error(err, "TurboQuant telemetry tracing is disabled");
    }
    if (!has_metrics_) {
        return tq_metrics_error(err, "TurboQuant telemetry has no recorded consensus metrics");
    }
    out = metrics_;
    return true;
}

void llama_tq_telemetry_state::reset() {
    metrics_ = {};
    has_metrics_ = false;
}
