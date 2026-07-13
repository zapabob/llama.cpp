#include "testing.h"

#include "../src/llama-turboquant-telemetry.h"

#include <cstring>

int main() {
    testing t;

    t.test("telemetry_requires_trace_and_a_recorded_sample", [](testing & t) {
        llama_tq_telemetry_state state;
        llama_tq_consensus_metrics out {};
        llama_tq_error error {};
        t.assert_true("disabled trace rejects", !state.get(out, &error));
        t.assert_equal("disabled error", LLAMA_TQ_ERROR_UNAVAILABLE, error.code);
        state.set_trace_enabled(true);
        t.assert_true("empty trace rejects", !state.get(out, &error));
        t.assert_equal("empty error", LLAMA_TQ_ERROR_UNAVAILABLE, error.code);
    });

    t.test("telemetry_record_round_trips_and_reset_removes_it", [](testing & t) {
        llama_tq_telemetry_state state;
        llama_tq_consensus_metrics sample {};
        sample.pairwise_js[1] = 0.125f;
        sample.operator_word_hash_lo = 0x1234;
        sample.operator_word_hash_hi = 0x5678;
        state.set_trace_enabled(true);
        state.record(sample);

        llama_tq_consensus_metrics out {};
        llama_tq_error error {};
        t.assert_true("record available", state.get(out, &error));
        t.assert_true("whole structure preserved", std::memcmp(&sample, &out, sizeof(sample)) == 0);
        state.reset();
        t.assert_true("reset removes sample", !state.get(out, &error));
        state.record(sample);
        state.set_trace_enabled(false);
        state.set_trace_enabled(true);
        t.assert_true("disable clears sample", !state.get(out, &error));
    });

    return t.summary();
}
