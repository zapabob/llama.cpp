#include "testing.h"

#include "llama-turboquant-telemetry.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<llama_tq_layer_config> make_consensus_layers(size_t count) {
    std::vector<llama_tq_layer_config> layers(count);
    for (auto & layer : layers) {
        layer.active_mask = 0x07;
        for (uint32_t branch = 0; branch < 3; ++branch) {
            layer.branches[branch].view = static_cast<llama_tq_view>(branch);
            layer.branches[branch].weight = 1.0f / 3.0f;
            layer.branches[branch].scale = 1.0f;
            layer.branches[branch].temperature = 1.0f;
            layer.branches[branch].expected_error = 0.01f * static_cast<float>(branch + 1);
        }
    }
    return layers;
}

llama_tq_context_config make_config(
        const std::vector<llama_tq_layer_config> & layers,
        bool trace_enabled) {
    llama_tq_context_config config {};
    config.schema_version = 2;
    config.execution = LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS;
    config.layers = layers.data();
    config.n_layers = layers.size();
    config.trace_enabled = trace_enabled;
    config.js_fallback_threshold = 0.1f;
    config.allow_identity_view_fallback = true;
    return config;
}

std::vector<llama_token> tokenize(const llama_model * model, const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (count >= 0) {
        return {};
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
    if (count <= 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "SKIP: pass a local Triality schema-v2 GGUF to run live telemetry validation\n");
        return 0;
    }

    testing t;
    llama_backend_init();
    {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) {
            std::fprintf(stderr, "failed to load live telemetry fixture: %s\n", argv[1]);
            llama_backend_free();
            return 1;
        }

        const size_t n_layers = static_cast<size_t>(llama_model_n_layer(model.get()));
        auto layers = make_consensus_layers(n_layers);
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 64;
        context_params.n_batch = 64;
        context_params.n_ubatch = 64;
        context_params.n_threads = 2;
        context_params.n_threads_batch = 2;
        context_params.offload_kqv = false;

        t.test("null_layer_storage_is_rejected_before_preflight_dereference", [&](testing & t) {
            llama_tq_context_config invalid {};
            invalid.schema_version = 2;
            invalid.execution = LLAMA_TQ_EXEC_SINGLE_VIEW;
            invalid.layers = nullptr;
            invalid.n_layers = 1;
            invalid.allow_identity_view_fallback = true;
            llama_tq_error error {};
            std::unique_ptr<llama_context, decltype(&llama_free)> context(
                llama_tq_init_from_model(model.get(), context_params, &invalid, &error), llama_free);
            t.assert_true("invalid context rejected", context == nullptr);
            t.assert_equal("invalid config error", LLAMA_TQ_ERROR_INVALID_CONFIG, error.code);
        });

        t.test("successful_live_consensus_decode_records_actual_metrics", [&](testing & t) {
            llama_tq_context_config config = make_config(layers, true);
            llama_tq_error error {};
            std::unique_ptr<llama_context, decltype(&llama_free)> context(
                llama_tq_init_from_model(model.get(), context_params, &config, &error), llama_free);
            t.assert_true("trace context created", context != nullptr);
            if (!context) {
                return;
            }

            llama_tq_consensus_metrics metrics {};
            t.assert_true("no metrics before decode", !llama_tq_context_get_last_metrics(context.get(), &metrics, &error));

            auto tokens = tokenize(model.get(), "hello");
            t.assert_true("tokenization", !tokens.empty());
            if (tokens.empty()) {
                return;
            }
            llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
            t.assert_equal("decode succeeds", int32_t(0), llama_decode(context.get(), batch));
            t.assert_true("metrics available after decode", llama_tq_context_get_last_metrics(context.get(), &metrics, &error));
            t.assert_equal("metrics getter clears error", LLAMA_TQ_ERROR_NONE, error.code);
            for (uint32_t branch = 0; branch < 3; ++branch) {
                t.assert_true("branch mean finite", std::isfinite(metrics.branches[branch].logit_mean));
                t.assert_true("branch variance finite", std::isfinite(metrics.branches[branch].logit_variance));
                t.assert_true("branch l2 observed", metrics.branches[branch].logit_l2 > 0.0f);
                t.assert_true("branch entropy finite", std::isfinite(metrics.branches[branch].probability_entropy));
                t.assert_true("branch top1 probability valid",
                    metrics.branches[branch].top1_probability > 0.0f &&
                    metrics.branches[branch].top1_probability <= 1.0f);
                t.assert_true("configured expected error recorded",
                    std::fabs(metrics.branches[branch].expected_quantisation_error -
                        layers.back().branches[branch].expected_error) < 1.0e-6f);
            }
            t.assert_true("mean divergence finite and nonnegative",
                std::isfinite(metrics.mean_pairwise_js) && metrics.mean_pairwise_js >= 0.0f);
            t.assert_true("max divergence finite and nonnegative",
                std::isfinite(metrics.max_pairwise_js) && metrics.max_pairwise_js >= 0.0f);
            t.assert_true("numerical rank observed", metrics.numerical_rank >= 1.0f);
            t.assert_true("effective rank observed", metrics.effective_rank >= 1.0f);

            llama_tq_context_reset_metrics(context.get());
            t.assert_true("reset makes metrics unavailable",
                !llama_tq_context_get_last_metrics(context.get(), &metrics, &error));
            t.assert_equal("reset unavailable code", LLAMA_TQ_ERROR_UNAVAILABLE, error.code);

            t.assert_equal("same-shape decode succeeds after graph reuse", int32_t(0),
                llama_decode(context.get(), batch));
            t.assert_true("metrics available after graph reuse",
                llama_tq_context_get_last_metrics(context.get(), &metrics, &error));
            t.assert_equal("graph reuse getter clears error", LLAMA_TQ_ERROR_NONE, error.code);
            t.assert_true("graph reuse preserves observed metrics", metrics.effective_rank >= 1.0f);
        });

        t.test("trace_disabled_decode_does_not_publish_metrics", [&](testing & t) {
            llama_tq_context_config config = make_config(layers, false);
            llama_tq_error error {};
            std::unique_ptr<llama_context, decltype(&llama_free)> context(
                llama_tq_init_from_model(model.get(), context_params, &config, &error), llama_free);
            t.assert_true("non-trace context created", context != nullptr);
            if (!context) {
                return;
            }
            auto tokens = tokenize(model.get(), "hello");
            t.assert_true("tokenization", !tokens.empty());
            if (tokens.empty()) {
                return;
            }
            llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
            t.assert_equal("decode succeeds", int32_t(0), llama_decode(context.get(), batch));
            llama_tq_consensus_metrics metrics {};
            t.assert_true("disabled trace remains unavailable",
                !llama_tq_context_get_last_metrics(context.get(), &metrics, &error));
            t.assert_equal("disabled trace unavailable code", LLAMA_TQ_ERROR_UNAVAILABLE, error.code);
        });
    }
    llama_backend_free();
    return t.summary();
}
