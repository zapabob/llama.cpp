#include "testing.h"

#include "llama-turboquant.h"
#include "../src/llama-model.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<llama_tq_layer_config> make_layers(size_t count, llama_tq_execution execution) {
    std::vector<llama_tq_layer_config> layers(count);
    for (llama_tq_layer_config & layer : layers) {
        layer.active_mask = execution == LLAMA_TQ_EXEC_SINGLE_VIEW ? 0x01 : 0x07;
        for (uint32_t branch = 0; branch < 3; ++branch) {
            layer.branches[branch].view = static_cast<llama_tq_view>(branch);
            layer.branches[branch].weight = execution == LLAMA_TQ_EXEC_SINGLE_VIEW ? (branch == 0 ? 1.0f : 0.0f) : 1.0f / 3.0f;
            layer.branches[branch].scale = 1.0f;
            layer.branches[branch].temperature = 1.0f;
        }
    }
    return layers;
}

llama_tq_context_config make_config(
        const std::vector<llama_tq_layer_config> & layers,
        llama_tq_execution execution,
        float js_threshold = 0.1f) {
    llama_tq_context_config config {};
    config.schema_version = 2;
    config.execution = execution;
    config.layers = layers.data();
    config.n_layers = layers.size();
    config.js_fallback_threshold = js_threshold;
    return config;
}

std::string environment_value(const char * name) {
    const char * value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

}

int main(int argc, char ** argv) {
    testing t;
    if (argc != 2) {
        return 1;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], model_params), llama_model_free);
    if (!model) {
        llama_backend_free();
        return 1;
    }
    const size_t n_layers = model->hparams.n_layer();
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;

    t.test("initialization_deep_copies_configuration_and_keeps_contexts_independent", [&](testing & t) {
        const std::string environment_before = environment_value("LLAMA_TURBOQUANT_MODE");
        auto first_layers = make_layers(n_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        auto second_layers = make_layers(n_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_context_config first_config = make_config(first_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS, 0.1f);
        llama_tq_context_config second_config = make_config(second_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS, 0.2f);
        llama_tq_error error {};
        std::unique_ptr<llama_context, decltype(&llama_free)> first(
            llama_tq_init_from_model(model.get(), context_params, &first_config, &error), llama_free);
        std::unique_ptr<llama_context, decltype(&llama_free)> second(
            llama_tq_init_from_model(model.get(), context_params, &second_config, &error), llama_free);
        t.assert_true("first context", first != nullptr);
        t.assert_true("second context", second != nullptr);
        first_layers[0].branches[0].weight = 0.9f;

        std::vector<llama_tq_layer_config> first_copy(n_layers);
        std::vector<llama_tq_layer_config> second_copy(n_layers);
        llama_tq_context_config first_out {};
        llama_tq_context_config second_out {};
        size_t required = 0;
        t.assert_true("first getter", llama_tq_context_get_config(first.get(), &first_out, first_copy.data(), first_copy.size(), &required, &error));
        t.assert_true("second getter", llama_tq_context_get_config(second.get(), &second_out, second_copy.data(), second_copy.size(), &required, &error));
        t.assert_true("deep copy", first_copy[0].branches[0].weight < 0.34f);
        t.assert_true("independent thresholds", first_out.js_fallback_threshold == 0.1f && second_out.js_fallback_threshold == 0.2f);
        t.assert_equal("environment unchanged", environment_before, environment_value("LLAMA_TURBOQUANT_MODE"));
    });

    t.test("getter_reports_required_capacity_without_copying", [&](testing & t) {
        auto layers = make_layers(n_layers, LLAMA_TQ_EXEC_SINGLE_VIEW);
        llama_tq_context_config config = make_config(layers, LLAMA_TQ_EXEC_SINGLE_VIEW);
        llama_tq_error error {};
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_tq_init_from_model(model.get(), context_params, &config, &error), llama_free);
        llama_tq_context_config out {};
        llama_tq_layer_config sentinel {};
        sentinel.active_mask = 0x55;
        size_t required = 0;
        t.assert_true("small buffer fails", !llama_tq_context_get_config(context.get(), &out, &sentinel, 1, &required, &error));
        t.assert_equal("buffer error", LLAMA_TQ_ERROR_BUFFER_TOO_SMALL, error.code);
        t.assert_equal("required count", n_layers, required);
        t.assert_equal("no partial copy", uint8_t(0x55), sentinel.active_mask);
    });

    t.test("post_init_configuration_cannot_expand_storage", [&](testing & t) {
        llama_tq_error error {};
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
        auto layers = make_layers(n_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_context_config config = make_config(layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        t.assert_true("expansion rejected", !llama_tq_context_configure(context.get(), &config, &error));
        t.assert_equal("invalid config", LLAMA_TQ_ERROR_INVALID_CONFIG, error.code);
    });

    t.test("required_unimplemented_execution_fails_closed", [&](testing & t) {
        auto layers = make_layers(n_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_context_config config = make_config(layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        config.required = true;
        llama_tq_error error {};
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_tq_init_from_model(model.get(), context_params, &config, &error), llama_free);
        t.assert_true("initialization rejected", context == nullptr);
        t.assert_equal("unavailable error", LLAMA_TQ_ERROR_UNAVAILABLE, error.code);

        std::unique_ptr<llama_context, decltype(&llama_free)> normal(
            llama_init_from_model(model.get(), context_params), llama_free);
        t.assert_true("post-init rejected", !llama_tq_context_configure(normal.get(), &config, &error));
        t.assert_equal("post-init unavailable", LLAMA_TQ_ERROR_UNAVAILABLE, error.code);
    });

    t.test("first_successful_decode_locks_configuration", [&](testing & t) {
        auto layers = make_layers(n_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_context_config config = make_config(layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_error error {};
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_tq_init_from_model(model.get(), context_params, &config, &error), llama_free);
        t.assert_true("context", context != nullptr);

        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        int32_t token_count = llama_tokenize(vocab, "hello", 5, nullptr, 0, true, true);
        t.assert_true("token sizing", token_count < 0);
        std::vector<llama_token> tokens(static_cast<size_t>(-token_count));
        token_count = llama_tokenize(vocab, "hello", 5, tokens.data(), tokens.size(), true, true);
        t.assert_true("tokenization", token_count > 0);
        llama_batch batch = llama_batch_get_one(tokens.data(), token_count);
        t.assert_equal("decode succeeds", int32_t(0), llama_decode(context.get(), batch));

        t.assert_true("configure locked", !llama_tq_context_configure(context.get(), &config, &error));
        t.assert_equal("started error", LLAMA_TQ_ERROR_CONTEXT_STARTED, error.code);
    });

    model.reset();
    llama_backend_free();
    return t.summary();
}
