#include "testing.h"

#include "llama-turboquant.h"
#include "ggml-backend.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-turboquant.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct backend_buffer_owner {
    ggml_backend_buffer_t value = nullptr;

    ~backend_buffer_owner() {
        if (value) {
            ggml_backend_buffer_free(value);
        }
    }
};

struct model_rotation_attachment_guard {
    llama_model * model = nullptr;

    ~model_rotation_attachment_guard() {
        if (model) {
            model->turboquant_rotations.clear();
        }
    }
};

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
        float js_threshold = 0.1f,
        bool allow_identity_view_fallback = true) {
    llama_tq_context_config config {};
    config.schema_version = 2;
    config.execution = execution;
    config.layers = layers.data();
    config.n_layers = layers.size();
    config.js_fallback_threshold = js_threshold;
    config.allow_identity_view_fallback = allow_identity_view_fallback;
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

    t.test("post_init_storage_kind_rejects_residual_before_any_configuration", [&](testing & t) {
        llama_tq_context_state state(n_layers);
        auto layers = make_layers(n_layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY);
        llama_tq_context_config config = make_config(layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY);
        llama_tq_error error {};
        t.assert_true("normal storage cannot become packed residual storage",
            !state.configure(config, false, &error));
        t.assert_equal("storage mismatch is invalid config", LLAMA_TQ_ERROR_INVALID_CONFIG, error.code);
        t.assert_true("storage mismatch is explicit",
            std::string(error.message).find("packed storage") != std::string::npos);
    });

    t.test("initialized_residual_storage_cannot_reconfigure_away", [&](testing & t) {
        llama_tq_context_state state(n_layers);
        auto residual_layers = make_layers(n_layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY);
        llama_tq_context_config residual = make_config(residual_layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY);
        llama_tq_error error {};
        t.assert_true("residual storage initialization succeeds", state.configure(residual, true, &error));

        auto single_layers = make_layers(n_layers, LLAMA_TQ_EXEC_SINGLE_VIEW);
        llama_tq_context_config single = make_config(single_layers, LLAMA_TQ_EXEC_SINGLE_VIEW);
        t.assert_true("packed residual storage cannot become normal storage",
            !state.configure(single, false, &error));
        t.assert_equal("reverse storage mismatch is invalid config", LLAMA_TQ_ERROR_INVALID_CONFIG, error.code);
    });

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

    t.test("required_residual_parity_execution_fails_closed", [&](testing & t) {
        llama_tq_model_capabilities capabilities {};
        auto layers = make_layers(n_layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY);
        llama_tq_context_config config = make_config(layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY);
        config.required = true;
        llama_tq_error error {};
        t.assert_true("capability query succeeds", llama_tq_model_get_capabilities(model.get(), &capabilities, &error));
        t.assert_true("residual parity is not advertised",
            (capabilities.supported_execution_mask & LLAMA_TQ_CAP_RESIDUAL_PARITY) == 0);
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_tq_init_from_model(model.get(), context_params, &config, &error), llama_free);
        t.assert_true("initialization rejected", context == nullptr);
        t.assert_equal("unavailable error", LLAMA_TQ_ERROR_UNAVAILABLE, error.code);
        t.assert_true("missing production metadata is explicit",
            std::string(error.message).find("canonical fixed-scale schema-v2 metadata") != std::string::npos);

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

    t.test("full_triple_turbo_shift_handles_padded_head_dimension", [&](testing & t) {
        auto layers = make_layers(n_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_context_config config = make_config(layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_context_params turbo_params = context_params;
        turbo_params.type_k = GGML_TYPE_TURBO3_0;
        llama_tq_error error {};
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_tq_init_from_model(model.get(), turbo_params, &config, &error), llama_free);
        t.assert_true("turbo context", context != nullptr);
        if (!context) {
            return;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        int32_t token_count = llama_tokenize(vocab, "hello", 5, nullptr, 0, true, true);
        std::vector<llama_token> tokens(static_cast<size_t>(-token_count));
        token_count = llama_tokenize(vocab, "hello", 5, tokens.data(), tokens.size(), true, true);
        llama_batch batch = llama_batch_get_one(tokens.data(), token_count);
        t.assert_equal("initial turbo decode succeeds", int32_t(0), llama_decode(context.get(), batch));

        llama_memory_seq_add(llama_get_memory(context.get()), 0, 0, -1, 1);
        batch = llama_batch_get_one(tokens.data(), 1);
        t.assert_equal("decode after triple-K shift succeeds", int32_t(0), llama_decode(context.get(), batch));
    });

    t.test("best_per_layer_shift_uses_the_selected_nonvector_view", [&](testing & t) {
        auto initial_layers = make_layers(n_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_context_config initial_config = make_config(
            initial_layers, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_error error {};
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_tq_init_from_model(model.get(), context_params, &initial_config, &error), llama_free);
        t.assert_true("three-view context", context != nullptr);
        if (!context) {
            return;
        }

        auto best_layers = make_layers(n_layers, LLAMA_TQ_EXEC_BEST_PER_LAYER);
        for (auto & layer : best_layers) {
            layer.branches[0].expected_error = 0.3f;
            layer.branches[1].expected_error = 0.1f;
            layer.branches[2].expected_error = 0.2f;
        }
        llama_tq_context_config best_config = make_config(best_layers, LLAMA_TQ_EXEC_BEST_PER_LAYER);
        t.assert_true("storage-compatible reconfiguration succeeds",
            llama_tq_context_configure(context.get(), &best_config, &error));

        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        int32_t token_count = llama_tokenize(vocab, "hello", 5, nullptr, 0, true, true);
        std::vector<llama_token> tokens(static_cast<size_t>(-token_count));
        token_count = llama_tokenize(vocab, "hello", 5, tokens.data(), tokens.size(), true, true);
        llama_batch batch = llama_batch_get_one(tokens.data(), token_count);
        t.assert_equal("initial selected-view decode succeeds", int32_t(0), llama_decode(context.get(), batch));

        llama_memory_seq_add(llama_get_memory(context.get()), 0, 0, -1, 1);
        batch = llama_batch_get_one(tokens.data(), 1);
        t.assert_equal("decode after selected-view shift succeeds", int32_t(0), llama_decode(context.get(), batch));
    });

    t.test("residual_parity_actual_context_state_multistream_and_shift", [&](testing & t) {
        const uint32_t head_dim = model->hparams.n_embd_head_k(0);
        ggml_init_params rotation_params {};
        rotation_params.mem_size = ggml_tensor_overhead() * n_layers * 3 + 4096;
        rotation_params.no_alloc = true;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> rotation_context(
            ggml_init(rotation_params), ggml_free);
        t.assert_true("rotation context", rotation_context != nullptr);
        if (!rotation_context) {
            return;
        }

        model->turboquant_rotations.assign(n_layers, {});
        model_rotation_attachment_guard rotation_attachment {model.get()};
        for (uint32_t il = 0; il < n_layers; ++il) {
            for (uint32_t view = 0; view < 3; ++view) {
                model->turboquant_rotations[il][view] = ggml_new_tensor_2d(
                    rotation_context.get(), GGML_TYPE_F32, head_dim, head_dim);
            }
        }
        backend_buffer_owner rotation_buffer;
        rotation_buffer.value = ggml_backend_alloc_ctx_tensors_from_buft(
            rotation_context.get(), ggml_backend_cpu_buffer_type());
        t.assert_true("rotation buffer", rotation_buffer.value != nullptr);
        if (!rotation_buffer.value) {
            return;
        }
        std::vector<float> identity(static_cast<size_t>(head_dim) * head_dim, 0.0f);
        for (uint32_t i = 0; i < head_dim; ++i) {
            identity[static_cast<size_t>(i) * head_dim + i] = 1.0f;
        }
        for (const auto & layer : model->turboquant_rotations) {
            for (ggml_tensor * rotation : layer) {
                ggml_backend_tensor_set(rotation, identity.data(), 0, identity.size() * sizeof(float));
            }
        }

        auto & metadata = model->turboquant_metadata;
        metadata.present = true;
        metadata.schema_version = 1;
        metadata.public_schema_version = 2;
        metadata.profile = "key_only_block_so8_triality_residual_parity";
        metadata.three_view_bundle = true;
        metadata.execution = LLAMA_TQ_EXEC_RESIDUAL_PARITY;
        metadata.layers.resize(n_layers);
        metadata.consensus_layers = make_layers(n_layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY);
        metadata.residual_parity.present = true;
        metadata.residual_parity.mode = metadata.profile;
        metadata.residual_parity.payload_bits_per_channel_milli = 5000;
        metadata.residual_parity.controller_bytes = 51;
        metadata.residual_parity.layers.resize(n_layers);
        for (auto & layer : metadata.residual_parity.layers) {
            layer.sector_bits = {3u, 1u, 1u};
            layer.fixed_sector_scales = {0.25f, 0.125f, 0.0625f};
            layer.beta = {0.5f, 0.25f};
        }

        llama_tq_model_capabilities capabilities {};
        llama_tq_error error {};
        t.assert_true("capability query", llama_tq_model_get_capabilities(model.get(), &capabilities, &error));
        t.assert_true("residual capability advertised",
            (capabilities.supported_execution_mask & LLAMA_TQ_CAP_RESIDUAL_PARITY) != 0);
        t.assert_equal("capability schema v2", uint32_t(2), capabilities.schema_version);
        t.assert_equal("capability selected execution", LLAMA_TQ_EXEC_RESIDUAL_PARITY,
            capabilities.selected_execution);

        auto layers = make_layers(n_layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY);
        llama_tq_context_config config = make_config(
            layers, LLAMA_TQ_EXEC_RESIDUAL_PARITY, 0.1f, false);
        config.required = true;
        llama_context_params residual_params = context_params;
        residual_params.n_ctx = 64;
        residual_params.n_batch = 8;
        residual_params.n_ubatch = 8;
        residual_params.n_seq_max = 2;
        residual_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        residual_params.offload_kqv = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_tq_init_from_model(model.get(), residual_params, &config, &error), llama_free);
        t.assert_true("actual residual context", context != nullptr);
        if (!context) {
            return;
        }

        auto * kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(context.get()));
        t.assert_true("actual memory is KV cache", kv != nullptr);
        llama_kv_cache::tq_residual_storage_info storage_info;
        t.assert_true("physical storage inspection", kv && kv->tq_get_residual_storage_info(0, storage_info));
        t.assert_equal("one physical K view", uint32_t(1), storage_info.physical_view_count);
        const uint32_t logical_channels = model->hparams.n_embd_k_gqa(0);
        const uint32_t expected_row_bytes = (logical_channels * 4u + 7u) / 8u;
        t.assert_equal("logical K channels", logical_channels, storage_info.logical_channels);
        t.assert_equal("packed row bytes", expected_row_bytes, storage_info.row_bytes);
        t.assert_equal("fixed controller bytes", size_t(51), storage_info.controller_bytes);
        t.assert_true("controller reserved bytes are zero", storage_info.controller_reserved_zero);
        t.assert_true("initial packed rows are canonical", storage_info.rows_canonical);
        const double logical_payload_bpc = metadata.residual_parity.payload_bits_per_channel_milli / 1000.0;
        t.assert_true("logical payload metadata is exactly five BPC",
            std::fabs(logical_payload_bpc - 5.0) < 1e-12);
        t.assert_true("storage inspection reports logical 3+1+1 as five BPC",
            std::fabs(storage_info.logical_payload_bits_per_channel - 5.0) < 1e-12);
        t.assert_true("physical payload follows parity-coupled four-bit row ceiling",
            std::fabs(storage_info.physical_payload_bits_per_channel -
                static_cast<double>(expected_row_bytes) * 8.0 /
                logical_channels) < 1e-12);
        t.assert_true("actual physical storage includes the separate controller",
            storage_info.actual_bits_per_channel > storage_info.physical_payload_bits_per_channel);
        t.assert_true("actual controller-inclusive storage meets five BPC",
            storage_info.actual_bits_per_channel <= 5.0 && storage_info.five_bit_target_met);

        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        int32_t token_count = llama_tokenize(vocab, "hello", 5, nullptr, 0, true, true);
        std::vector<llama_token> tokens(static_cast<size_t>(-token_count));
        token_count = llama_tokenize(vocab, "hello", 5, tokens.data(), tokens.size(), true, true);
        t.assert_true("integration tokenization", token_count > 0);
        const llama_token token = tokens[0];
        auto decode_two_streams = [&](llama_context * target, llama_pos pos) {
            llama_batch batch = llama_batch_init(2, 0, 2);
            batch.n_tokens = 2;
            for (int32_t i = 0; i < 2; ++i) {
                batch.token[i] = token;
                batch.pos[i] = pos;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = i;
                batch.logits[i] = i == 1;
            }
            const int32_t result = llama_decode(target, batch);
            llama_batch_free(batch);
            return result;
        };
        t.assert_equal("two-stream residual decode", int32_t(0), decode_two_streams(context.get(), 0));
        llama_memory_seq_rm(llama_get_memory(context.get()), 1, 0, -1);
        llama_memory_seq_cp(llama_get_memory(context.get()), 0, 1, 0, -1);
        t.assert_equal("decode after sequence copy", int32_t(0), decode_two_streams(context.get(), 1));
        llama_synchronize(context.get());
        t.assert_true("encoded rows remain canonical",
            kv->tq_get_residual_storage_info(0, storage_info) && storage_info.rows_canonical);

        const size_t state_size = llama_state_get_size(context.get());
        std::vector<uint8_t> state(state_size);
        t.assert_equal("state save size", state_size,
            llama_state_get_data(context.get(), state.data(), state.size()));

        std::unique_ptr<llama_context, decltype(&llama_free)> restored(
            llama_tq_init_from_model(model.get(), residual_params, &config, &error), llama_free);
        t.assert_true("restore context", restored != nullptr);
        if (!restored) {
            return;
        }
        t.assert_equal("state restore size", state_size,
            llama_state_set_data(restored.get(), state.data(), state.size()));
        t.assert_equal("decode after state restore", int32_t(0), decode_two_streams(restored.get(), 2));

        llama_memory_seq_add(llama_get_memory(restored.get()), 0, 0, -1, 1);
        llama_batch shifted_batch = llama_batch_init(1, 0, 2);
        shifted_batch.n_tokens = 1;
        shifted_batch.token[0] = token;
        shifted_batch.pos[0] = 4;
        shifted_batch.n_seq_id[0] = 1;
        shifted_batch.seq_id[0][0] = 0;
        shifted_batch.logits[0] = true;
        const int32_t shifted_result = llama_decode(restored.get(), shifted_batch);
        llama_batch_free(shifted_batch);
        t.assert_equal("decode after residual shift re-encode", int32_t(0), shifted_result);
        llama_synchronize(restored.get());
        auto * restored_kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(restored.get()));
        t.assert_true("restored KV inspection", restored_kv != nullptr);
        t.assert_true("shift re-encode keeps padding canonical",
            restored_kv && restored_kv->tq_get_residual_storage_info(0, storage_info) &&
            storage_info.rows_canonical && storage_info.controller_reserved_zero);

        const std::array<uint8_t, 4> marker {0x50u, 0x52u, 0x51u, 0x54u};
        auto marker_it = std::search(state.begin(), state.end(), marker.begin(), marker.end());
        t.assert_true("residual state marker present", marker_it != state.end());
        if (marker_it != state.end() && state.end() - marker_it > 12 + 44) {
            std::vector<uint8_t> corrupt = state;
            const size_t marker_offset = static_cast<size_t>(marker_it - state.begin());
            corrupt[marker_offset + 12 + 44] = 1u;
            std::unique_ptr<llama_context, decltype(&llama_free)> rejected(
                llama_tq_init_from_model(model.get(), residual_params, &config, &error), llama_free);
            t.assert_true("corruption test context", rejected != nullptr);
            if (rejected) {
                t.assert_true("controller reserved corruption fails closed",
                    llama_state_set_data(rejected.get(), corrupt.data(), corrupt.size()) != corrupt.size());
            }
        }
    });

    model.reset();
    llama_backend_free();
    return t.summary();
}
