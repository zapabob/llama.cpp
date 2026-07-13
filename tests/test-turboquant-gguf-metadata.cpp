#include "testing.h"

#include "../src/llama-turboquant.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "ggml.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using gguf_ctx_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

struct backend_buffer_owner {
    ggml_backend_buffer_t value = nullptr;

    ~backend_buffer_owner() {
        if (value != nullptr) {
            ggml_backend_buffer_free(value);
        }
    }
};

static gguf_ctx_ptr make_ctx() {
    return gguf_ctx_ptr(gguf_init_empty(), gguf_free);
}

static void set_f32_array(gguf_context * ctx, const char * key, const std::vector<float> & values) {
    gguf_set_arr_data(ctx, key, GGUF_TYPE_FLOAT32, values.data(), values.size());
}

static void set_u32_array(gguf_context * ctx, const char * key, const std::vector<uint32_t> & values) {
    gguf_set_arr_data(ctx, key, GGUF_TYPE_UINT32, values.data(), values.size());
}

static void set_str_array(gguf_context * ctx, const char * key, const std::vector<std::string> & values) {
    std::vector<const char *> raw;
    raw.reserve(values.size());
    for (const std::string & value : values) {
        raw.push_back(value.c_str());
    }
    gguf_set_arr_str(ctx, key, raw.data(), raw.size());
}

static void populate_complete_metadata(gguf_context * ctx, uint32_t n_layers = 2, bool include_norm_dtype = true) {
    gguf_set_val_u32(ctx, "tq_schema_version", 1);

    std::vector<float> total_bits(n_layers, 3.5f);
    std::vector<float> runtime_bits(n_layers, 3.25f);
    std::vector<float> stage1_bits(n_layers, 2.25f);
    std::vector<uint32_t> qjl_bits(n_layers, 1);
    std::vector<uint32_t> qjl_dim(n_layers, 128);
    std::vector<std::string> rotation_policy(n_layers, "block_so8_learned");
    std::vector<uint32_t> rotation_seed(n_layers, 17);
    std::vector<uint32_t> qjl_seed(n_layers, 71);
    std::vector<std::string> triality_mode(n_layers, "research-kv-split");
    std::vector<std::string> triality_view(n_layers, "vector");
    std::vector<std::string> allocation_scheme(n_layers, "magnitude-topk");
    std::vector<std::string> bitwidth_payload_dtype(n_layers, "uint8");
    std::vector<std::string> norm_dtype(n_layers, "float32");
    std::vector<std::string> sign_pack_format(n_layers, "int8_unpacked_binary");

    set_f32_array(ctx, "tq_total_bits", total_bits);
    set_f32_array(ctx, "tq_runtime_bits_per_channel", runtime_bits);
    set_f32_array(ctx, "tq_stage1_effective_bits", stage1_bits);
    set_u32_array(ctx, "tq_qjl_bits", qjl_bits);
    set_u32_array(ctx, "tq_qjl_dim", qjl_dim);
    set_str_array(ctx, "tq_rotation_policy", rotation_policy);
    set_u32_array(ctx, "tq_rotation_seed", rotation_seed);
    set_u32_array(ctx, "tq_qjl_seed", qjl_seed);
    set_str_array(ctx, "tq_triality_mode", triality_mode);
    set_str_array(ctx, "tq_triality_view", triality_view);
    set_str_array(ctx, "tq_stage1_allocation_scheme", allocation_scheme);
    set_str_array(ctx, "tq_stage1_bitwidth_payload_dtype", bitwidth_payload_dtype);
    if (include_norm_dtype) {
        set_str_array(ctx, "tq_norm_dtype", norm_dtype);
    }
    set_str_array(ctx, "tq_sign_pack_format", sign_pack_format);
}

static void populate_public_weight_metadata(gguf_context * ctx) {
    gguf_set_val_str(ctx, "hypura.turboquant.codec", "tq4_1s");
    gguf_set_val_u32(ctx, "hypura.turboquant.rotation_block_size", 8);
    gguf_set_val_f32(ctx, "hypura.turboquant.orthogonality_error", 0.0f);
    gguf_set_val_f32(ctx, "hypura.turboquant.determinant_error_max", 0.0f);
    gguf_set_val_bool(ctx, "hypura.turboquant.view_bundle_complete", true);
    gguf_set_val_str(ctx, "hypura.turboquant.runtime_mode", "research-kv-split");
    gguf_set_val_str(ctx, "hypura.turboquant.triality_view", "vector");
    gguf_set_val_str(ctx, "hypura.turboquant.cache_type_k", "triality-vector");
    gguf_set_val_str(ctx, "hypura.turboquant.cache_type_v", "q8_0");
    gguf_set_val_bool(ctx, "hypura.turboquant.weight.enabled", true);
    gguf_set_val_str(ctx, "hypura.turboquant.weight.source_ftype", "q8_0");
    gguf_set_val_str(ctx, "hypura.turboquant.weight.policy", "qwen35-full-attention-ffn");
    gguf_set_val_str(
        ctx,
        "hypura.turboquant.weight.protected_roles",
        "[\"embedding\",\"norm\",\"output_head\",\"recurrent_state\"]");
    gguf_set_val_str(ctx, "hypura.turboquant.weight.protected_layers", "[0,1,30,31]");
    gguf_set_val_str(ctx, "hypura.turboquant.weight.modality_scope", "text-only");
    gguf_set_val_str(ctx, "hypura.turboquant.weight.payload_format", "json-inline-v1");
    gguf_set_val_u64(ctx, "hypura.turboquant.weight.payload_bytes", 32);
    gguf_set_val_str(ctx, "hypura.turboquant.weight.payload_json", "{\"enabled\":true}");
}

static void add_f32_tensor(gguf_context * ctx, const std::string & name, const std::vector<int64_t> & dimensions) {
    ggml_init_params params = {};
    params.mem_size = 64 * 1024;
    params.no_alloc = true;
    ggml_context * tensor_ctx = ggml_init(params);
    ggml_tensor * tensor = ggml_new_tensor(tensor_ctx, GGML_TYPE_F32, dimensions.size(), dimensions.data());
    ggml_set_name(tensor, name.c_str());
    gguf_add_tensor(ctx, tensor);
    ggml_free(tensor_ctx);
}

static void populate_schema_v2(
        gguf_context * ctx,
        uint32_t n_layers = 2,
        bool include_last_rotation = true,
        bool transpose_consensus = false,
        bool residual_parity = false,
        int64_t rotation_rows = 8,
        int64_t rotation_columns = 8) {
    populate_complete_metadata(ctx, n_layers);
    populate_public_weight_metadata(ctx);
    const std::string profile = residual_parity ?
        "key_only_block_so8_triality_residual_parity" : "balanced";
    const std::string tensor_namespace = residual_parity ?
        "turboquant.residual_parity" : "turboquant.profile." + profile;
    gguf_set_val_u32(ctx, "hypura.turboquant.schema_version", 2);
    gguf_set_val_str(ctx, "hypura.turboquant.triality.profile_id", profile.c_str());
    gguf_set_val_str(ctx, "hypura.turboquant.triality.execution",
        residual_parity ? "residual_parity" : "attention_logit_consensus");
    gguf_set_val_u32(ctx, "hypura.turboquant.triality.view_count", 3);
    set_str_array(ctx, "hypura.turboquant.triality.views", {"vector", "spinor_plus_proxy", "spinor_minus_proxy"});
    std::vector<float> weights(static_cast<size_t>(n_layers) * 3, 1.0f / 3.0f);
    std::vector<float> bias(static_cast<size_t>(n_layers) * 3, 0.0f);
    std::vector<float> scale(static_cast<size_t>(n_layers) * 3, 1.0f);
    std::vector<float> temperature(static_cast<size_t>(n_layers) * 3, 1.0f);
    set_f32_array(ctx, "hypura.turboquant.triality.weights", weights);
    set_f32_array(ctx, "hypura.turboquant.triality.bias", bias);
    set_f32_array(ctx, "hypura.turboquant.triality.scale", scale);
    set_f32_array(ctx, "hypura.turboquant.triality.temperature", temperature);
    gguf_set_val_f32(ctx, "hypura.turboquant.triality.js_fallback_threshold", 0.1f);

    const std::vector<std::string> views = {"vector", "spinor_plus_proxy", "spinor_minus_proxy"};
    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        for (size_t view = 0; view < views.size(); ++view) {
            if (!include_last_rotation && layer + 1 == n_layers && view + 1 == views.size()) {
                continue;
            }
            add_f32_tensor(
                ctx,
                tensor_namespace + ".layer." + std::to_string(layer) + ".rotation." + views[view],
                {rotation_rows, rotation_columns});
        }
    }
    for (const std::string & field : {"weights", "bias", "scale", "temperature"}) {
        add_f32_tensor(
            ctx,
            tensor_namespace + ".consensus." + field,
            transpose_consensus ? std::vector<int64_t>{n_layers, 3} : std::vector<int64_t>{3, n_layers});
    }

    if (residual_parity) {
        std::vector<uint32_t> sector_bits;
        std::vector<float> fixed_sector_scales;
        std::vector<float> beta;
        for (uint32_t layer = 0; layer < n_layers; ++layer) {
            sector_bits.insert(sector_bits.end(), {3u, 1u, 1u});
            fixed_sector_scales.insert(fixed_sector_scales.end(), {0.25f, 0.125f, 0.0625f});
            beta.insert(beta.end(), {0.5f, 0.25f});
        }
        gguf_set_val_str(ctx, "hypura.turboquant.triality.residual_parity.mode", profile.c_str());
        set_u32_array(ctx, "hypura.turboquant.triality.residual_parity.sector_bits", sector_bits);
        set_f32_array(ctx, "hypura.turboquant.triality.residual_parity.fixed_sector_scales", fixed_sector_scales);
        set_f32_array(ctx, "hypura.turboquant.triality.residual_parity.beta", beta);
        gguf_set_val_u32(ctx, "hypura.turboquant.triality.residual_parity.payload_bits_per_channel_milli", 5000);
        gguf_set_val_u32(ctx, "hypura.turboquant.triality.residual_parity.controller_bytes", 51);
    }

    gguf_set_val_bool(ctx, "hypura.turboquant.ncka.enabled", false);
    gguf_set_val_bool(ctx, "hypura.turboquant.ncka.required", false);
    gguf_set_val_u32(ctx, "hypura.turboquant.ncka.schema_version", 1);
    gguf_set_val_str(ctx, "hypura.turboquant.ncka.controller_type", "finite_moment_ka_v1");
    set_str_array(ctx, "hypura.turboquant.ncka.coordinate_names", {});
    gguf_set_val_u32(ctx, "hypura.turboquant.ncka.outer_count", 0);
    gguf_set_val_u32(ctx, "hypura.turboquant.ncka.knot_count", 0);
    gguf_set_val_bool(ctx, "hypura.turboquant.ncka.s3_equivariant", true);
    gguf_set_val_str(ctx, "hypura.turboquant.ncka.controller_sha256", "");
    gguf_set_val_str(ctx, "hypura.turboquant.ncka.normalisation_sha256", "");

    gguf_set_val_bool(ctx, "hypura.turboquant.urt.enabled", false);
    gguf_set_val_u32(ctx, "hypura.turboquant.urt.schema_version", 1);
    gguf_set_val_str(ctx, "hypura.turboquant.urt.abstract_algebra_id", "");
    gguf_set_val_str(ctx, "hypura.turboquant.urt.operator_word_manifest", "");
    gguf_set_val_str(ctx, "hypura.turboquant.urt.operator_word_sha256", "");
    gguf_set_val_str(ctx, "hypura.turboquant.urt.reference_representation", "");
    set_str_array(ctx, "hypura.turboquant.urt.supported_representations", {});
    gguf_set_val_f32(ctx, "hypura.turboquant.urt.consistency_tolerance", 0.0f);
    gguf_set_val_u32(ctx, "hypura.turboquant.urt.moment_degree", 0);
    gguf_set_val_str(ctx, "hypura.turboquant.urt.moment_manifest_sha256", "");
}

static llama_tq_layer_config make_context_layer(llama_tq_execution execution) {
    llama_tq_layer_config layer {};
    layer.active_mask = execution == LLAMA_TQ_EXEC_SINGLE_VIEW ? 0x01 : 0x07;
    for (uint32_t branch = 0; branch < 3; ++branch) {
        layer.branches[branch].view = static_cast<llama_tq_view>(branch);
        layer.branches[branch].weight = execution == LLAMA_TQ_EXEC_SINGLE_VIEW ?
            (branch == 0 ? 1.0f : 0.0f) : 1.0f / 3.0f;
        layer.branches[branch].scale = 1.0f;
        layer.branches[branch].temperature = 1.0f;
    }
    return layer;
}

static llama_tq_context_config make_context_config(
        const llama_tq_layer_config * layers,
        size_t n_layers,
        llama_tq_execution execution) {
    llama_tq_context_config config {};
    config.schema_version = 2;
    config.execution = execution;
    config.layers = layers;
    config.n_layers = n_layers;
    config.js_fallback_threshold = 0.1f;
    return config;
}

} // namespace

int main() {
    testing t;

    t.test("llama_turboquant_load_gguf_metadata_returns_not_present_when_keys_are_absent", [](testing & t) {
        auto ctx = make_ctx();
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("metadata absent", !metadata.present);
        t.assert_true("no layers", metadata.layers.empty());
    });

    t.test("llama_turboquant_load_gguf_metadata_parses_complete_layer_arrays", [](testing & t) {
        auto ctx = make_ctx();
        populate_complete_metadata(ctx.get(), 2);
        populate_public_weight_metadata(ctx.get());

        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("metadata present", metadata.present);
        t.assert_equal("schema version", uint32_t(1), metadata.schema_version);
        t.assert_equal("layer count", size_t(2), metadata.layers.size());
        t.assert_true("runtime bits preserved", std::fabs(metadata.layers[0].runtime_bits_per_channel - 3.25f) < 1e-6f);
        t.assert_equal("triality mode", std::string("key_only_block_so8_triality_vector"), metadata.layers[1].triality_mode);
        t.assert_equal("triality view", std::string("vector"), metadata.layers[1].triality_view);
        t.assert_true("weight metadata enabled", metadata.weight.enabled);
        t.assert_equal("weight source ftype", std::string("q8_0"), metadata.weight.source_ftype);
        t.assert_equal("weight policy", std::string("qwen35-full-attention-ffn"), metadata.weight.policy);
        t.assert_equal("public cache type k", std::string("triality-vector"), metadata.public_cache_type_k);
        t.assert_equal("public cache type v", std::string("q8_0"), metadata.public_cache_type_v);
        t.assert_equal("weight modality scope", std::string("text-only"), metadata.weight.modality_scope);
        t.assert_equal("weight payload format", std::string("json-inline-v1"), metadata.weight.payload_format);
        t.assert_equal("weight payload bytes", uint64_t(32), metadata.weight.payload_bytes);
    });

    t.test("llama_turboquant_load_gguf_metadata_rejects_missing_required_key", [](testing & t) {
        auto ctx = make_ctx();
        populate_complete_metadata(ctx.get(), 2, false);
        populate_public_weight_metadata(ctx.get());

        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("mentions missing key", error.find("tq_norm_dtype") != std::string::npos);
    });

    t.test("llama_turboquant_load_gguf_metadata_rejects_layer_length_mismatch", [](testing & t) {
        auto ctx = make_ctx();
        populate_complete_metadata(ctx.get(), 2);
        populate_public_weight_metadata(ctx.get());
        set_u32_array(ctx.get(), "tq_qjl_bits", std::vector<uint32_t>{1});

        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("mentions length", error.find("length 2") != std::string::npos);
    });

    t.test("llama_turboquant_load_gguf_metadata_rejects_inconsistent_runtime_bits", [](testing & t) {
        auto ctx = make_ctx();
        populate_complete_metadata(ctx.get(), 2);
        populate_public_weight_metadata(ctx.get());
        set_f32_array(ctx.get(), "tq_runtime_bits_per_channel", std::vector<float>{3.0f, 3.25f});

        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("mentions layer", error.find("layer 0") != std::string::npos);
        t.assert_true("mentions runtime bits", error.find("tq_runtime_bits_per_channel") != std::string::npos);
    });

    t.test("llama_turboquant_load_gguf_metadata_rejects_incomplete_shared_abi_metadata", [](testing & t) {
        auto ctx = make_ctx();
        populate_complete_metadata(ctx.get(), 2);
        gguf_set_val_str(ctx.get(), "hypura.turboquant.codec", "tq4_1s");
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.rotation_block_size", 8);
        gguf_set_val_f32(ctx.get(), "hypura.turboquant.orthogonality_error", 0.0f);
        gguf_set_val_f32(ctx.get(), "hypura.turboquant.determinant_error_max", 0.0f);

        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("mentions shared abi metadata", error.find("shared ABI metadata") != std::string::npos);
    });

    t.test("llama_turboquant_load_gguf_metadata_accepts_spinor_minus_and_best_per_layer", [](testing & t) {
        auto ctx = make_ctx();
        populate_complete_metadata(ctx.get(), 2);
        populate_public_weight_metadata(ctx.get());
        set_str_array(ctx.get(), "tq_triality_mode", std::vector<std::string>{"triality-minus", "triality-minus"});
        set_str_array(ctx.get(), "tq_triality_view", std::vector<std::string>{"minus", "minus"});
        gguf_set_val_str(ctx.get(), "hypura.turboquant.runtime_mode", "best_per_layer");
        gguf_set_val_str(ctx.get(), "hypura.turboquant.triality_view", "minus");
        gguf_set_val_str(ctx.get(), "hypura.turboquant.cache_type_k", "best_per_layer");
        gguf_set_val_str(ctx.get(), "hypura.turboquant.cache_type_v", "turbo4");

        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_equal("triality mode", std::string("key_only_block_so8_triality_minus"), metadata.layers[0].triality_mode);
        t.assert_equal("triality view", std::string("spinor_minus_proxy"), metadata.layers[0].triality_view);
        t.assert_equal("public cache type k", std::string("best_per_layer"), metadata.public_cache_type_k);
        t.assert_equal("public cache type v", std::string("turbo4"), metadata.public_cache_type_v);
    });

    t.test("schema_v2_accepts_the_complete_three_view_contract", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_equal("artifact schema remains v1", uint32_t(1), metadata.schema_version);
        t.assert_equal("public schema is v2", uint32_t(2), metadata.public_schema_version);
        t.assert_true("three-view bundle", metadata.three_view_bundle);
        t.assert_equal("profile", std::string("balanced"), metadata.profile);
        t.assert_equal("consensus layers", size_t(2), metadata.consensus_layers.size());
    });

    t.test("public_schema_v1_keeps_the_legacy_parser_path", [](testing & t) {
        auto ctx = make_ctx();
        populate_complete_metadata(ctx.get(), 2);
        populate_public_weight_metadata(ctx.get());
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.schema_version", 1);
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("three-view bundle is not inferred", !metadata.three_view_bundle);
        t.assert_equal("artifact schema remains v1", uint32_t(1), metadata.schema_version);
        t.assert_equal("public schema remains v1", uint32_t(1), metadata.public_schema_version);
    });

    t.test("schema_v2_accepts_canonical_residual_parity_metadata", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get(), 2, true, false, true);
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("residual metadata parses",
            llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_equal("public schema is v2", uint32_t(2), metadata.public_schema_version);
        t.assert_equal("execution is residual parity", LLAMA_TQ_EXEC_RESIDUAL_PARITY, metadata.execution);
        t.assert_true("residual production profile is present", metadata.residual_parity.present);
        t.assert_equal("residual layer count", size_t(2), metadata.residual_parity.layers.size());
        t.assert_equal("logical payload metadata is 5.000 BPC", uint32_t(5000),
            metadata.residual_parity.payload_bits_per_channel_milli);
        t.assert_equal("controller metadata is 51 bytes", uint32_t(51),
            metadata.residual_parity.controller_bytes);
        t.assert_equal("main sector is three bits", uint8_t(3),
            metadata.residual_parity.layers[0].sector_bits[0]);
        t.assert_equal("plus sector is one bit", uint8_t(1),
            metadata.residual_parity.layers[0].sector_bits[1]);
        t.assert_true("fixed model scale is preserved",
            std::fabs(metadata.residual_parity.layers[0].fixed_sector_scales[2] - 0.0625f) < 1e-7f);
        t.assert_true("model beta is preserved",
            std::fabs(metadata.residual_parity.layers[1].beta[1] - 0.25f) < 1e-7f);
    });

    t.test("schema_v2_residual_parity_fails_closed_on_noncanonical_accounting", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get(), 2, true, false, true);
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.triality.residual_parity.controller_bytes", 50);
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("controller size mismatch fails",
            !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("accounting failure is diagnostic", error.find("51 bytes") != std::string::npos);
        t.assert_true("failed metadata is not advertised", !metadata.residual_parity.present);
    });

    t.test("schema_v2_residual_parity_fails_closed_on_noncanonical_sector_bits", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get(), 2, true, false, true);
        set_u32_array(ctx.get(), "hypura.turboquant.triality.residual_parity.sector_bits",
            {3u, 2u, 1u, 3u, 1u, 1u});
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("sector bit mismatch fails",
            !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("sector failure is diagnostic", error.find("[3,1,1]") != std::string::npos);
        t.assert_true("failed metadata is not advertised", !metadata.residual_parity.present);
    });

    t.test("artifact_schema_v2_does_not_select_triality_v2", [](testing & t) {
        auto ctx = make_ctx();
        populate_complete_metadata(ctx.get(), 2);
        populate_public_weight_metadata(ctx.get());
        gguf_set_val_u32(ctx.get(), "tq_schema_version", 2);
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("artifact schema error", error.find("unsupported GGUF TurboQuant schema version") != std::string::npos);
        t.assert_true("metadata reset", !metadata.present);
    });

    t.test("public_schema_rejects_invalid_type_and_value", [](testing & t) {
        {
            auto ctx = make_ctx();
            populate_complete_metadata(ctx.get(), 2);
            populate_public_weight_metadata(ctx.get());
            gguf_set_val_str(ctx.get(), "hypura.turboquant.schema_version", "2");
            llama_turboquant_gguf_metadata metadata;
            std::string error;
            t.assert_true("string type fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
            t.assert_true("type error", error.find("must be uint32") != std::string::npos);
        }
        {
            auto ctx = make_ctx();
            populate_complete_metadata(ctx.get(), 2);
            populate_public_weight_metadata(ctx.get());
            gguf_set_val_u32(ctx.get(), "hypura.turboquant.schema_version", 3);
            llama_turboquant_gguf_metadata metadata;
            std::string error;
            t.assert_true("unknown value fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
            t.assert_true("value error", error.find("unsupported hypura.turboquant.schema_version") != std::string::npos);
        }
    });

    t.test("schema_v2_rejects_a_missing_rotation", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get(), 2, false);
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("metadata reset", !metadata.present);
        t.assert_true("rotation named", error.find("rotation.spinor_minus_proxy") != std::string::npos);
    });

    t.test("schema_v2_rejects_invalid_weight_rows", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        set_f32_array(ctx.get(), "hypura.turboquant.triality.weights", {0.5f, 0.5f, -0.1f, 0.3f, 0.3f, 0.4f});
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("negative weight fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
    });

    t.test("schema_v2_rejects_weight_rows_that_do_not_sum_to_one", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        set_f32_array(ctx.get(), "hypura.turboquant.triality.weights", {0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.3f});
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("row-sum mismatch fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("row-sum failure is diagnostic", error.find("sum to one") != std::string::npos);
    });

    t.test("schema_v2_rejects_zero_scale", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        set_f32_array(ctx.get(), "hypura.turboquant.triality.scale", {1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f});
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("zero scale fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("scale failure is diagnostic", error.find("invalid values") != std::string::npos);
    });

    t.test("schema_v2_rejects_invalid_temperature", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        set_f32_array(ctx.get(), "hypura.turboquant.triality.temperature", {
            1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("negative temperature fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("temperature failure is diagnostic", error.find("invalid values") != std::string::npos);
    });

    t.test("schema_v2_rejects_non_so8_rotation_shape", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get(), 2, true, false, false, 8, 16);
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("non-square rotation fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("rotation shape failure is diagnostic", error.find("square") != std::string::npos);
    });

    t.test("loaded_rotation_tensor_rejects_nonfinite_nonorthogonal_and_reflection_payloads", [](testing & t) {
        ggml_init_params params {};
        params.mem_size = ggml_tensor_overhead() + 4096;
        params.no_alloc = true;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> tensor_ctx(ggml_init(params), ggml_free);
        t.assert_true("tensor context", tensor_ctx != nullptr);
        if (!tensor_ctx) {
            return;
        }

        ggml_tensor * rotation = ggml_new_tensor_2d(tensor_ctx.get(), GGML_TYPE_F32, 8, 8);
        backend_buffer_owner buffer;
        buffer.value = ggml_backend_alloc_ctx_tensors_from_buft(tensor_ctx.get(), ggml_backend_cpu_buffer_type());
        t.assert_true("backend buffer", buffer.value != nullptr);
        if (!buffer.value) {
            return;
        }

        std::vector<float> matrix(64, 0.0f);
        for (size_t i = 0; i < 8; ++i) {
            matrix[i * 8 + i] = 1.0f;
        }
        std::string error;
        ggml_backend_tensor_set(rotation, matrix.data(), 0, matrix.size() * sizeof(float));
        t.assert_true("identity payload succeeds", llama_turboquant_validate_so8_rotation_tensor(rotation, 1e-6f, &error));

        matrix[1] = std::numeric_limits<float>::quiet_NaN();
        ggml_backend_tensor_set(rotation, matrix.data(), 0, matrix.size() * sizeof(float));
        error.clear();
        t.assert_true("non-finite payload fails", !llama_turboquant_validate_so8_rotation_tensor(rotation, 1e-6f, &error));
        t.assert_true("non-finite failure is diagnostic", error.find("non-finite") != std::string::npos);

        matrix[1] = 0.25f;
        ggml_backend_tensor_set(rotation, matrix.data(), 0, matrix.size() * sizeof(float));
        error.clear();
        t.assert_true("non-orthogonal payload fails", !llama_turboquant_validate_so8_rotation_tensor(rotation, 1e-6f, &error));
        t.assert_true("orthogonality failure is diagnostic",
            error.find("norm") != std::string::npos || error.find("orthogonality") != std::string::npos);

        matrix[1] = 0.0f;
        matrix[0] = -1.0f;
        ggml_backend_tensor_set(rotation, matrix.data(), 0, matrix.size() * sizeof(float));
        error.clear();
        t.assert_true("reflection payload fails", !llama_turboquant_validate_so8_rotation_tensor(rotation, 1e-6f, &error));
        t.assert_true("determinant failure is diagnostic", error.find("determinant") != std::string::npos);
    });

    t.test("schema_v2_rejects_transposed_consensus_tensors", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get(), 2, true, true);
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("shape error", error.find("invalid shape") != std::string::npos);
    });

    t.test("schema_v2_optional_unsupported_ncka_selects_static_fallback", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        gguf_set_val_bool(ctx.get(), "hypura.turboquant.ncka.enabled", true);
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.ncka.schema_version", 99);
        set_str_array(ctx.get(), "hypura.turboquant.ncka.coordinate_names", {"mean", "variance"});
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.ncka.outer_count", 1);
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.ncka.knot_count", 2);
        gguf_set_val_str(ctx.get(), "hypura.turboquant.ncka.controller_sha256", std::string(64, 'a').c_str());
        gguf_set_val_str(ctx.get(), "hypura.turboquant.ncka.normalisation_sha256", std::string(64, 'b').c_str());
        add_f32_tensor(ctx.get(), "turboquant.profile.balanced.ncka.fallback_weights", {3});
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("static fallback selected", metadata.ncka.static_fallback_selected);
    });

    t.test("schema_v2_required_unsupported_ncka_fails_closed", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        gguf_set_val_bool(ctx.get(), "hypura.turboquant.ncka.enabled", true);
        gguf_set_val_bool(ctx.get(), "hypura.turboquant.ncka.required", true);
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.ncka.schema_version", 99);
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("required error", error.find("required NC-KA") != std::string::npos);
    });

    t.test("schema_v2_accepts_supported_ncka_tensor_layout", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        gguf_set_val_bool(ctx.get(), "hypura.turboquant.ncka.enabled", true);
        set_str_array(ctx.get(), "hypura.turboquant.ncka.coordinate_names", {"mean", "variance"});
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.ncka.outer_count", 1);
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.ncka.knot_count", 2);
        gguf_set_val_str(ctx.get(), "hypura.turboquant.ncka.controller_sha256", std::string(64, 'a').c_str());
        gguf_set_val_str(ctx.get(), "hypura.turboquant.ncka.normalisation_sha256", std::string(64, 'b').c_str());
        add_f32_tensor(ctx.get(), "turboquant.profile.balanced.ncka.fallback_weights", {3});
        add_f32_tensor(ctx.get(), "turboquant.profile.balanced.ncka.coordinate_min", {2});
        add_f32_tensor(ctx.get(), "turboquant.profile.balanced.ncka.coordinate_max", {2});
        add_f32_tensor(ctx.get(), "turboquant.profile.balanced.ncka.inner_knots", {2, 2, 1, 3});
        add_f32_tensor(ctx.get(), "turboquant.profile.balanced.ncka.inner_values", {2, 2, 1, 3});
        add_f32_tensor(ctx.get(), "turboquant.profile.balanced.ncka.outer_knots", {2, 1, 3});
        add_f32_tensor(ctx.get(), "turboquant.profile.balanced.ncka.outer_values", {2, 1, 3});
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("controller active", metadata.ncka.enabled && !metadata.ncka.static_fallback_selected);
    });

    t.test("schema_v2_urt_rejects_a_manifest_hash_mismatch", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        gguf_set_val_bool(ctx.get(), "hypura.turboquant.urt.enabled", true);
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.abstract_algebra_id", "triality-v1");
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.operator_word_manifest", "{}");
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.operator_word_sha256", std::string(64, '0').c_str());
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.reference_representation", "vector");
        set_str_array(ctx.get(), "hypura.turboquant.urt.supported_representations", {"vector"});
        gguf_set_val_f32(ctx.get(), "hypura.turboquant.urt.consistency_tolerance", 1e-4f);
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.urt.moment_degree", 2);
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.moment_manifest_sha256", std::string(64, 'c').c_str());
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse fails", !llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("hash error", error.find("hash-invalid") != std::string::npos);
    });

    t.test("schema_v2_urt_accepts_a_canonical_manifest_hash", [](testing & t) {
        auto ctx = make_ctx();
        populate_schema_v2(ctx.get());
        gguf_set_val_bool(ctx.get(), "hypura.turboquant.urt.enabled", true);
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.abstract_algebra_id", "triality-v1");
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.operator_word_manifest", "{}");
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.operator_word_sha256", "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a");
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.reference_representation", "vector");
        set_str_array(ctx.get(), "hypura.turboquant.urt.supported_representations", {"vector", "spinor_plus_proxy"});
        gguf_set_val_f32(ctx.get(), "hypura.turboquant.urt.consistency_tolerance", 1e-4f);
        gguf_set_val_u32(ctx.get(), "hypura.turboquant.urt.moment_degree", 2);
        gguf_set_val_str(ctx.get(), "hypura.turboquant.urt.moment_manifest_sha256", std::string(64, 'c').c_str());
        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_true("urt active", metadata.urt.enabled);
    });

    t.test("context_weight_sum_uses_one_e_minus_six_tolerance", [](testing & t) {
        llama_tq_layer_config accepted = make_context_layer(LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        accepted.branches[0].weight += 5e-7f;
        llama_tq_context_config accepted_config = make_context_config(
            &accepted, 1, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_context_state accepted_state(1);
        llama_tq_error error {};
        t.assert_true("sub-tolerance weight error succeeds", accepted_state.configure(accepted_config, true, &error));

        llama_tq_layer_config rejected = make_context_layer(LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        rejected.branches[0].weight += 2e-6f;
        llama_tq_context_config rejected_config = make_context_config(
            &rejected, 1, LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS);
        llama_tq_context_state rejected_state(1);
        t.assert_true("above-tolerance weight error fails", !rejected_state.configure(rejected_config, true, &error));
        t.assert_true("weight failure is diagnostic", std::string(error.message).find("sum to one") != std::string::npos);
    });

    t.test("context_zero_layer_count_normalizes_to_model_count_and_deep_copies", [](testing & t) {
        std::vector<llama_tq_layer_config> layers(2, make_context_layer(LLAMA_TQ_EXEC_SINGLE_VIEW));
        llama_tq_context_config config = make_context_config(layers.data(), 0, LLAMA_TQ_EXEC_SINGLE_VIEW);
        llama_tq_context_state state(2);
        llama_tq_error error {};
        t.assert_true("zero count uses model layer count", state.configure(config, true, &error));
        const llama_tq_owned_context_config * owned = state.config();
        t.assert_true("owned config exists", owned != nullptr);
        if (owned == nullptr) {
            return;
        }
        t.assert_equal("effective layer count is copied", size_t(2), owned->layers.size());
        layers[0].branches[0].weight = 0.5f;
        t.assert_true("owned layer data is independent", std::fabs(owned->layers[0].branches[0].weight - 1.0f) < 1e-7f);
    });

    t.test("context_zero_layer_count_still_rejects_a_null_layer_pointer", [](testing & t) {
        llama_tq_context_config config = make_context_config(nullptr, 0, LLAMA_TQ_EXEC_SINGLE_VIEW);
        llama_tq_context_state state(2);
        llama_tq_error error {};
        t.assert_true("null layers fail", !state.configure(config, true, &error));
        t.assert_true("null failure is diagnostic", std::string(error.message).find("at least one layer") != std::string::npos);
    });

    return t.summary();
}
