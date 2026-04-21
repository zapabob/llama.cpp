#include "testing.h"

#include "../src/llama-turboquant.h"

#include "gguf.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

using gguf_ctx_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

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

        llama_turboquant_gguf_metadata metadata;
        std::string error;
        t.assert_true("parse succeeds", llama_turboquant_load_gguf_metadata(ctx.get(), 2, metadata, &error));
        t.assert_equal("triality mode", std::string("key_only_block_so8_triality_minus"), metadata.layers[0].triality_mode);
        t.assert_equal("triality view", std::string("spinor_minus_proxy"), metadata.layers[0].triality_view);
    });

    return t.summary();
}
