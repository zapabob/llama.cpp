#include "testing.h"

#include "../src/llama-turboquant.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

static std::filesystem::path make_temp_path(const std::string & stem) {
    const auto base = std::filesystem::temp_directory_path();
    std::ostringstream oss;
    oss << stem << "-" << std::rand() << ".tq";
    return base / oss.str();
}

static llama_turboquant_artifact_metadata make_metadata() {
    llama_turboquant_artifact_metadata meta;
    meta.schema_version = 1;
    meta.total_bits = 3.5f;
    meta.runtime_bits_per_channel = 3.25f;
    meta.stage1_effective_bits = 2.25f;
    meta.qjl_bits = 1;
    meta.qjl_dim = 8;
    meta.rotation_policy = "block_so8_learned";
    meta.rotation_seed = 17;
    meta.qjl_seed = 71;
    meta.triality_mode = "research-kv-split";
    meta.triality_view = "vector";
    meta.stage1_allocation_scheme = "magnitude-topk";
    meta.stage1_bitwidth_payload_dtype = "uint8";
    meta.norm_dtype = "float32";
    meta.sign_pack_format = "int8_unpacked_binary";
    return meta;
}

static llama_turboquant_artifact make_artifact() {
    llama_turboquant_artifact artifact;
    artifact.head_dim = 8;
    artifact.so8_learned = true;
    artifact.triality_enabled = true;
    artifact.metadata = make_metadata();
    artifact.so8_rotation = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    artifact.triality_codebook = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    return artifact;
}

static void write_artifact_file(
    const std::filesystem::path & path,
    const std::vector<std::string> & metadata_lines,
    bool legacy_v1 = false,
    uint32_t head_dim = 8,
    size_t rotation_size = 64,
    size_t codebook_size = 24) {
    std::ofstream out(path, std::ios::binary);
    out << (legacy_v1 ? "TQCUDA1\n" : "TQCUDA2\n");
    out << head_dim << "\n";
    out << 1 << "\n";
    out << 1 << "\n";
    if (!legacy_v1) {
        out << metadata_lines.size() << "\n";
        for (const auto & line : metadata_lines) {
            out << line << "\n";
        }
    }
    out << rotation_size << "\n";
    for (size_t i = 0; i < rotation_size; ++i) {
        out << std::setprecision(9) << (i % 9 == 0 ? 1.0f : 0.0f) << "\n";
    }
    out << codebook_size << "\n";
    for (size_t i = 0; i < codebook_size; ++i) {
        out << std::setprecision(9) << static_cast<float>(i) << "\n";
    }
}

} // namespace

int main() {
    testing t;

    t.test("llama_turboquant_save_and_load_roundtrip_preserves_required_metadata", [](testing & t) {
        const auto path = make_temp_path("turboquant-roundtrip");
        const auto artifact = make_artifact();
        std::string error;
        t.assert_true("save succeeds", llama_turboquant_save_artifact(path.string(), artifact, &error));

        llama_turboquant_artifact loaded;
        error.clear();
        t.assert_true("load succeeds", llama_turboquant_load_artifact(path.string(), loaded, &error));
        t.assert_equal("head_dim", artifact.head_dim, loaded.head_dim);
        t.assert_equal("schema_version", artifact.metadata.schema_version, loaded.metadata.schema_version);
        t.assert_true("runtime bits", std::fabs(loaded.metadata.runtime_bits_per_channel - 3.25f) < 1e-6f);
        t.assert_equal(
            "triality_mode",
            std::string("key_only_block_so8_triality_vector"),
            loaded.metadata.triality_mode);
        t.assert_equal("triality_view", artifact.metadata.triality_view, loaded.metadata.triality_view);
        t.assert_equal("rotation_policy", artifact.metadata.rotation_policy, loaded.metadata.rotation_policy);
        std::filesystem::remove(path);
    });

    t.test("llama_turboquant_load_rejects_legacy_artifact_without_metadata", [](testing & t) {
        const auto path = make_temp_path("turboquant-legacy");
        write_artifact_file(path, {}, true);
        llama_turboquant_artifact loaded;
        std::string error;
        t.assert_true("load fails", !llama_turboquant_load_artifact(path.string(), loaded, &error));
        t.assert_true("mentions strict metadata", error.find("metadata") != std::string::npos);
        std::filesystem::remove(path);
    });

    t.test("llama_turboquant_load_rejects_inconsistent_runtime_bits", [](testing & t) {
        const auto path = make_temp_path("turboquant-inconsistent");
        write_artifact_file(
            path,
            {
                "tq_schema_version=1",
                "tq_total_bits=3.5",
                "tq_runtime_bits_per_channel=3.0",
                "tq_stage1_effective_bits=2.25",
                "tq_qjl_bits=1",
                "tq_qjl_dim=8",
                "tq_rotation_policy=block_so8_learned",
                "tq_rotation_seed=17",
                "tq_qjl_seed=71",
                "tq_triality_mode=research-kv-split",
                "tq_triality_view=vector",
                "tq_stage1_allocation_scheme=magnitude-topk",
                "tq_stage1_bitwidth_payload_dtype=uint8",
                "tq_norm_dtype=float32",
                "tq_sign_pack_format=int8_unpacked_binary",
            });
        llama_turboquant_artifact loaded;
        std::string error;
        t.assert_true("load fails", !llama_turboquant_load_artifact(path.string(), loaded, &error));
        t.assert_true("mentions runtime bits", error.find("runtime_bits_per_channel") != std::string::npos);
        std::filesystem::remove(path);
    });

    return t.summary();
}
