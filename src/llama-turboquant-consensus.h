#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class llama_tq_k_storage_mode : uint32_t {
    single_view     = 0,
    full_triple     = 1,
    residual_parity = 2,
};

struct llama_tq_runtime_capabilities {
    bool full_triple_kv_graph          = false;
    bool residual_parity_kv_graph      = false;
    bool cpu_attention_consensus_graph = false;
};

bool llama_tq_require_runtime_capabilities(llama_tq_k_storage_mode               mode,
                                           bool                                  attention_consensus,
                                           const llama_tq_runtime_capabilities & capabilities,
                                           std::string *                         error);

struct llama_tq_k_memory_breakdown {
    std::array<size_t, 3> view_bytes{ 0, 0, 0 };
    size_t                controller_bytes = 0;

    size_t total_bytes() const;
};

class llama_tq_full_k_storage {
  public:
    bool initialize(llama_tq_k_storage_mode mode, size_t n_tokens, size_t n_channels, std::string * error);

    bool set_token(size_t token, const std::array<std::vector<float>, 3> & views, std::string * error);

    bool copy_tokens(size_t source, size_t destination, size_t count, std::string * error);
    bool shift_left(size_t count, std::string * error);

    llama_tq_k_storage_mode     mode() const;
    size_t                      n_tokens() const;
    size_t                      n_channels() const;
    const std::vector<float> &  view(size_t index) const;
    llama_tq_k_memory_breakdown memory_breakdown() const;

    std::vector<uint8_t> serialize() const;
    static bool deserialize(const std::vector<uint8_t> & bytes, llama_tq_full_k_storage & storage, std::string * error);

  private:
    llama_tq_k_storage_mode           storage_mode  = llama_tq_k_storage_mode::single_view;
    size_t                            token_count   = 0;
    size_t                            channel_count = 0;
    std::array<std::vector<float>, 3> k_views;
};

struct llama_tq_consensus_config {
    std::array<float, 3> weights{ 1.0f, 0.0f, 0.0f };
    std::array<float, 3> bias{ 0.0f, 0.0f, 0.0f };
    std::array<float, 3> scale{ 1.0f, 1.0f, 1.0f };
    float                temperature = 1.0f;
};

struct llama_tq_consensus_result {
    std::array<std::vector<float>, 3> branch_logits;
    std::vector<float>                consensus_logits;
    std::vector<float>                probabilities;
    std::vector<float>                output;
    uint32_t                          softmax_passes      = 0;
    uint32_t                          value_matmul_passes = 0;
};

bool llama_tq_consensus_attention_f32(const std::vector<float> &                query,
                                      const std::array<std::vector<float>, 3> & keys,
                                      const std::vector<float> &                values,
                                      const std::vector<float> &                additive_mask,
                                      size_t                                    n_query,
                                      size_t                                    n_key,
                                      size_t                                    head_dim,
                                      size_t                                    value_dim,
                                      const llama_tq_consensus_config &         config,
                                      llama_tq_consensus_result &               result,
                                      std::string *                             error);

struct llama_tq_rotation_bundle {
    std::array<std::vector<float>, 3> forward;
    std::array<std::vector<float>, 3> inverse;
};

struct llama_tq_residual_parity_profile {
    std::array<uint8_t, 3> bits{ 3, 1, 1 };
    std::array<float, 2>   beta{ 0.5f, 0.5f };
};

struct llama_tq_packed_sector {
    uint8_t              bits  = 0;
    float                scale = 0.0f;
    std::vector<uint8_t> payload;
};

struct llama_tq_residual_parity_storage {
    size_t                                n_vectors = 0;
    size_t                                width     = 0;
    llama_tq_residual_parity_profile      profile;
    std::array<llama_tq_packed_sector, 3> sectors;
};

struct llama_tq_residual_parity_budget {
    std::array<uint64_t, 3> sector_payload_bits{ 0, 0, 0 };
    uint64_t                payload_bytes            = 0;
    uint64_t                controller_bytes         = 0;
    uint64_t                total_bytes              = 0;
    double                  payload_bits_per_channel = 0.0;
    double                  actual_bits_per_channel  = 0.0;
};

bool llama_tq_residual_parity_encode_f32(const std::vector<float> &               values,
                                         size_t                                   n_vectors,
                                         size_t                                   width,
                                         const llama_tq_rotation_bundle &         rotations,
                                         const llama_tq_residual_parity_profile & profile,
                                         llama_tq_residual_parity_storage &       storage,
                                         std::string *                            error);

bool llama_tq_residual_parity_decode_f32(const llama_tq_residual_parity_storage & storage,
                                         const llama_tq_rotation_bundle &         rotations,
                                         std::vector<float> &                     values,
                                         std::string *                            error);

llama_tq_residual_parity_budget llama_tq_residual_parity_measure(const llama_tq_residual_parity_storage & storage);
