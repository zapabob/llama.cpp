#include "llama-turboquant-consensus.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

constexpr std::array<uint8_t, 8> TQ_K_STORAGE_MAGIC{ 'T', 'Q', 'K', 'V', 'T', 'R', 'I', '1' };
constexpr uint32_t               TQ_K_STORAGE_VERSION         = 1;
constexpr uint64_t               TQ_RESIDUAL_CONTROLLER_BYTES = 51;

bool set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

bool checked_product(size_t lhs, size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool all_finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    for (uint32_t shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffu));
    }
}

void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (uint32_t shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffu));
    }
}

void append_f32(std::vector<uint8_t> & bytes, float value) {
    uint32_t encoded = 0;
    static_assert(sizeof(encoded) == sizeof(value), "float must be 32-bit");
    std::memcpy(&encoded, &value, sizeof(encoded));
    append_u32(bytes, encoded);
}

bool read_u32(const std::vector<uint8_t> & bytes, size_t & offset, uint32_t & value) {
    if (bytes.size() - offset < 4) {
        return false;
    }
    value = 0;
    for (uint32_t shift = 0; shift < 32; shift += 8) {
        value |= static_cast<uint32_t>(bytes[offset++]) << shift;
    }
    return true;
}

bool read_u64(const std::vector<uint8_t> & bytes, size_t & offset, uint64_t & value) {
    if (bytes.size() - offset < 8) {
        return false;
    }
    value = 0;
    for (uint32_t shift = 0; shift < 64; shift += 8) {
        value |= static_cast<uint64_t>(bytes[offset++]) << shift;
    }
    return true;
}

bool read_f32(const std::vector<uint8_t> & bytes, size_t & offset, float & value) {
    uint32_t encoded = 0;
    if (!read_u32(bytes, offset, encoded)) {
        return false;
    }
    std::memcpy(&value, &encoded, sizeof(value));
    return std::isfinite(value);
}

bool validate_rotation_bundle(const llama_tq_rotation_bundle & rotations, size_t width, std::string * error) {
    size_t matrix_size = 0;
    if (!checked_product(width, width, matrix_size)) {
        return set_error(error, "Triality rotation matrix dimensions overflow");
    }
    for (size_t view = 0; view < 3; ++view) {
        if (rotations.forward[view].size() != matrix_size || rotations.inverse[view].size() != matrix_size) {
            return set_error(error, "Triality rotation bundle has an invalid matrix shape");
        }
        if (!all_finite(rotations.forward[view]) || !all_finite(rotations.inverse[view])) {
            return set_error(error, "Triality rotation bundle contains a non-finite value");
        }
        for (size_t row = 0; row < width; ++row) {
            for (size_t col = 0; col < width; ++col) {
                double value = 0.0;
                for (size_t k = 0; k < width; ++k) {
                    value += static_cast<double>(rotations.inverse[view][row * width + k]) *
                             rotations.forward[view][k * width + col];
                }
                const double expected = row == col ? 1.0 : 0.0;
                if (std::fabs(value - expected) > 1e-4) {
                    return set_error(error, "Triality rotation inverse validation failed");
                }
            }
        }
    }
    return true;
}

std::vector<float> apply_matrix(const std::vector<float> & matrix,
                                const std::vector<float> & values,
                                size_t                     n_vectors,
                                size_t                     width) {
    std::vector<float> result(values.size(), 0.0f);
    for (size_t vector = 0; vector < n_vectors; ++vector) {
        for (size_t row = 0; row < width; ++row) {
            double sum = 0.0;
            for (size_t col = 0; col < width; ++col) {
                sum += static_cast<double>(matrix[row * width + col]) * values[vector * width + col];
            }
            result[vector * width + row] = static_cast<float>(sum);
        }
    }
    return result;
}

std::vector<uint8_t> pack_codes(const std::vector<uint8_t> & codes, uint8_t bits) {
    const size_t         bit_count = codes.size() * bits;
    std::vector<uint8_t> payload((bit_count + 7) / 8, 0);
    size_t               bit_offset = 0;
    for (uint8_t code : codes) {
        for (uint8_t bit = 0; bit < bits; ++bit) {
            if ((code >> bit) & 1u) {
                payload[bit_offset / 8] |= static_cast<uint8_t>(1u << (bit_offset % 8));
            }
            ++bit_offset;
        }
    }
    return payload;
}

bool unpack_codes(const llama_tq_packed_sector & sector,
                  size_t                         count,
                  std::vector<uint8_t> &         codes,
                  std::string *                  error) {
    if (sector.bits == 0 || sector.bits > 7) {
        return set_error(error, "Residual-parity sector has an unsupported bit width");
    }
    size_t bit_count = 0;
    if (!checked_product(count, sector.bits, bit_count)) {
        return set_error(error, "Residual-parity payload bit count overflows");
    }
    const size_t expected_bytes = (bit_count + 7) / 8;
    if (sector.payload.size() != expected_bytes) {
        return set_error(error, "Residual-parity payload length does not match its shape");
    }
    if (!std::isfinite(sector.scale) || sector.scale < 0.0f) {
        return set_error(error, "Residual-parity sector scale is invalid");
    }
    if (bit_count % 8 != 0 && !sector.payload.empty()) {
        const uint8_t used         = static_cast<uint8_t>(bit_count % 8);
        const uint8_t padding_mask = static_cast<uint8_t>(0xffu << used);
        if ((sector.payload.back() & padding_mask) != 0) {
            return set_error(error, "Residual-parity payload has non-zero padding bits");
        }
    }

    codes.assign(count, 0);
    size_t bit_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        uint8_t code = 0;
        for (uint8_t bit = 0; bit < sector.bits; ++bit) {
            code |= static_cast<uint8_t>(((sector.payload[bit_offset / 8] >> (bit_offset % 8)) & 1u) << bit);
            ++bit_offset;
        }
        codes[i] = code;
    }
    return true;
}

llama_tq_packed_sector quantize_sector(const std::vector<float> & values, uint8_t bits) {
    llama_tq_packed_sector sector;
    sector.bits = bits;
    if (bits == 1) {
        double absolute_sum = 0.0;
        for (float value : values) {
            absolute_sum += std::fabs(value);
        }
        sector.scale = values.empty() ? 0.0f : static_cast<float>(absolute_sum / values.size());
    } else {
        for (float value : values) {
            sector.scale = std::max(sector.scale, std::fabs(value));
        }
    }

    std::vector<uint8_t> codes(values.size(), 0);
    if (sector.scale > 0.0f) {
        if (bits == 1) {
            for (size_t i = 0; i < values.size(); ++i) {
                codes[i] = values[i] >= 0.0f ? 1u : 0u;
            }
        } else {
            const int32_t max_magnitude = (1 << (bits - 1)) - 1;
            const uint8_t mask          = static_cast<uint8_t>((1u << bits) - 1u);
            for (size_t i = 0; i < values.size(); ++i) {
                const float   normalized = values[i] / sector.scale;
                const int32_t quantized =
                    std::max(-max_magnitude,
                             std::min(max_magnitude, static_cast<int32_t>(std::lround(normalized * max_magnitude))));
                codes[i] = static_cast<uint8_t>(quantized) & mask;
            }
        }
    }
    sector.payload = pack_codes(codes, bits);
    return sector;
}

bool dequantize_sector(const llama_tq_packed_sector & sector,
                       size_t                         count,
                       std::vector<float> &           values,
                       std::string *                  error) {
    std::vector<uint8_t> codes;
    if (!unpack_codes(sector, count, codes, error)) {
        return false;
    }
    if (sector.bits > 1) {
        const uint8_t reserved_code = static_cast<uint8_t>(1u << (sector.bits - 1));
        if (std::find(codes.begin(), codes.end(), reserved_code) != codes.end()) {
            return set_error(error, "Residual-parity payload contains a reserved signed code");
        }
    }
    if (sector.scale == 0.0f && std::any_of(codes.begin(), codes.end(), [](uint8_t code) { return code != 0; })) {
        return set_error(error, "Residual-parity zero-scale payload is not canonical");
    }
    values.assign(count, 0.0f);
    if (sector.scale == 0.0f) {
        return true;
    }
    if (sector.bits == 1) {
        for (size_t i = 0; i < count; ++i) {
            values[i] = codes[i] ? sector.scale : -sector.scale;
        }
        return true;
    }

    const int32_t max_magnitude = (1 << (sector.bits - 1)) - 1;
    const uint8_t sign_bit      = static_cast<uint8_t>(1u << (sector.bits - 1));
    const uint8_t value_mask    = static_cast<uint8_t>((1u << sector.bits) - 1u);
    for (size_t i = 0; i < count; ++i) {
        int32_t quantized = codes[i];
        if (codes[i] & sign_bit) {
            quantized = static_cast<int32_t>(static_cast<int8_t>(codes[i] | static_cast<uint8_t>(~value_mask)));
        }
        values[i] = sector.scale * static_cast<float>(quantized) / static_cast<float>(max_magnitude);
    }
    return true;
}

}  // namespace

bool llama_tq_require_runtime_capabilities(llama_tq_k_storage_mode               mode,
                                           bool                                  attention_consensus,
                                           const llama_tq_runtime_capabilities & capabilities,
                                           std::string *                         error) {
    if (mode != llama_tq_k_storage_mode::single_view && mode != llama_tq_k_storage_mode::full_triple &&
        mode != llama_tq_k_storage_mode::residual_parity) {
        return set_error(error, "Triality K storage mode is unsupported");
    }
    if (attention_consensus && mode == llama_tq_k_storage_mode::single_view) {
        return set_error(error, "Triality attention consensus requires multi-view K storage");
    }
    if (mode == llama_tq_k_storage_mode::full_triple && !capabilities.full_triple_kv_graph) {
        return set_error(error, "full-triple Triality K storage is not connected to the production graph");
    }
    if (mode == llama_tq_k_storage_mode::residual_parity && !capabilities.residual_parity_kv_graph) {
        return set_error(error, "residual-parity Triality K storage is not connected to the production graph");
    }
    if (attention_consensus && !capabilities.cpu_attention_consensus_graph) {
        return set_error(error, "Triality attention consensus is not connected to the production graph");
    }
    if (error) {
        error->clear();
    }
    return true;
}

size_t llama_tq_k_memory_breakdown::total_bytes() const {
    return view_bytes[0] + view_bytes[1] + view_bytes[2] + controller_bytes;
}

bool llama_tq_full_k_storage::initialize(llama_tq_k_storage_mode mode,
                                         size_t                  n_tokens,
                                         size_t                  n_channels,
                                         std::string *           error) {
    if (mode != llama_tq_k_storage_mode::single_view && mode != llama_tq_k_storage_mode::full_triple &&
        mode != llama_tq_k_storage_mode::residual_parity) {
        return set_error(error, "Triality K storage mode is unsupported");
    }
    if (mode == llama_tq_k_storage_mode::residual_parity) {
        return set_error(error, "residual-parity storage must use the packed residual-parity codec");
    }
    size_t value_count = 0;
    if (n_tokens == 0 || n_channels == 0 || !checked_product(n_tokens, n_channels, value_count)) {
        return set_error(error, "Triality K storage shape is empty or overflows");
    }
    storage_mode  = mode;
    token_count   = n_tokens;
    channel_count = n_channels;
    k_views[0].assign(value_count, 0.0f);
    k_views[1].assign(mode == llama_tq_k_storage_mode::full_triple ? value_count : 0, 0.0f);
    k_views[2].assign(mode == llama_tq_k_storage_mode::full_triple ? value_count : 0, 0.0f);
    if (error) {
        error->clear();
    }
    return true;
}

bool llama_tq_full_k_storage::set_token(size_t                                    token,
                                        const std::array<std::vector<float>, 3> & views,
                                        std::string *                             error) {
    if (token >= token_count) {
        return set_error(error, "Triality K token index is out of range");
    }
    const size_t active_views = storage_mode == llama_tq_k_storage_mode::full_triple ? 3 : 1;
    for (size_t view = 0; view < active_views; ++view) {
        if (views[view].size() != channel_count || !all_finite(views[view])) {
            return set_error(error, "Triality K token view has an invalid shape or non-finite value");
        }
    }
    for (size_t view = 0; view < active_views; ++view) {
        std::copy(views[view].begin(), views[view].end(), k_views[view].begin() + token * channel_count);
    }
    if (error) {
        error->clear();
    }
    return true;
}

bool llama_tq_full_k_storage::copy_tokens(size_t source, size_t destination, size_t count, std::string * error) {
    if (source > token_count || destination > token_count || count > token_count - source ||
        count > token_count - destination) {
        return set_error(error, "Triality K copy range is out of bounds");
    }
    const size_t active_views = storage_mode == llama_tq_k_storage_mode::full_triple ? 3 : 1;
    const size_t float_count  = count * channel_count;
    for (size_t view = 0; view < active_views; ++view) {
        std::memmove(k_views[view].data() + destination * channel_count, k_views[view].data() + source * channel_count,
                     float_count * sizeof(float));
    }
    if (error) {
        error->clear();
    }
    return true;
}

bool llama_tq_full_k_storage::shift_left(size_t count, std::string * error) {
    if (count > token_count) {
        return set_error(error, "Triality K shift count is out of bounds");
    }
    const size_t active_views = storage_mode == llama_tq_k_storage_mode::full_triple ? 3 : 1;
    const size_t remaining    = token_count - count;
    for (size_t view = 0; view < active_views; ++view) {
        std::memmove(k_views[view].data(), k_views[view].data() + count * channel_count,
                     remaining * channel_count * sizeof(float));
        std::fill(k_views[view].begin() + remaining * channel_count, k_views[view].end(), 0.0f);
    }
    if (error) {
        error->clear();
    }
    return true;
}

llama_tq_k_storage_mode llama_tq_full_k_storage::mode() const {
    return storage_mode;
}

size_t llama_tq_full_k_storage::n_tokens() const {
    return token_count;
}

size_t llama_tq_full_k_storage::n_channels() const {
    return channel_count;
}

const std::vector<float> & llama_tq_full_k_storage::view(size_t index) const {
    static const std::vector<float> empty;
    return index < k_views.size() ? k_views[index] : empty;
}

llama_tq_k_memory_breakdown llama_tq_full_k_storage::memory_breakdown() const {
    llama_tq_k_memory_breakdown result;
    for (size_t view = 0; view < 3; ++view) {
        result.view_bytes[view] = k_views[view].size() * sizeof(float);
    }
    result.controller_bytes = TQ_K_STORAGE_MAGIC.size() + sizeof(uint32_t) * 2 + sizeof(uint64_t) * 2;
    return result;
}

std::vector<uint8_t> llama_tq_full_k_storage::serialize() const {
    std::vector<uint8_t> bytes;
    const auto           breakdown = memory_breakdown();
    bytes.reserve(breakdown.total_bytes());
    bytes.insert(bytes.end(), TQ_K_STORAGE_MAGIC.begin(), TQ_K_STORAGE_MAGIC.end());
    append_u32(bytes, TQ_K_STORAGE_VERSION);
    append_u32(bytes, static_cast<uint32_t>(storage_mode));
    append_u64(bytes, token_count);
    append_u64(bytes, channel_count);
    const size_t active_views = storage_mode == llama_tq_k_storage_mode::full_triple ? 3 : 1;
    for (size_t view = 0; view < active_views; ++view) {
        for (float value : k_views[view]) {
            append_f32(bytes, value);
        }
    }
    return bytes;
}

bool llama_tq_full_k_storage::deserialize(const std::vector<uint8_t> & bytes,
                                          llama_tq_full_k_storage &    storage,
                                          std::string *                error) {
    if (bytes.size() < TQ_K_STORAGE_MAGIC.size() ||
        !std::equal(TQ_K_STORAGE_MAGIC.begin(), TQ_K_STORAGE_MAGIC.end(), bytes.begin())) {
        return set_error(error, "Triality K storage magic does not match");
    }
    size_t   offset       = TQ_K_STORAGE_MAGIC.size();
    uint32_t version      = 0;
    uint32_t encoded_mode = 0;
    uint64_t n_tokens     = 0;
    uint64_t n_channels   = 0;
    if (!read_u32(bytes, offset, version) || !read_u32(bytes, offset, encoded_mode) ||
        !read_u64(bytes, offset, n_tokens) || !read_u64(bytes, offset, n_channels)) {
        return set_error(error, "Triality K storage header is truncated");
    }
    if (version != TQ_K_STORAGE_VERSION) {
        return set_error(error, "Triality K storage version is unsupported");
    }
    if (encoded_mode > static_cast<uint32_t>(llama_tq_k_storage_mode::full_triple)) {
        return set_error(error, "Triality K storage mode is unsupported");
    }
    if (n_tokens > std::numeric_limits<size_t>::max() || n_channels > std::numeric_limits<size_t>::max()) {
        return set_error(error, "Triality K storage shape exceeds this platform");
    }

    const auto   mode           = static_cast<llama_tq_k_storage_mode>(encoded_mode);
    const size_t active_views   = mode == llama_tq_k_storage_mode::full_triple ? 3 : 1;
    size_t       value_count    = 0;
    size_t       payload_values = 0;
    size_t       payload_bytes  = 0;
    if (!checked_product(static_cast<size_t>(n_tokens), static_cast<size_t>(n_channels), value_count) ||
        !checked_product(value_count, active_views, payload_values) ||
        !checked_product(payload_values, sizeof(float), payload_bytes) || payload_bytes > bytes.size() - offset ||
        offset + payload_bytes != bytes.size()) {
        return set_error(error, "Triality K storage payload length does not match its shape");
    }

    llama_tq_full_k_storage decoded;
    if (!decoded.initialize(mode, static_cast<size_t>(n_tokens), static_cast<size_t>(n_channels), error)) {
        return false;
    }
    for (size_t view = 0; view < active_views; ++view) {
        for (float & value : decoded.k_views[view]) {
            if (!read_f32(bytes, offset, value)) {
                return set_error(error, "Triality K storage payload is truncated or non-finite");
            }
        }
    }
    if (offset != bytes.size()) {
        return set_error(error, "Triality K storage has trailing bytes");
    }
    storage = std::move(decoded);
    if (error) {
        error->clear();
    }
    return true;
}

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
                                      std::string *                             error) {
    size_t query_count = 0;
    size_t key_count   = 0;
    size_t value_count = 0;
    size_t logit_count = 0;
    if (n_query == 0 || n_key == 0 || head_dim == 0 || value_dim == 0 ||
        !checked_product(n_query, head_dim, query_count) || !checked_product(n_key, head_dim, key_count) ||
        !checked_product(n_key, value_dim, value_count) || !checked_product(n_query, n_key, logit_count)) {
        return set_error(error, "Triality attention shape is empty or overflows");
    }
    if (query.size() != query_count || values.size() != value_count ||
        (!additive_mask.empty() && additive_mask.size() != logit_count)) {
        return set_error(error, "Triality attention input shape does not match its dimensions");
    }
    if (!all_finite(query) || !all_finite(values)) {
        return set_error(error, "Triality attention inputs contain a non-finite value");
    }
    for (const auto & key : keys) {
        if (key.size() != key_count || !all_finite(key)) {
            return set_error(error, "Triality attention key view has an invalid shape or non-finite value");
        }
    }

    double weight_sum = 0.0;
    for (size_t view = 0; view < 3; ++view) {
        if (!std::isfinite(config.weights[view]) || config.weights[view] < 0.0f || !std::isfinite(config.bias[view]) ||
            !std::isfinite(config.scale[view]) || config.scale[view] <= 0.0f) {
            return set_error(error, "Triality consensus calibration is invalid");
        }
        weight_sum += config.weights[view];
    }
    if (std::fabs(weight_sum - 1.0) > 1e-6 || !std::isfinite(config.temperature) || config.temperature <= 0.0f) {
        return set_error(error, "Triality consensus weights or temperature are invalid");
    }
    for (float mask : additive_mask) {
        if (!std::isfinite(mask) && mask != -std::numeric_limits<float>::infinity()) {
            return set_error(error, "Triality attention mask contains an invalid value");
        }
    }

    llama_tq_consensus_result computed;
    for (auto & branch : computed.branch_logits) {
        branch.assign(logit_count, 0.0f);
    }
    computed.consensus_logits.assign(logit_count, 0.0f);
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (size_t q = 0; q < n_query; ++q) {
        for (size_t k = 0; k < n_key; ++k) {
            const size_t logit_index = q * n_key + k;
            double       consensus   = 0.0;
            for (size_t view = 0; view < 3; ++view) {
                double dot = 0.0;
                for (size_t channel = 0; channel < head_dim; ++channel) {
                    dot += static_cast<double>(query[q * head_dim + channel]) * keys[view][k * head_dim + channel];
                }
                if (!std::isfinite(dot)) {
                    return set_error(error, "Triality attention dot product overflowed");
                }
                const float calibrated = config.scale[view] * static_cast<float>(dot) * kq_scale + config.bias[view];
                if (!std::isfinite(calibrated)) {
                    return set_error(error, "Triality attention calibration overflowed");
                }
                computed.branch_logits[view][logit_index] = calibrated;
                consensus += static_cast<double>(config.weights[view]) * calibrated;
            }
            float logit = static_cast<float>(consensus / config.temperature);
            if (!std::isfinite(logit)) {
                return set_error(error, "Triality attention consensus overflowed");
            }
            if (!additive_mask.empty()) {
                logit += additive_mask[logit_index];
            }
            computed.consensus_logits[logit_index] = logit;
        }
    }

    computed.probabilities.assign(logit_count, 0.0f);
    for (size_t q = 0; q < n_query; ++q) {
        const auto  row_begin = computed.consensus_logits.begin() + q * n_key;
        const float maximum   = *std::max_element(row_begin, row_begin + n_key);
        if (!std::isfinite(maximum)) {
            return set_error(error, "Triality attention mask hides every key in a query row");
        }
        double denominator = 0.0;
        for (size_t k = 0; k < n_key; ++k) {
            const float probability               = std::exp(computed.consensus_logits[q * n_key + k] - maximum);
            computed.probabilities[q * n_key + k] = probability;
            denominator += probability;
        }
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            return set_error(error, "Triality attention softmax normalization failed");
        }
        for (size_t k = 0; k < n_key; ++k) {
            computed.probabilities[q * n_key + k] /= static_cast<float>(denominator);
        }
    }
    computed.softmax_passes = 1;

    computed.output.assign(n_query * value_dim, 0.0f);
    for (size_t q = 0; q < n_query; ++q) {
        for (size_t channel = 0; channel < value_dim; ++channel) {
            double sum = 0.0;
            for (size_t k = 0; k < n_key; ++k) {
                sum += static_cast<double>(computed.probabilities[q * n_key + k]) * values[k * value_dim + channel];
            }
            computed.output[q * value_dim + channel] = static_cast<float>(sum);
        }
    }
    computed.value_matmul_passes = 1;
    result                       = std::move(computed);
    if (error) {
        error->clear();
    }
    return true;
}

bool llama_tq_residual_parity_encode_f32(const std::vector<float> &               values,
                                         size_t                                   n_vectors,
                                         size_t                                   width,
                                         const llama_tq_rotation_bundle &         rotations,
                                         const llama_tq_residual_parity_profile & profile,
                                         llama_tq_residual_parity_storage &       storage,
                                         std::string *                            error) {
    size_t value_count = 0;
    if (n_vectors == 0 || width == 0 || !checked_product(n_vectors, width, value_count) ||
        values.size() != value_count) {
        return set_error(error, "Residual-parity input shape is empty, mismatched, or overflows");
    }
    if (!all_finite(values)) {
        return set_error(error, "Residual-parity input contains a non-finite value");
    }
    if (!validate_rotation_bundle(rotations, width, error)) {
        return false;
    }
    if (profile.bits != std::array<uint8_t, 3>{ 3, 1, 1 }) {
        return set_error(error, "Residual-parity reference supports only the 3+1+1 profile");
    }
    if (!std::isfinite(profile.beta[0]) || !std::isfinite(profile.beta[1]) || profile.beta[0] < 0.0f ||
        profile.beta[1] < 0.0f) {
        return set_error(error, "Residual-parity beta values are invalid");
    }

    llama_tq_residual_parity_storage encoded;
    encoded.n_vectors = n_vectors;
    encoded.width     = width;
    encoded.profile   = profile;

    const std::vector<float> main_rotated = apply_matrix(rotations.forward[0], values, n_vectors, width);
    if (!all_finite(main_rotated)) {
        return set_error(error, "Residual-parity main rotation overflowed");
    }
    encoded.sectors[0] = quantize_sector(main_rotated, profile.bits[0]);

    std::vector<float> main_decoded;
    if (!dequantize_sector(encoded.sectors[0], value_count, main_decoded, error)) {
        return false;
    }
    const std::vector<float> main_reconstructed = apply_matrix(rotations.inverse[0], main_decoded, n_vectors, width);
    if (!all_finite(main_reconstructed)) {
        return set_error(error, "Residual-parity main reconstruction overflowed");
    }
    std::vector<float> residual(value_count, 0.0f);
    for (size_t i = 0; i < value_count; ++i) {
        residual[i] = values[i] - main_reconstructed[i];
    }

    for (size_t view = 1; view < 3; ++view) {
        const std::vector<float> rotated = apply_matrix(rotations.forward[view], residual, n_vectors, width);
        if (!all_finite(rotated)) {
            return set_error(error, "Residual-parity residual rotation overflowed");
        }
        encoded.sectors[view] = quantize_sector(rotated, profile.bits[view]);
    }
    storage = std::move(encoded);
    if (error) {
        error->clear();
    }
    return true;
}

bool llama_tq_residual_parity_decode_f32(const llama_tq_residual_parity_storage & storage,
                                         const llama_tq_rotation_bundle &         rotations,
                                         std::vector<float> &                     values,
                                         std::string *                            error) {
    size_t value_count = 0;
    if (storage.n_vectors == 0 || storage.width == 0 ||
        !checked_product(storage.n_vectors, storage.width, value_count)) {
        return set_error(error, "Residual-parity storage shape is empty or overflows");
    }
    if (storage.profile.bits != std::array<uint8_t, 3>{ 3, 1, 1 }) {
        return set_error(error, "Residual-parity storage uses an unsupported profile");
    }
    if (!validate_rotation_bundle(rotations, storage.width, error)) {
        return false;
    }
    if (!std::isfinite(storage.profile.beta[0]) || !std::isfinite(storage.profile.beta[1]) ||
        storage.profile.beta[0] < 0.0f || storage.profile.beta[1] < 0.0f) {
        return set_error(error, "Residual-parity beta values are invalid");
    }

    std::array<std::vector<float>, 3> decoded;
    std::array<std::vector<float>, 3> reconstructed;
    for (size_t view = 0; view < 3; ++view) {
        if (storage.sectors[view].bits != storage.profile.bits[view]) {
            return set_error(error, "Residual-parity sector bit width does not match its profile");
        }
        if (!dequantize_sector(storage.sectors[view], value_count, decoded[view], error)) {
            return false;
        }
        reconstructed[view] = apply_matrix(rotations.inverse[view], decoded[view], storage.n_vectors, storage.width);
        if (!all_finite(reconstructed[view])) {
            return set_error(error, "Residual-parity reconstruction overflowed");
        }
    }

    std::vector<float> combined(value_count, 0.0f);
    const double       float_max = std::numeric_limits<float>::max();
    for (size_t i = 0; i < value_count; ++i) {
        const double value = static_cast<double>(reconstructed[0][i]) +
                             static_cast<double>(storage.profile.beta[0]) * reconstructed[1][i] +
                             static_cast<double>(storage.profile.beta[1]) * reconstructed[2][i];
        if (!std::isfinite(value) || std::fabs(value) > float_max) {
            return set_error(error, "Residual-parity final reconstruction overflowed");
        }
        combined[i] = static_cast<float>(value);
    }
    values = std::move(combined);
    if (error) {
        error->clear();
    }
    return true;
}

llama_tq_residual_parity_budget llama_tq_residual_parity_measure(const llama_tq_residual_parity_storage & storage) {
    llama_tq_residual_parity_budget budget;
    const uint64_t                  channels = static_cast<uint64_t>(storage.n_vectors) * storage.width;
    for (size_t view = 0; view < 3; ++view) {
        budget.sector_payload_bits[view] = channels * storage.sectors[view].bits;
        budget.payload_bytes += storage.sectors[view].payload.size();
    }
    budget.controller_bytes = TQ_RESIDUAL_CONTROLLER_BYTES;
    budget.total_bytes      = budget.payload_bytes + budget.controller_bytes;
    if (channels != 0) {
        budget.payload_bits_per_channel =
            static_cast<double>(budget.sector_payload_bits[0] + budget.sector_payload_bits[1] +
                                budget.sector_payload_bits[2]) /
            channels;
        budget.actual_bits_per_channel = static_cast<double>(budget.total_bytes * 8) / channels;
    }
    return budget;
}
