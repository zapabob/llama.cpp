#include "llama-turboquant-telemetry.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-turboquant.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

struct tq_tensor_snapshot {
    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> ne {};
    std::array<size_t, GGML_MAX_DIMS> nb {};
    std::vector<uint8_t> data;
};

bool tq_is_quantized_key_type(ggml_type type) {
    return type == GGML_TYPE_TURBO2_0 ||
           type == GGML_TYPE_TURBO3_0 ||
           type == GGML_TYPE_TURBO4_0;
}

bool tq_snapshot_tensor(const ggml_tensor * tensor, tq_tensor_snapshot & snapshot) {
    if (!tensor || !tensor->buffer || !tensor->data) {
        return false;
    }
    const size_t bytes = ggml_nbytes(tensor);
    if (bytes == 0) {
        return false;
    }
    snapshot.type = tensor->type;
    std::copy_n(tensor->ne, GGML_MAX_DIMS, snapshot.ne.begin());
    std::copy_n(tensor->nb, GGML_MAX_DIMS, snapshot.nb.begin());
    snapshot.data.resize(bytes);
    ggml_backend_tensor_get(tensor, snapshot.data.data(), 0, bytes);
    return true;
}

bool tq_load_scalar(const tq_tensor_snapshot & tensor, size_t offset, float & value) {
    if (tensor.type == GGML_TYPE_F32) {
        if (offset + sizeof(float) > tensor.data.size()) {
            return false;
        }
        std::memcpy(&value, tensor.data.data() + offset, sizeof(value));
        return std::isfinite(value);
    }
    if (tensor.type == GGML_TYPE_F16) {
        if (offset + sizeof(ggml_fp16_t) > tensor.data.size()) {
            return false;
        }
        ggml_fp16_t half;
        std::memcpy(&half, tensor.data.data() + offset, sizeof(half));
        value = ggml_fp16_to_fp32(half);
        return std::isfinite(value);
    }
    return false;
}

bool tq_load_q(
        const tq_tensor_snapshot & q,
        int64_t dim,
        int64_t query,
        int64_t head,
        int64_t stream,
        float & value) {
    const size_t offset =
        static_cast<size_t>(dim) * q.nb[0] +
        static_cast<size_t>(query) * q.nb[1] +
        static_cast<size_t>(head) * q.nb[2] +
        static_cast<size_t>(stream) * q.nb[3];
    return tq_load_scalar(q, offset, value);
}

bool tq_rotate_q(
        const tq_tensor_snapshot & q,
        const tq_tensor_snapshot * rotation,
        int64_t dim,
        int64_t query,
        int64_t head,
        int64_t stream,
        float & value) {
    if (!rotation) {
        return tq_load_q(q, dim, query, head, stream, value);
    }
    if (rotation->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t block = dim / 8;
    const int64_t row = dim % 8;
    value = 0.0f;
    for (int64_t col = 0; col < 8; ++col) {
        const int64_t q_dim = block * 8 + col;
        float q_value;
        if (!tq_load_q(q, q_dim, query, head, stream, q_value)) {
            return false;
        }
        const size_t coefficient_offset = rotation->ne[0] == q.ne[0]
            ? static_cast<size_t>(q_dim) * rotation->nb[0] + static_cast<size_t>(dim) * rotation->nb[1]
            : static_cast<size_t>(col) * rotation->nb[0] + static_cast<size_t>(row) * rotation->nb[1] +
                static_cast<size_t>(block) * rotation->nb[2];
        float coefficient;
        if (!tq_load_scalar(*rotation, coefficient_offset, coefficient)) {
            return false;
        }
        value += coefficient * q_value;
    }
    return std::isfinite(value);
}

void tq_wht(std::array<float, 128> & values) {
    static constexpr std::array<float, 128> signs1 = {-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
    static constexpr std::array<float, 128> signs2 = {1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] *= signs1[i];
    }
    for (size_t width = 1; width < values.size(); width *= 2) {
        for (size_t base = 0; base < values.size(); base += 2 * width) {
            for (size_t offset = 0; offset < width; ++offset) {
                const float lhs = values[base + offset];
                const float rhs = values[base + offset + width];
                values[base + offset] = lhs + rhs;
                values[base + offset + width] = lhs - rhs;
            }
        }
    }
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] *= signs2[i] * 0.08838834764831845f;
    }
}

bool tq_load_key_row(
        const tq_tensor_snapshot & key,
        size_t row_offset,
        std::vector<float> & values) {
    values.resize(static_cast<size_t>(key.ne[0]));
    if (key.type == GGML_TYPE_F32 || key.type == GGML_TYPE_F16) {
        for (int64_t dim = 0; dim < key.ne[0]; ++dim) {
            if (!tq_load_scalar(key, row_offset + static_cast<size_t>(dim) * key.nb[0], values[dim])) {
                return false;
            }
        }
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(key.type);
    const size_t row_size = ggml_row_size(key.type, key.ne[0]);
    if (!traits || !traits->to_float || row_offset + row_size > key.data.size()) {
        return false;
    }
    traits->to_float(key.data.data() + row_offset, values.data(), key.ne[0]);
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool tq_parse_layer(const ggml_tensor * node, size_t layer_count, size_t & layer_index) {
    const char * name = ggml_get_name(node);
    if (!name) {
        return false;
    }
    const char * separator = std::strrchr(name, '-');
    if (!separator || separator[1] == '\0') {
        return false;
    }
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(separator + 1, &end, 10);
    if (!end || *end != '\0' || parsed >= layer_count) {
        return false;
    }
    layer_index = static_cast<size_t>(parsed);
    return true;
}

std::array<double, 3> tq_gram_eigenvalues(const std::array<std::vector<float>, 3> & logits) {
    double matrix[3][3] = {};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = row; col < 3; ++col) {
            double value = 0.0;
            for (size_t i = 0; i < logits[row].size(); ++i) {
                value += static_cast<double>(logits[row][i]) * logits[col][i];
            }
            matrix[row][col] = value;
            matrix[col][row] = value;
        }
    }
    for (int iteration = 0; iteration < 24; ++iteration) {
        size_t p = 0;
        size_t q = 1;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = row + 1; col < 3; ++col) {
                if (std::fabs(matrix[row][col]) > std::fabs(matrix[p][q])) {
                    p = row;
                    q = col;
                }
            }
        }
        if (std::fabs(matrix[p][q]) <= 1.0e-12) {
            break;
        }
        const double angle = 0.5 * std::atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p]);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const double app = matrix[p][p];
        const double aqq = matrix[q][q];
        const double apq = matrix[p][q];
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for (size_t other = 0; other < 3; ++other) {
            if (other == p || other == q) {
                continue;
            }
            const double mop = matrix[other][p];
            const double moq = matrix[other][q];
            matrix[other][p] = cosine * mop - sine * moq;
            matrix[p][other] = matrix[other][p];
            matrix[other][q] = sine * mop + cosine * moq;
            matrix[q][other] = matrix[other][q];
        }
    }
    std::array<double, 3> eigenvalues = {
        std::max(0.0, matrix[0][0]),
        std::max(0.0, matrix[1][1]),
        std::max(0.0, matrix[2][2]),
    };
    std::sort(eigenvalues.begin(), eigenvalues.end(), std::greater<double>());
    return eigenvalues;
}

bool tq_compute_metrics(
        const ggml_tensor * node,
        const llama_tq_owned_context_config & config,
        llama_tq_consensus_metrics & metrics) {
    if (!node || node->op != GGML_OP_TQ_TRIALITY_KQ_CONSENSUS ||
        node->type != GGML_TYPE_F32 || !node->src[0] || !node->src[1] || !node->src[2] || !node->src[3]) {
        return false;
    }
    size_t layer_index;
    if (!tq_parse_layer(node, config.layers.size(), layer_index)) {
        return false;
    }

    tq_tensor_snapshot output;
    tq_tensor_snapshot query;
    std::array<tq_tensor_snapshot, 3> keys;
    std::array<tq_tensor_snapshot, 3> rotations;
    std::array<bool, 3> has_rotation = {};
    if (!tq_snapshot_tensor(node, output) || !tq_snapshot_tensor(node->src[0], query)) {
        return false;
    }
    for (size_t branch = 0; branch < 3; ++branch) {
        if (!tq_snapshot_tensor(node->src[branch + 1], keys[branch])) {
            return false;
        }
        has_rotation[branch] = node->src[branch + 4] != nullptr;
        if (has_rotation[branch] && !tq_snapshot_tensor(node->src[branch + 4], rotations[branch])) {
            return false;
        }
        const bool quantized = tq_is_quantized_key_type(keys[branch].type);
        const int64_t expected_key_dim = quantized
            ? ((query.ne[0] + 127) / 128) * 128
            : query.ne[0];
        if (keys[branch].ne[0] != expected_key_dim || query.ne[2] % keys[branch].ne[2] != 0 ||
            query.ne[3] % keys[branch].ne[3] != 0) {
            return false;
        }
    }

    const float * op_params = reinterpret_cast<const float *>(node->op_params);
    const float * weights = op_params;
    const float * bias = op_params + 3;
    const float * scale = op_params + 6;
    const size_t element_count = static_cast<size_t>(ggml_nelements(node));
    std::array<std::vector<float>, 3> branch_logits;
    for (auto & logits : branch_logits) {
        logits.resize(element_count);
    }
    std::vector<float> key_values;
    std::array<float, 128> q_group {};
    std::array<float, 128> key_group {};
    const int64_t group_count = (query.ne[0] + 127) / 128;

    for (size_t output_index = 0; output_index < element_count; ++output_index) {
        int64_t remaining = static_cast<int64_t>(output_index);
        const int64_t kv = remaining % output.ne[0];
        remaining /= output.ne[0];
        const int64_t query_index = remaining % output.ne[1];
        remaining /= output.ne[1];
        const int64_t head = remaining % output.ne[2];
        const int64_t stream = remaining / output.ne[2];
        float reconstructed = 0.0f;

        for (size_t branch = 0; branch < 3; ++branch) {
            const auto & key = keys[branch];
            const int64_t key_head = head / (query.ne[2] / key.ne[2]);
            const int64_t key_stream = stream / (query.ne[3] / key.ne[3]);
            const size_t key_row_offset =
                static_cast<size_t>(kv) * key.nb[1] +
                static_cast<size_t>(key_head) * key.nb[2] +
                static_cast<size_t>(key_stream) * key.nb[3];
            if (!tq_load_key_row(key, key_row_offset, key_values)) {
                return false;
            }
            double dot = 0.0;
            for (int64_t group = 0; group < group_count; ++group) {
                q_group.fill(0.0f);
                key_group.fill(0.0f);
                for (int64_t offset = 0; offset < 128; ++offset) {
                    const int64_t dim = group * 128 + offset;
                    if (dim >= query.ne[0]) {
                        continue;
                    }
                    if (!tq_rotate_q(
                            query,
                            has_rotation[branch] ? &rotations[branch] : nullptr,
                            dim,
                            query_index,
                            head,
                            stream,
                            q_group[offset])) {
                        return false;
                    }
                    key_group[offset] = key_values[dim];
                }
                tq_wht(q_group);
                const bool quantized = tq_is_quantized_key_type(key.type);
                if (!quantized) {
                    tq_wht(key_group);
                }
                for (size_t offset = 0; offset < q_group.size(); ++offset) {
                    dot += static_cast<double>(q_group[offset]) * key_group[offset];
                }
            }
            const float calibrated = static_cast<float>(
                (dot - static_cast<double>(bias[branch])) /
                std::max(static_cast<double>(scale[branch]), 1.0e-6));
            if (!std::isfinite(calibrated)) {
                return false;
            }
            branch_logits[branch][output_index] = calibrated;
            reconstructed += weights[branch] * calibrated;
        }

        const size_t output_offset =
            static_cast<size_t>(kv) * output.nb[0] +
            static_cast<size_t>(query_index) * output.nb[1] +
            static_cast<size_t>(head) * output.nb[2] +
            static_cast<size_t>(stream) * output.nb[3];
        float actual;
        if (!tq_load_scalar(output, output_offset, actual)) {
            return false;
        }
        if (std::fabs(actual - reconstructed) > 1.0e-4f * (1.0f + std::fabs(actual))) {
            return false;
        }
    }

    metrics = {};
    const llama_tq_layer_config & layer = config.layers[layer_index];
    for (size_t branch = 0; branch < 3; ++branch) {
        double sum = 0.0;
        double sum_squares = 0.0;
        for (float value : branch_logits[branch]) {
            sum += value;
            sum_squares += static_cast<double>(value) * value;
        }
        const double mean = sum / element_count;
        metrics.branches[branch].logit_mean = static_cast<float>(mean);
        metrics.branches[branch].logit_variance = static_cast<float>(
            std::max(0.0, sum_squares / element_count - mean * mean));
        metrics.branches[branch].logit_l2 = static_cast<float>(std::sqrt(sum_squares));
        for (const auto & configured_branch : layer.branches) {
            if (static_cast<size_t>(configured_branch.view) == branch) {
                metrics.branches[branch].expected_quantisation_error = configured_branch.expected_error;
                break;
            }
        }
    }

    const size_t row_width = static_cast<size_t>(output.ne[0]);
    const size_t row_count = element_count / row_width;
    std::array<std::vector<double>, 3> probabilities;
    for (auto & probability : probabilities) {
        probability.resize(row_width);
    }
    double js_sum[3] = {};
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t branch = 0; branch < 3; ++branch) {
            const auto begin = branch_logits[branch].begin() + row * row_width;
            const float maximum = *std::max_element(begin, begin + row_width);
            double normalizer = 0.0;
            for (size_t column = 0; column < row_width; ++column) {
                probabilities[branch][column] = std::exp(
                    static_cast<double>(branch_logits[branch][row * row_width + column] - maximum));
                normalizer += probabilities[branch][column];
            }
            if (!(normalizer > 0.0) || !std::isfinite(normalizer)) {
                return false;
            }
            double entropy = 0.0;
            double top1 = 0.0;
            for (double & probability : probabilities[branch]) {
                probability /= normalizer;
                if (probability > 0.0) {
                    entropy -= probability * std::log(probability);
                }
                top1 = std::max(top1, probability);
            }
            metrics.branches[branch].probability_entropy += static_cast<float>(entropy / row_count);
            metrics.branches[branch].top1_probability += static_cast<float>(top1 / row_count);
        }
        size_t pair = 0;
        for (size_t lhs = 0; lhs < 3; ++lhs) {
            for (size_t rhs = lhs + 1; rhs < 3; ++rhs, ++pair) {
                double divergence = 0.0;
                for (size_t column = 0; column < row_width; ++column) {
                    const double p = probabilities[lhs][column];
                    const double q = probabilities[rhs][column];
                    const double midpoint = 0.5 * (p + q);
                    if (p > 0.0) {
                        divergence += 0.5 * p * std::log(p / midpoint);
                    }
                    if (q > 0.0) {
                        divergence += 0.5 * q * std::log(q / midpoint);
                    }
                }
                js_sum[pair] += divergence;
            }
        }
    }
    for (size_t pair = 0; pair < 3; ++pair) {
        metrics.pairwise_js[pair] = static_cast<float>(js_sum[pair] / row_count);
        metrics.mean_pairwise_js += metrics.pairwise_js[pair] / 3.0f;
        metrics.max_pairwise_js = std::max(metrics.max_pairwise_js, metrics.pairwise_js[pair]);
    }

    const auto eigenvalues = tq_gram_eigenvalues(branch_logits);
    const double maximum_eigenvalue = eigenvalues[0];
    const double eigenvalue_sum = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    if (!(maximum_eigenvalue > 0.0) || !(eigenvalue_sum > 0.0)) {
        return false;
    }
    for (double eigenvalue : eigenvalues) {
        if (eigenvalue > maximum_eigenvalue * 1.0e-6) {
            metrics.numerical_rank += 1.0f;
        }
    }
    double rank_entropy = 0.0;
    for (double eigenvalue : eigenvalues) {
        const double probability = eigenvalue / eigenvalue_sum;
        if (probability > 0.0) {
            rank_entropy -= probability * std::log(probability);
        }
    }
    metrics.effective_rank = static_cast<float>(std::exp(rank_entropy));
    return std::isfinite(metrics.effective_rank);
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

bool llama_tq_telemetry_state::trace_enabled() const {
    return trace_enabled_;
}

bool llama_tq_telemetry_state::wants_graph_tensor(const ggml_tensor * tensor) const {
    return trace_enabled_ && tensor && tensor->op == GGML_OP_TQ_TRIALITY_KQ_CONSENSUS;
}

bool llama_tq_telemetry_state::record_graph_tensor(
        const ggml_tensor * tensor,
        const llama_tq_owned_context_config * config) {
    if (!wants_graph_tensor(tensor) || !config ||
        config->execution != LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS) {
        return false;
    }
    llama_tq_consensus_metrics metrics {};
    if (!tq_compute_metrics(tensor, *config, metrics)) {
        return false;
    }
    record(metrics);
    return true;
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
