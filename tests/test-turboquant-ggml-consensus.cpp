#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef TQ_CONSENSUS_CUDA_COMPILER_VERSION
#define TQ_CONSENSUS_CUDA_COMPILER_VERSION "unavailable"
#endif

namespace {

constexpr int64_t head_dim = 128;
constexpr int64_t n_query = 3;
constexpr int64_t n_kv = 5;
constexpr int64_t n_head = 4;
constexpr int64_t n_kv_head = 2;
constexpr int64_t n_stream = 2;
constexpr int64_t n_k_stream = 1;

const std::array<float, 3> weights = { 0.5f, 0.3f, 0.2f };
const std::array<float, 3> bias = { 0.0f, 0.1f, -0.2f };
const std::array<float, 3> scale = { 1.0f, 1.2f, 0.8f };
const std::array<float, 3> temperature = { 1.0f, 0.5f, 2.0f };
const std::array<float, 128> wht_s1 = {-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
const std::array<float, 128> wht_s2 = {1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};

struct context_owner {
    ggml_context * value = nullptr;

    context_owner() = default;
    context_owner(const context_owner &) = delete;
    context_owner & operator=(const context_owner &) = delete;

    context_owner(context_owner && other) noexcept : value(other.value) {
        other.value = nullptr;
    }

    ~context_owner() {
        if (value != nullptr) {
            ggml_free(value);
        }
    }
};

struct buffer_owner {
    ggml_backend_buffer_t value = nullptr;

    ~buffer_owner() {
        if (value != nullptr) {
            ggml_backend_buffer_free(value);
        }
    }
};

struct backend_owner {
    ggml_backend_t value = nullptr;

    ~backend_owner() {
        if (value != nullptr) {
            ggml_backend_free(value);
        }
    }
};

struct tensor_input {
    ggml_tensor * storage = nullptr;
    ggml_tensor * view = nullptr;
    std::vector<float> logical;
};

struct graph_fixture {
    context_owner ctx;
    tensor_input q;
    std::array<tensor_input, 3> k;
    std::array<ggml_tensor *, 3> rotations = { nullptr, nullptr, nullptr };
    std::array<std::vector<float>, 3> rotation_values;
    ggml_tensor * output = nullptr;
};

struct execution_result {
    std::vector<float> output;
    double reference_relative_error = 0.0;
};

struct latency_result {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

size_t logical_index(
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    return static_cast<size_t>(((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0);
}

tensor_input make_input(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        int64_t row_padding,
        float phase) {
    tensor_input result;
    result.storage = ggml_new_tensor_4d(ctx, type, ne0 + row_padding, ne1, ne2, ne3);
    result.view = ggml_view_4d(
        ctx,
        result.storage,
        ne0,
        ne1,
        ne2,
        ne3,
        result.storage->nb[1],
        result.storage->nb[2],
        result.storage->nb[3],
        0);
    result.logical.resize(static_cast<size_t>(ne0 * ne1 * ne2 * ne3));
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const size_t index = logical_index(i0, i1, i2, i3, ne0, ne1, ne2);
                    result.logical[index] =
                        0.65f * std::sin(phase + 0.17f * static_cast<float>(index)) +
                        0.25f * std::cos(0.11f * static_cast<float>(i0 + 3 * i1 + 5 * i2 + 7 * i3));
                }
            }
        }
    }
    return result;
}

std::vector<float> make_rotation(float angle, bool per_block) {
    const int64_t n_blocks = head_dim / 8;
    std::vector<float> result(
        per_block ? static_cast<size_t>(64 * n_blocks) : static_cast<size_t>(head_dim * head_dim),
        0.0f);
    for (int64_t block = 0; block < n_blocks; ++block) {
        const float theta = angle * static_cast<float>(block + 1);
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        float matrix[64] = {};
        for (int row = 0; row < 8; ++row) {
            matrix[row * 8 + row] = 1.0f;
        }
        matrix[0 * 8 + 0] = c;
        matrix[0 * 8 + 1] = -s;
        matrix[1 * 8 + 0] = s;
        matrix[1 * 8 + 1] = c;
        matrix[4 * 8 + 4] = c;
        matrix[4 * 8 + 5] = -s;
        matrix[5 * 8 + 4] = s;
        matrix[5 * 8 + 5] = c;
        if (per_block) {
            std::copy(matrix, matrix + 64, result.data() + block * 64);
        } else {
            for (int row = 0; row < 8; ++row) {
                for (int col = 0; col < 8; ++col) {
                    result[static_cast<size_t>((block * 8 + row) * head_dim + block * 8 + col)] =
                        matrix[row * 8 + col];
                }
            }
        }
    }
    return result;
}

graph_fixture make_fixture(
        ggml_type type,
        int64_t row_padding,
        const std::array<ggml_type, 3> * key_types = nullptr) {
    graph_fixture fixture;
    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    fixture.ctx.value = ggml_init(params);
    require(fixture.ctx.value != nullptr, "ggml_init failed");

    fixture.q = make_input(
        fixture.ctx.value, type, head_dim, n_query, n_head, n_stream, row_padding, 0.2f);
    for (int branch = 0; branch < 3; ++branch) {
        const ggml_type key_type = key_types == nullptr ? type : (*key_types)[branch];
        const bool quantized =
            key_type == GGML_TYPE_TURBO2_0 ||
            key_type == GGML_TYPE_TURBO3_0 ||
            key_type == GGML_TYPE_TURBO4_0;
        fixture.k[branch] = make_input(
            fixture.ctx.value,
            key_type,
            head_dim,
            n_kv,
            n_kv_head,
            n_k_stream,
            quantized ? 0 : row_padding + branch,
            0.5f + 0.3f * static_cast<float>(branch));
    }

    fixture.rotation_values[1] = make_rotation(0.25f, false);
    fixture.rotations[1] = ggml_new_tensor_4d(
        fixture.ctx.value, GGML_TYPE_F32, head_dim, head_dim, 1, 1);
    fixture.rotation_values[2] = make_rotation(-0.15f, true);
    fixture.rotations[2] = ggml_new_tensor_4d(fixture.ctx.value, GGML_TYPE_F32, 8, 8, head_dim / 8, 1);

    fixture.output = ggml_tq_triality_kq_consensus(
        fixture.ctx.value,
        fixture.q.view,
        fixture.k[0].view,
        fixture.k[1].view,
        fixture.k[2].view,
        fixture.rotations[0],
        fixture.rotations[1],
        fixture.rotations[2],
        weights.data(),
        bias.data(),
        scale.data(),
        temperature.data());
    return fixture;
}

void set_tensor_input(tensor_input & input) {
    const int64_t storage_ne0 = input.storage->ne[0];
    const int64_t ne0 = input.view->ne[0];
    const int64_t ne1 = input.view->ne[1];
    const int64_t ne2 = input.view->ne[2];
    const int64_t ne3 = input.view->ne[3];

    if (input.storage->type == GGML_TYPE_F32) {
        std::vector<float> storage(static_cast<size_t>(ggml_nelements(input.storage)), -1000.0f);
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const size_t src = logical_index(i0, i1, i2, i3, ne0, ne1, ne2);
                        const size_t dst = logical_index(i0, i1, i2, i3, storage_ne0, ne1, ne2);
                        storage[dst] = input.logical[src];
                    }
                }
            }
        }
        ggml_backend_tensor_set(input.storage, storage.data(), 0, storage.size() * sizeof(float));
        return;
    }

    const bool quantized =
        input.storage->type == GGML_TYPE_TURBO2_0 ||
        input.storage->type == GGML_TYPE_TURBO3_0 ||
        input.storage->type == GGML_TYPE_TURBO4_0;
    if (quantized) {
        std::vector<uint8_t> storage(ggml_nbytes(input.storage));
        std::array<float, head_dim> dequantized;
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const size_t logical_offset = logical_index(0, i1, i2, i3, ne0, ne1, ne2);
                    uint8_t * row = storage.data() +
                        i1 * input.storage->nb[1] + i2 * input.storage->nb[2] + i3 * input.storage->nb[3];
                    const float * source = input.logical.data() + logical_offset;
                    if (input.storage->type == GGML_TYPE_TURBO2_0) {
                        quantize_row_turbo2_0_ref(source, (block_turbo2_0 *) row, ne0);
                        dequantize_row_turbo2_0((const block_turbo2_0 *) row, dequantized.data(), ne0);
                    } else if (input.storage->type == GGML_TYPE_TURBO3_0) {
                        quantize_row_turbo3_0_ref(source, (block_turbo3_0 *) row, ne0);
                        dequantize_row_turbo3_0((const block_turbo3_0 *) row, dequantized.data(), ne0);
                    } else {
                        quantize_row_turbo4_0_ref(source, (block_turbo4_0 *) row, ne0);
                        dequantize_row_turbo4_0((const block_turbo4_0 *) row, dequantized.data(), ne0);
                    }
                    std::copy(dequantized.begin(), dequantized.end(), input.logical.begin() + logical_offset);
                }
            }
        }
        ggml_backend_tensor_set(input.storage, storage.data(), 0, storage.size());
        return;
    }

    std::vector<ggml_fp16_t> storage(static_cast<size_t>(ggml_nelements(input.storage)));
    const ggml_fp16_t padding = ggml_fp32_to_fp16(-1000.0f);
    std::fill(storage.begin(), storage.end(), padding);
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const size_t src = logical_index(i0, i1, i2, i3, ne0, ne1, ne2);
                    const size_t dst = logical_index(i0, i1, i2, i3, storage_ne0, ne1, ne2);
                    storage[dst] = ggml_fp32_to_fp16(input.logical[src]);
                }
            }
        }
    }
    ggml_backend_tensor_set(input.storage, storage.data(), 0, storage.size() * sizeof(ggml_fp16_t));
}

float rotated_q(
        const tensor_input & q,
        const std::vector<float> & rotation,
        int64_t dim,
        int64_t query,
        int64_t head,
        int64_t stream,
        bool round_to_f16) {
    const int64_t block = dim / 8;
    const int64_t row = dim % 8;
    if (rotation.empty()) {
        float value = q.logical[logical_index(dim, query, head, stream, head_dim, n_query, n_head)];
        return round_to_f16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)) : value;
    }

    float value = 0.0f;
    for (int64_t col = 0; col < 8; ++col) {
        const int64_t q_dim = block * 8 + col;
        const float coefficient = rotation.size() == static_cast<size_t>(head_dim * head_dim)
            ? rotation[static_cast<size_t>((block * 8 + row) * head_dim + q_dim)]
            : rotation[static_cast<size_t>(block * 64 + row * 8 + col)];
        float q_value = q.logical[logical_index(q_dim, query, head, stream, head_dim, n_query, n_head)];
        if (round_to_f16) {
            q_value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(q_value));
        }
        value += coefficient * q_value;
    }
    return value;
}

std::array<float, 128> transformed_q(
        const tensor_input & q,
        const std::vector<float> & rotation,
        int64_t query,
        int64_t head,
        int64_t stream,
        bool round_to_f16) {
    std::array<float, 128> values;
    for (int64_t dim = 0; dim < 128; ++dim) {
        values[static_cast<size_t>(dim)] = rotated_q(
            q, rotation, dim, query, head, stream, round_to_f16) * wht_s1[static_cast<size_t>(dim)];
    }
    for (int stride = 1; stride < 128; stride *= 2) {
        for (int base = 0; base < 128; base += 2 * stride) {
            for (int offset = 0; offset < stride; ++offset) {
                const float x = values[static_cast<size_t>(base + offset)];
                const float y = values[static_cast<size_t>(base + offset + stride)];
                values[static_cast<size_t>(base + offset)] = x + y;
                values[static_cast<size_t>(base + offset + stride)] = x - y;
            }
        }
    }
    for (int64_t dim = 0; dim < 128; ++dim) {
        values[static_cast<size_t>(dim)] *= wht_s2[static_cast<size_t>(dim)] * 0.08838834764831845f;
    }
    return values;
}

std::vector<float> reference_output(const graph_fixture & fixture, bool round_to_f16) {
    std::vector<float> result(static_cast<size_t>(n_kv * n_query * n_head * n_stream));
    const int64_t heads_per_k_head = n_head / n_kv_head;
    const int64_t streams_per_k_stream = n_stream / n_k_stream;

    for (int64_t stream = 0; stream < n_stream; ++stream) {
        for (int64_t head = 0; head < n_head; ++head) {
            for (int64_t query = 0; query < n_query; ++query) {
                for (int64_t kv = 0; kv < n_kv; ++kv) {
                    float consensus = 0.0f;
                    for (int branch = 0; branch < 3; ++branch) {
                        const ggml_type key_type = fixture.k[branch].view->type;
                        const bool quantized =
                            key_type == GGML_TYPE_TURBO2_0 ||
                            key_type == GGML_TYPE_TURBO3_0 ||
                            key_type == GGML_TYPE_TURBO4_0;
                        std::array<float, 128> q_values = {};
                        if (quantized) {
                            q_values = transformed_q(
                                fixture.q,
                                fixture.rotation_values[branch],
                                query,
                                head,
                                stream,
                                round_to_f16);
                        }
                        float dot = 0.0f;
                        for (int64_t dim = 0; dim < head_dim; ++dim) {
                            const float q_value = quantized
                                ? q_values[static_cast<size_t>(dim)]
                                : rotated_q(
                                    fixture.q,
                                    fixture.rotation_values[branch],
                                    dim,
                                    query,
                                    head,
                                    stream,
                                    round_to_f16);
                            float k_value = fixture.k[branch].logical[logical_index(
                                dim,
                                kv,
                                head / heads_per_k_head,
                                stream / streams_per_k_stream,
                                head_dim,
                                n_kv,
                                n_kv_head)];
                            if (key_type == GGML_TYPE_F16) {
                                k_value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(k_value));
                            }
                            dot += q_value * k_value;
                        }
                        consensus += weights[branch] *
                            ((dot - bias[branch]) / std::max(scale[branch], 1.0e-6f));
                    }
                    result[logical_index(kv, query, head, stream, n_kv, n_query, n_head)] = consensus;
                }
            }
        }
    }
    return result;
}

execution_result execute(
        ggml_backend_t backend,
        ggml_type type,
        int64_t row_padding,
        const std::array<ggml_type, 3> * key_types = nullptr) {
    graph_fixture fixture = make_fixture(type, row_padding, key_types);
    require(ggml_backend_supports_op(backend, fixture.output), "backend rejected a supported consensus layout");

    buffer_owner buffer;
    buffer.value = ggml_backend_alloc_ctx_tensors(fixture.ctx.value, backend);
    require(buffer.value != nullptr, "backend tensor allocation failed");

    set_tensor_input(fixture.q);
    for (tensor_input & k : fixture.k) {
        set_tensor_input(k);
    }
    for (int branch = 0; branch < 3; ++branch) {
        if (fixture.rotations[branch] != nullptr) {
            ggml_backend_tensor_set(
                fixture.rotations[branch],
                fixture.rotation_values[branch].data(),
                0,
                fixture.rotation_values[branch].size() * sizeof(float));
        }
    }

    ggml_cgraph * graph = ggml_new_graph(fixture.ctx.value);
    ggml_build_forward_expand(graph, fixture.output);
    require(
        ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
        "backend graph execution failed");
    ggml_backend_synchronize(backend);

    std::vector<float> output(static_cast<size_t>(ggml_nelements(fixture.output)));
    ggml_backend_tensor_get(fixture.output, output.data(), 0, output.size() * sizeof(float));

    const std::vector<float> expected = reference_output(fixture, type == GGML_TYPE_F16);
    double error_sq = 0.0;
    double expected_sq = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        const double delta = static_cast<double>(output[i]) - expected[i];
        error_sq += delta * delta;
        expected_sq += static_cast<double>(expected[i]) * expected[i];
    }
    const double relative_error = std::sqrt(error_sq) / std::max(std::sqrt(expected_sq), 1.0e-8);
    require(relative_error <= 1.0e-5, "backend result differs from the scalar reference");
    return { std::move(output), relative_error };
}

float execute_dense_golden(ggml_backend_t backend, ggml_type type, int64_t dim) {
    context_owner ctx;
    ggml_init_params params = { 2 * 1024 * 1024, nullptr, true };
    ctx.value = ggml_init(params);
    require(ctx.value != nullptr, "dense golden ggml_init failed");

    tensor_input q = make_input(ctx.value, type, dim, 1, 1, 1, 0, 0.35f);
    std::array<tensor_input, 3> keys;
    for (int branch = 0; branch < 3; ++branch) {
        keys[branch] = make_input(
            ctx.value, type, dim, 1, 1, 1, 0, 0.65f + 0.2f * static_cast<float>(branch));
    }
    ggml_tensor * output = ggml_tq_triality_kq_consensus(
        ctx.value,
        q.view,
        keys[0].view,
        keys[1].view,
        keys[2].view,
        nullptr,
        nullptr,
        nullptr,
        weights.data(),
        bias.data(),
        scale.data(),
        temperature.data());
    require(ggml_backend_supports_op(backend, output), "backend rejected a dense golden layout");

    buffer_owner buffer;
    buffer.value = ggml_backend_alloc_ctx_tensors(ctx.value, backend);
    require(buffer.value != nullptr, "dense golden allocation failed");
    set_tensor_input(q);
    for (tensor_input & key : keys) {
        set_tensor_input(key);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.value);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "dense golden execution failed");
    ggml_backend_synchronize(backend);

    float actual = 0.0f;
    ggml_backend_tensor_get(output, &actual, 0, sizeof(actual));
    float expected = 0.0f;
    for (int branch = 0; branch < 3; ++branch) {
        float dot = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            float q_value = q.logical[static_cast<size_t>(i)];
            float k_value = keys[branch].logical[static_cast<size_t>(i)];
            if (type == GGML_TYPE_F16) {
                q_value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(q_value));
                k_value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(k_value));
            }
            dot += q_value * k_value;
        }
        expected += weights[branch] *
            ((dot - bias[branch]) / std::max(scale[branch], 1.0e-6f));
    }
    const double relative_error = std::abs(static_cast<double>(actual) - expected) /
        std::max(std::abs(static_cast<double>(expected)), 1.0e-8);
    require(relative_error <= 2.0e-5, "dense golden result differs from direct dot reference");
    return actual;
}

ggml_backend_dev_t find_cuda_device() {
    ggml_backend_load_all();
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(device);
        if (name != nullptr && std::strstr(name, "CUDA") != nullptr) {
            return device;
        }
    }
    return nullptr;
}

double require_vectors_close(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        double tolerance,
        const char * message) {
    require(lhs.size() == rhs.size(), "output sizes differ");
    double error_sq = 0.0;
    double lhs_sq = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const double delta = static_cast<double>(lhs[i]) - rhs[i];
        error_sq += delta * delta;
        lhs_sq += static_cast<double>(lhs[i]) * lhs[i];
    }
    const double relative_error = std::sqrt(error_sq) / std::max(std::sqrt(lhs_sq), 1.0e-8);
    require(relative_error <= tolerance, message);
    return relative_error;
}

void test_unsupported_cuda_layout(ggml_backend_t cuda) {
    graph_fixture fixture = make_fixture(GGML_TYPE_F32, 0);
    fixture.q.view->nb[0] = 2 * sizeof(float);
    require(
        !ggml_backend_supports_op(cuda, fixture.output),
        "CUDA must reject a non-dense feature dimension");
}

latency_result benchmark_cuda(
        ggml_backend_t backend,
        int64_t query_count,
        int64_t kv_count,
        int warmup,
        int repeats) {
    context_owner ctx;
    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ctx.value = ggml_init(params);
    require(ctx.value != nullptr, "benchmark ggml_init failed");

    ggml_tensor * q = ggml_new_tensor_4d(ctx.value, GGML_TYPE_F32, head_dim, query_count, n_head, 1);
    std::array<ggml_tensor *, 3> keys;
    for (ggml_tensor * & key : keys) {
        key = ggml_new_tensor_4d(ctx.value, GGML_TYPE_F32, head_dim, kv_count, n_kv_head, 1);
    }
    ggml_tensor * output = ggml_tq_triality_kq_consensus(
        ctx.value,
        q,
        keys[0],
        keys[1],
        keys[2],
        nullptr,
        nullptr,
        nullptr,
        weights.data(),
        bias.data(),
        scale.data(),
        temperature.data());
    require(ggml_backend_supports_op(backend, output), "CUDA rejected benchmark layout");

    buffer_owner buffer;
    buffer.value = ggml_backend_alloc_ctx_tensors(ctx.value, backend);
    require(buffer.value != nullptr, "benchmark allocation failed");

    std::vector<float> q_data(static_cast<size_t>(ggml_nelements(q)));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = std::sin(0.013f * static_cast<float>(i));
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    for (int branch = 0; branch < 3; ++branch) {
        std::vector<float> key_data(static_cast<size_t>(ggml_nelements(keys[branch])));
        for (size_t i = 0; i < key_data.size(); ++i) {
            key_data[i] = std::cos(
                0.009f * static_cast<float>(i) + 0.2f * static_cast<float>(branch));
        }
        ggml_backend_tensor_set(keys[branch], key_data.data(), 0, key_data.size() * sizeof(float));
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.value);
    ggml_build_forward_expand(graph, output);
    ggml_backend_synchronize(backend);
    for (int i = 0; i < warmup; ++i) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "benchmark warmup failed");
        ggml_backend_synchronize(backend);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeats));
    double sum_ms = 0.0;
    for (int i = 0; i < repeats; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "benchmark repeat failed");
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        samples.push_back(elapsed_ms);
        sum_ms += elapsed_ms;
    }
    std::sort(samples.begin(), samples.end());
    const size_t p50_index = static_cast<size_t>(std::ceil(0.50 * samples.size())) - 1;
    const size_t p95_index = static_cast<size_t>(std::ceil(0.95 * samples.size())) - 1;
    return { sum_ms / samples.size(), samples[p50_index], samples[p95_index] };
}

}

int main() {
    try {
        backend_owner cpu;
        cpu.value = ggml_backend_cpu_init();
        require(cpu.value != nullptr, "CPU backend initialization failed");

        const execution_result cpu_f32 = execute(cpu.value, GGML_TYPE_F32, 3);
        const execution_result cpu_f16 = execute(cpu.value, GGML_TYPE_F16, 2);
        const std::array<ggml_type, 3> quantized_key_types = {
            GGML_TYPE_TURBO2_0,
            GGML_TYPE_TURBO3_0,
            GGML_TYPE_TURBO4_0,
        };
        const execution_result cpu_quantized = execute(
            cpu.value, GGML_TYPE_F32, 0, &quantized_key_types);
        std::array<float, 8> cpu_dense = {};
        int dense_index = 0;
        for (int64_t dim : { int64_t(8), int64_t(48), int64_t(64), int64_t(128) }) {
            cpu_dense[dense_index++] = execute_dense_golden(cpu.value, GGML_TYPE_F32, dim);
            cpu_dense[dense_index++] = execute_dense_golden(cpu.value, GGML_TYPE_F16, dim);
        }
        require(!cpu_f32.output.empty() && !cpu_f16.output.empty(), "CPU coverage produced no output");

        ggml_backend_dev_t cuda_device = find_cuda_device();
        if (cuda_device != nullptr) {
            backend_owner cuda;
            cuda.value = ggml_backend_dev_init(cuda_device, nullptr);
            require(cuda.value != nullptr, "CUDA backend initialization failed");
            const execution_result cuda_f32 = execute(cuda.value, GGML_TYPE_F32, 3);
            const execution_result cuda_f16 = execute(cuda.value, GGML_TYPE_F16, 2);
            const execution_result cuda_quantized = execute(
                cuda.value, GGML_TYPE_F32, 0, &quantized_key_types);
            const char * gpu_name = ggml_backend_dev_description(cuda_device);
            std::printf(
                "{\"event\":\"metadata\",\"mode\":\"triality_consensus_attention\","
                "\"gpu\":\"%s\",\"cuda_compiler\":\"%s\",\"commit\":\"%s\"}\n",
                gpu_name == nullptr ? "unknown" : gpu_name,
                TQ_CONSENSUS_CUDA_COMPILER_VERSION,
                ggml_commit());

            double max_relative_error = 0.0;
            const double f32_error = require_vectors_close(
                cpu_f32.output,
                cuda_f32.output,
                1.0e-4,
                "CPU/CUDA F32 relative error exceeded 1e-4");
            const double f16_error = require_vectors_close(
                cpu_f16.output,
                cuda_f16.output,
                1.0e-4,
                "CPU/CUDA F16 relative error exceeded 1e-4");
            const double quantized_error = require_vectors_close(
                cpu_quantized.output,
                cuda_quantized.output,
                1.0e-4,
                "CPU/CUDA quantized-key relative error exceeded 1e-4");
            std::printf(
                "{\"event\":\"accuracy\",\"case\":\"dense_f32_strided\","
                "\"mode\":\"prefill\",\"shape\":\"q=3,kv=5,h=4,s=2,d=128\","
                "\"dtype\":\"f32\",\"metric\":\"attention_output_frobenius_relative_error\","
                "\"value\":%.9g}\n",
                f32_error);
            std::printf(
                "{\"event\":\"accuracy\",\"case\":\"dense_f16_strided\","
                "\"mode\":\"prefill\",\"shape\":\"q=3,kv=5,h=4,s=2,d=128\","
                "\"dtype\":\"f16\",\"metric\":\"attention_output_frobenius_relative_error\","
                "\"value\":%.9g}\n",
                f16_error);
            std::printf(
                "{\"event\":\"accuracy\",\"case\":\"turbo2_3_4_mixed\","
                "\"mode\":\"prefill\",\"shape\":\"q=3,kv=5,h=4,s=2,d=128\","
                "\"dtype\":\"f32/turbo2_0/turbo3_0/turbo4_0\","
                "\"metric\":\"attention_output_frobenius_relative_error\",\"value\":%.9g}\n",
                quantized_error);
            max_relative_error = std::max({ f32_error, f16_error, quantized_error });
            dense_index = 0;
            for (int64_t dim : { int64_t(8), int64_t(48), int64_t(64), int64_t(128) }) {
                const float cuda_f32_dense = execute_dense_golden(cuda.value, GGML_TYPE_F32, dim);
                const float cuda_f16_dense = execute_dense_golden(cuda.value, GGML_TYPE_F16, dim);
                const double dense_f32_error = require_vectors_close(
                    { cpu_dense[dense_index++] },
                    { cuda_f32_dense },
                    1.0e-4,
                    "CPU/CUDA dense F32 golden mismatch");
                const double dense_f16_error = require_vectors_close(
                    { cpu_dense[dense_index++] },
                    { cuda_f16_dense },
                    1.0e-4,
                    "CPU/CUDA dense F16 golden mismatch");
                std::printf(
                    "{\"event\":\"accuracy\",\"case\":\"dense_direct_dot\","
                    "\"mode\":\"decode\",\"shape\":\"q=1,kv=1,h=1,s=1,d=%lld\","
                    "\"dtype\":\"f32\",\"metric\":\"attention_output_frobenius_relative_error\","
                    "\"value\":%.9g}\n",
                    static_cast<long long>(dim),
                    dense_f32_error);
                std::printf(
                    "{\"event\":\"accuracy\",\"case\":\"dense_direct_dot\","
                    "\"mode\":\"decode\",\"shape\":\"q=1,kv=1,h=1,s=1,d=%lld\","
                    "\"dtype\":\"f16\",\"metric\":\"attention_output_frobenius_relative_error\","
                    "\"value\":%.9g}\n",
                    static_cast<long long>(dim),
                    dense_f16_error);
                max_relative_error = std::max({ max_relative_error, dense_f32_error, dense_f16_error });
            }
            std::printf(
                "{\"event\":\"accuracy_summary\","
                "\"metric\":\"max_attention_output_frobenius_relative_error\",\"value\":%.9g}\n",
                max_relative_error);

            constexpr int warmup = 5;
            constexpr int repeats = 20;
            const latency_result prefill_latency = benchmark_cuda(cuda.value, 16, 64, warmup, repeats);
            const latency_result decode_latency = benchmark_cuda(cuda.value, 1, 64, warmup, repeats);
            std::printf(
                "{\"event\":\"latency\",\"mode\":\"prefill\","
                "\"shape\":\"q=16,kv=64,h=4,s=1,d=128\",\"dtype\":\"f32\","
                "\"warmup\":%d,\"repeats\":%d,\"gpu_synchronized\":true,"
                "\"mean_ms\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g}\n",
                warmup,
                repeats,
                prefill_latency.mean_ms,
                prefill_latency.p50_ms,
                prefill_latency.p95_ms);
            std::printf(
                "{\"event\":\"latency\",\"mode\":\"decode\","
                "\"shape\":\"q=1,kv=64,h=4,s=1,d=128\",\"dtype\":\"f32\","
                "\"warmup\":%d,\"repeats\":%d,\"gpu_synchronized\":true,"
                "\"mean_ms\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g}\n",
                warmup,
                repeats,
                decode_latency.mean_ms,
                decode_latency.p50_ms,
                decode_latency.p95_ms);

            size_t free_vram = 0;
            size_t total_vram = 0;
            ggml_backend_dev_memory(cuda_device, &free_vram, &total_vram);
            std::printf(
                "{\"event\":\"memory\",\"total_vram_bytes\":%zu,\"free_vram_bytes\":%zu,"
                "\"peak_vram_bytes\":null,\"reason\":\"backend API has no per-operation peak counter\"}\n",
                total_vram,
                free_vram);
            std::printf(
                "{\"event\":\"unavailable_metrics\",\"next_logit_kl\":null,"
                "\"hidden_state_cosine\":null,"
                "\"reason\":\"standalone GGML attention-score operator has no logits or hidden states\"}\n");
            test_unsupported_cuda_layout(cuda.value);
        } else {
            std::fprintf(stderr, "CUDA backend unavailable; CUDA parity coverage skipped\n");
        }

        std::printf("TurboQuant GGML consensus tests passed\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "TurboQuant GGML consensus test failed: %s\n", error.what());
        return 1;
    }
}
