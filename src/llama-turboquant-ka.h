#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_tq_ka_univariate_bank {
    uint32_t n_functions = 0;
    uint32_t n_knots = 0;
    std::vector<float> knots;
    std::vector<float> values;
};

struct llama_tq_ka_controller {
    uint32_t schema_version = 1;
    uint32_t coordinate_count = 0;
    uint32_t outer_count = 0;
    bool s3_equivariant = true;
    std::vector<float> coordinate_min;
    std::vector<float> coordinate_max;
    llama_tq_ka_univariate_bank inner;
    llama_tq_ka_univariate_bank outer;
    std::array<float, 3> fallback_weights {1.0f, 0.0f, 0.0f};
};

float llama_tq_interp_linear(
    const float * knots,
    const float * values,
    size_t n,
    float x);

bool llama_tq_ka_evaluate(
    const llama_tq_ka_controller & controller,
    const std::vector<float> & coordinates,
    bool required,
    std::array<float, 3> & weights,
    bool & fallback_used,
    std::string * error);
