#include "testing.h"

#include "../src/llama-turboquant-ka.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace {

llama_tq_ka_controller make_equivariant_controller() {
    llama_tq_ka_controller controller;
    controller.coordinate_count = 3;
    controller.outer_count = 1;
    controller.coordinate_min = {0.0f, 0.0f, 0.0f};
    controller.coordinate_max = {1.0f, 1.0f, 1.0f};
    controller.inner.n_functions = 9;
    controller.inner.n_knots = 2;
    controller.inner.knots = {0.0f, 1.0f};
    controller.inner.values.assign(18, 0.0f);
    for (uint32_t branch = 0; branch < 3; ++branch) {
        const size_t function = static_cast<size_t>(branch) * 3 + branch;
        controller.inner.values[function * 2 + 1] = 1.0f;
    }
    controller.outer.n_functions = 3;
    controller.outer.n_knots = 2;
    controller.outer.knots = {0.0f, 1.0f};
    controller.outer.values = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    return controller;
}

bool near(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1e-6f;
}

}

int main() {
    testing t;

    t.test("piecewise_linear_interpolation_is_continuous_at_knots", [](testing & t) {
        const float knots[] = {0.0f, 0.5f, 1.0f};
        const float values[] = {0.0f, 2.0f, 4.0f};
        t.assert_true("left limit", near(llama_tq_interp_linear(knots, values, 3, 0.5f - 1e-7f), 2.0f));
        t.assert_true("knot value", near(llama_tq_interp_linear(knots, values, 3, 0.5f), 2.0f));
        t.assert_true("right limit", near(llama_tq_interp_linear(knots, values, 3, 0.5f + 1e-7f), 2.0f));
    });

    t.test("ka_controller_is_s3_permutation_equivariant", [](testing & t) {
        const auto controller = make_equivariant_controller();
        std::array<float, 3> original {};
        std::array<float, 3> permuted {};
        bool fallback = false;
        std::string error;
        t.assert_true("original evaluation", llama_tq_ka_evaluate(controller, {0.1f, 0.5f, 0.9f}, true, original, fallback, &error));
        t.assert_true("permuted evaluation", llama_tq_ka_evaluate(controller, {0.9f, 0.1f, 0.5f}, true, permuted, fallback, &error));
        t.assert_true("branch zero follows permutation", near(permuted[0], original[2]));
        t.assert_true("branch one follows permutation", near(permuted[1], original[0]));
        t.assert_true("branch two follows permutation", near(permuted[2], original[1]));
    });

    t.test("optional_unsupported_controller_uses_validated_static_fallback", [](testing & t) {
        auto controller = make_equivariant_controller();
        controller.schema_version = 99;
        controller.fallback_weights = {0.2f, 0.3f, 0.5f};
        std::array<float, 3> weights {};
        bool fallback = false;
        std::string error;
        t.assert_true("optional controller succeeds", llama_tq_ka_evaluate(controller, {0.0f, 0.0f, 0.0f}, false, weights, fallback, &error));
        t.assert_true("fallback selected", fallback);
        t.assert_true("fallback copied", near(weights[0], 0.2f) && near(weights[1], 0.3f) && near(weights[2], 0.5f));
    });

    t.test("required_unsupported_controller_fails_closed", [](testing & t) {
        auto controller = make_equivariant_controller();
        controller.schema_version = 99;
        std::array<float, 3> weights {};
        bool fallback = false;
        std::string error;
        t.assert_true("required controller fails", !llama_tq_ka_evaluate(controller, {0.0f, 0.0f, 0.0f}, true, weights, fallback, &error));
        t.assert_true("error is preserved", !error.empty());
    });

    return t.summary();
}
