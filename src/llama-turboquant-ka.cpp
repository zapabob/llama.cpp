#include "llama-turboquant-ka.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool ka_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

bool ka_weights_valid(const std::array<float, 3> & weights) {
    float sum = 0.0f;
    for (float weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0f) {
            return false;
        }
        sum += weight;
    }
    return std::isfinite(sum) && std::fabs(sum - 1.0f) <= 1e-5f;
}

bool ka_bank_valid(const llama_tq_ka_univariate_bank & bank) {
    if (bank.n_functions == 0 || bank.n_knots < 2 || bank.values.size() != static_cast<size_t>(bank.n_functions) * bank.n_knots) {
        return false;
    }
    if (bank.knots.size() != bank.n_knots && bank.knots.size() != static_cast<size_t>(bank.n_functions) * bank.n_knots) {
        return false;
    }
    for (uint32_t function = 0; function < bank.n_functions; ++function) {
        const float * knots = bank.knots.data() + (bank.knots.size() == bank.n_knots ? 0 : static_cast<size_t>(function) * bank.n_knots);
        for (uint32_t knot = 0; knot < bank.n_knots; ++knot) {
            if (!std::isfinite(knots[knot]) || !std::isfinite(bank.values[static_cast<size_t>(function) * bank.n_knots + knot])) {
                return false;
            }
            if (knot > 0 && !(knots[knot] > knots[knot - 1])) {
                return false;
            }
        }
    }
    return true;
}

const float * ka_knots(const llama_tq_ka_univariate_bank & bank, uint32_t function) {
    return bank.knots.data() + (bank.knots.size() == bank.n_knots ? 0 : static_cast<size_t>(function) * bank.n_knots);
}

const float * ka_values(const llama_tq_ka_univariate_bank & bank, uint32_t function) {
    return bank.values.data() + static_cast<size_t>(function) * bank.n_knots;
}

bool ka_use_fallback(
        const llama_tq_ka_controller & controller,
        bool required,
        std::array<float, 3> & weights,
        bool & fallback_used,
        std::string * error,
        const std::string & message) {
    if (required) {
        return ka_error(error, message);
    }
    if (!ka_weights_valid(controller.fallback_weights)) {
        return ka_error(error, message + "; fallback weights are invalid");
    }
    weights = controller.fallback_weights;
    fallback_used = true;
    if (error) {
        error->clear();
    }
    return true;
}

}

float llama_tq_interp_linear(const float * knots, const float * values, size_t n, float x) {
    if (!knots || !values || n == 0 || !std::isfinite(x)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (n == 1 || x <= knots[0]) {
        return values[0];
    }
    if (x >= knots[n - 1]) {
        return values[n - 1];
    }
    const float * upper = std::upper_bound(knots, knots + n, x);
    const size_t right = static_cast<size_t>(upper - knots);
    const size_t left = right - 1;
    const float width = knots[right] - knots[left];
    if (!(width > 0.0f) || !std::isfinite(width)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float t = (x - knots[left]) / width;
    return values[left] + t * (values[right] - values[left]);
}

bool llama_tq_ka_evaluate(
        const llama_tq_ka_controller & controller,
        const std::vector<float> & coordinates,
        bool required,
        std::array<float, 3> & weights,
        bool & fallback_used,
        std::string * error) {
    fallback_used = false;
    if (error) {
        error->clear();
    }
    if (controller.schema_version != 1) {
        return ka_use_fallback(controller, required, weights, fallback_used, error, "unsupported KA controller schema version");
    }
    if (controller.coordinate_count == 0 || controller.outer_count == 0 || coordinates.size() != controller.coordinate_count ||
        controller.coordinate_min.size() != controller.coordinate_count || controller.coordinate_max.size() != controller.coordinate_count ||
        controller.inner.n_functions != 3 * controller.outer_count * controller.coordinate_count ||
        controller.outer.n_functions != 3 * controller.outer_count || !ka_bank_valid(controller.inner) || !ka_bank_valid(controller.outer)) {
        return ka_use_fallback(controller, required, weights, fallback_used, error, "invalid KA controller tensor shape");
    }

    std::vector<float> normalized(controller.coordinate_count);
    for (uint32_t coordinate = 0; coordinate < controller.coordinate_count; ++coordinate) {
        const float value = coordinates[coordinate];
        const float minimum = controller.coordinate_min[coordinate];
        const float maximum = controller.coordinate_max[coordinate];
        if (!std::isfinite(value) || !std::isfinite(minimum) || !std::isfinite(maximum) || maximum < minimum) {
            return ka_use_fallback(controller, required, weights, fallback_used, error, "KA controller received non-finite coordinates or bounds");
        }
        const float denominator = std::max(maximum - minimum, 1e-6f);
        normalized[coordinate] = std::clamp((value - minimum) / denominator, 0.0f, 1.0f);
    }

    std::vector<float> outer_inputs(3 * controller.outer_count, 0.0f);
    for (uint32_t branch = 0; branch < 3; ++branch) {
        for (uint32_t outer = 0; outer < controller.outer_count; ++outer) {
            double sum = 0.0;
            for (uint32_t coordinate = 0; coordinate < controller.coordinate_count; ++coordinate) {
                const uint32_t function = (branch * controller.outer_count + outer) * controller.coordinate_count + coordinate;
                sum += llama_tq_interp_linear(
                    ka_knots(controller.inner, function),
                    ka_values(controller.inner, function),
                    controller.inner.n_knots,
                    normalized[coordinate]);
            }
            if (!std::isfinite(sum)) {
                return ka_use_fallback(controller, required, weights, fallback_used, error, "KA inner bank produced a non-finite value");
            }
            outer_inputs[branch * controller.outer_count + outer] = static_cast<float>(sum);
        }
    }

    std::array<float, 3> logits {};
    for (uint32_t branch = 0; branch < 3; ++branch) {
        double sum = 0.0;
        for (uint32_t outer = 0; outer < controller.outer_count; ++outer) {
            const uint32_t function = branch * controller.outer_count + outer;
            sum += llama_tq_interp_linear(
                ka_knots(controller.outer, function),
                ka_values(controller.outer, function),
                controller.outer.n_knots,
                outer_inputs[branch * controller.outer_count + outer]);
        }
        if (!std::isfinite(sum)) {
            return ka_use_fallback(controller, required, weights, fallback_used, error, "KA outer bank produced a non-finite value");
        }
        logits[branch] = static_cast<float>(sum);
    }

    const float maximum = *std::max_element(logits.begin(), logits.end());
    double denominator = 0.0;
    for (uint32_t branch = 0; branch < 3; ++branch) {
        weights[branch] = std::exp(logits[branch] - maximum);
        denominator += weights[branch];
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
        return ka_use_fallback(controller, required, weights, fallback_used, error, "KA softmax normalization failed");
    }
    for (float & weight : weights) {
        weight = static_cast<float>(weight / denominator);
    }
    if (!ka_weights_valid(weights)) {
        return ka_use_fallback(controller, required, weights, fallback_used, error, "KA controller produced invalid weights");
    }
    return true;
}
