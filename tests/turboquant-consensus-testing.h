#pragma once

#include <exception>
#include <iostream>
#include <string>

struct tq_testing {
    int tests      = 0;
    int assertions = 0;
    int failures   = 0;

    template <typename Function> void test(const std::string & name, Function function) {
        ++tests;
        const int previous_failures = failures;
        try {
            function(*this);
        } catch (const std::exception & exception) {
            ++failures;
            std::cerr << name << ": exception: " << exception.what() << "\n";
        } catch (...) {
            ++failures;
            std::cerr << name << ": unknown exception\n";
        }
        std::cout << name << (failures == previous_failures ? " [PASS]\n" : " [FAIL]\n");
    }

    bool assert_true(const std::string & label, bool condition) {
        ++assertions;
        if (!condition) {
            ++failures;
            std::cerr << "  " << label << "\n";
        }
        return condition;
    }

    template <typename Expected, typename Actual>
    bool assert_equal(const std::string & label, const Expected & expected, const Actual & actual) {
        return assert_true(label, expected == actual);
    }

    int summary() const {
        std::cout << tests << " test(s), " << assertions << " assertion(s), " << failures << " failure(s)\n";
        return failures == 0 ? 0 : 1;
    }
};
