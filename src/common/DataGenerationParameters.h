#pragma once

#include <cstddef>

struct DataGenerationParameters {
    // For Matrix category
    std::size_t leftRows;
    std::size_t common;
    std::size_t rightColumns;

    // For Quadratic Sieve category
    std::size_t leftOperandDigits;
    std::size_t rightOperandDigits;
};
