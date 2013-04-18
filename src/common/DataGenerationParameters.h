#pragma once

#include <cstddef>

struct DataGenerationParameters {
    // For Matrix category
    std::size_t leftRows;
    std::size_t common;
    std::size_t rightColumns;

    // Additional for Matrix with online scheduling category
    std::string schedulingStrategy;

    // For Quadratic Sieve category
    std::size_t leftOperandDigits;
    std::size_t rightOperandDigits;

    // For Monte Carlo Tree Search (Othello)
    std::size_t monteCarloTrials; 
};
