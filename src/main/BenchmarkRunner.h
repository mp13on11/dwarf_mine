#pragma once

#include <vector>
#include <chrono>
#include "BenchmarkResults.h"
#include "ProblemStatement.h"
#include "../elves/Elf.h"

class BenchmarkRunner
{
private:
    size_t _iterations;
    int _rank;
    int _devices;
    std::vector<std::chrono::microseconds> _measurements;

    std::chrono::microseconds measureCall(int rank);

public:
    BenchmarkRunner(size_t iterations, IElf* elf);
    ~BenchmarkRunner();
    void runBenchmark(const ProblemStatement& statement);
    BenchmarkResult getResults();
};
