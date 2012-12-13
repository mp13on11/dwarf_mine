#pragma once

#include <vector>
#include <chrono>
#include "BenchmarkResults.h"
#include "ProblemStatement.h"
#include "../elves/IElf.h"

typedef void* Elf;

class BenchmarkRunner
{
private:
    size_t _iterations;
    std::vector<std::chrono::microseconds> _measurements;
    IElf* _elf;

    std::chrono::microseconds measureCall(int rank);

public:
    BenchmarkRunner(size_t iterations, IElf* elf);
    ~BenchmarkRunner();
    void runBenchmark();
    BenchmarkResult getResults();
};
