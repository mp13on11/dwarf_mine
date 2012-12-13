#pragma once

#include <vector>
#include <chrono>
#include "ElfFactory.h"
#include "BenchmarkResults.h"
#include "ProblemStatement.h"


class BenchmarkRunner
{
private:
    size_t _iterations;
    int _rank;
    int _devices;
    std::vector<std::chrono::microseconds> _measurements;

    std::chrono::microseconds measureCall(int rank, Elf& elf, const ProblemStatement& statement);

public:
    BenchmarkRunner(size_t iterations);
    ~BenchmarkRunner();
    void runBenchmark(const ProblemStatement& statement, const ElfFactory& factory);
    BenchmarkResult getResults();
};
