#pragma once

#include <vector>
#include <chrono>
#include <memory>
#include "ElfFactory.h"
#include "BenchmarkResults.h"
#include "ProblemStatement.h"

class Scheduler;

class BenchmarkRunner
{
private:
    size_t _iterations;
    std::vector<BenchmarkResult> _nodesets;
    BenchmarkResult _weightedResults;
    BenchmarkResult _timedResults;

    std::chrono::microseconds measureCall(ProblemStatement& statement, Scheduler& scheduler);
    unsigned int benchmarkNodeset(ProblemStatement& statement, Scheduler& scheduler);
    void getBenchmarked(ProblemStatement& statement, Scheduler& scheduler);
    void weightTimedResults();

public:
    explicit BenchmarkRunner(size_t iterations);
    BenchmarkRunner(size_t iterations, BenchmarkResult result);
    BenchmarkRunner(size_t iterations, NodeId device);
    void runBenchmark(ProblemStatement& statement, const ElfFactory& factory);
    BenchmarkResult getWeightedResults();
    BenchmarkResult getTimedResults();
};
