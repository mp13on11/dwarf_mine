#pragma once

#include <chrono>
#include <memory>
#include <vector>
#include "BenchmarkResults.h"
#include "Configuration.h"
#include "ProblemStatement.h"
#include "SchedulerFactory.h"

class Scheduler;

class BenchmarkRunner
{
public:
    explicit BenchmarkRunner(const Configuration& config);
    BenchmarkRunner(const Configuration& config, const BenchmarkResult& result);
    void runBenchmark(ProblemStatement& statement, const SchedulerFactory& factory);
    BenchmarkResult getWeightedResults();
    BenchmarkResult getTimedResults();

private:
    size_t _iterations;
    size_t _warmUps;
    std::vector<BenchmarkResult> _nodesets;
    BenchmarkResult _weightedResults;
    BenchmarkResult _timedResults;

    std::chrono::microseconds measureCall(Scheduler& scheduler);
    unsigned int benchmarkNodeset(ProblemStatement& statement, Scheduler& scheduler);
    void getBenchmarked(Scheduler& scheduler);
    void weightTimedResults();
};
