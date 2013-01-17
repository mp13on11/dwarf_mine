#pragma once

#include <vector>
#include <chrono>
#include <memory>
#include "BenchmarkResults.h"
#include "Configuration.h"
#include "ProblemStatement.h"
#include "SchedulerFactory.h"

class Scheduler;

class BenchmarkRunner
{
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

public:
    explicit BenchmarkRunner(Configuration& config);
    BenchmarkRunner(Configuration& config, const BenchmarkResult& result);
    void runBenchmark(ProblemStatement& statement, const SchedulerFactory& factory);
    BenchmarkResult getWeightedResults();
    BenchmarkResult getTimedResults();
};
