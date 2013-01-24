#pragma once

#include "BenchmarkResults.h"
#include "ProblemStatement.h"
#include "Scheduler.h"

#include <chrono>
#include <memory>
#include <vector>

class Configuration;

class BenchmarkRunner
{
public:
    explicit BenchmarkRunner(const Configuration& config);
    BenchmarkRunner(const Configuration& config, const BenchmarkResult& result);
    void runBenchmark();
    BenchmarkResult getWeightedResults() const;
    BenchmarkResult getTimedResults() const;

private:
    size_t _iterations;
    size_t _warmUps;
    std::unique_ptr<ProblemStatement> problemStatement;
    std::unique_ptr<Scheduler> scheduler;
    std::vector<BenchmarkResult> _nodesets;
    BenchmarkResult _weightedResults;
    BenchmarkResult _timedResults;

    unsigned int benchmarkNodeset();
    void getBenchmarked();
    std::chrono::microseconds measureCall(Scheduler& scheduler);
    void weightTimedResults();
};

inline BenchmarkResult BenchmarkRunner::getWeightedResults() const
{
    return _weightedResults;
}

inline BenchmarkResult BenchmarkRunner::getTimedResults() const
{
    return _timedResults;
}
