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
    typedef std::chrono::microseconds Measurement;

    explicit BenchmarkRunner(const Configuration& config);

    BenchmarkResult benchmarkIndividualNodes() const;
    std::vector<Measurement> runBenchmark(const BenchmarkResult& nodeWeights) const;

private:
    static Measurement averageOf(const std::vector<Measurement>& runTimes);
    static BenchmarkResult calculateNodeWeights(const std::vector<Measurement>& averageRunTimes);
    static bool slaveShouldRunWith(const BenchmarkResult& nodeWeights);

    size_t iterations;
    size_t warmUps;
    std::unique_ptr<ProblemStatement> individualProblem;
    std::unique_ptr<ProblemStatement> clusterProblem;
    std::unique_ptr<Scheduler> scheduler;

    std::vector<Measurement> runBenchmark(const BenchmarkResult& nodeWeights, ProblemStatement& problem) const;
    std::vector<Measurement> benchmarkNodeset(ProblemStatement& problem) const;
    void getBenchmarked() const;
    Measurement measureCall() const;
};
