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

    BenchmarkResult benchmarkIndividualNodes();
    std::vector<Measurement> runBenchmark(const BenchmarkResult& nodeWeights);

private:
    static Measurement averageOf(const std::vector<Measurement>& runTimes);
    static BenchmarkResult calculateNodeWeights(const std::vector<Measurement>& averageRunTimes);

    size_t iterations;
    size_t warmUps;
    std::unique_ptr<ProblemStatement> individualProblem;
    std::unique_ptr<ProblemStatement> clusterProblem;
    std::unique_ptr<Scheduler> scheduler;

    std::vector<Measurement> runBenchmark(const BenchmarkResult& nodeWeights, ProblemStatement& problem);
    std::vector<Measurement> benchmarkNodeset(ProblemStatement& problem);
    void getBenchmarked();
    Measurement measureCall();
};
