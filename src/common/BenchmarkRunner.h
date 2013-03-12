#pragma once

#include "BenchmarkResults.h"
#include "ProblemStatement.h"
#include "Scheduler.h"
#include "Configuration.h"

#include <chrono>
#include <memory>
#include <vector>
#include <functional>

class Configuration;

class BenchmarkRunner
{
public:
    typedef std::chrono::microseconds Measurement;

    explicit BenchmarkRunner(Configuration& config);

    BenchmarkResult benchmarkIndividualNodes() const;
    std::vector<Measurement> runBenchmark(const BenchmarkResult& nodeWeights) const;

private:
    typedef std::function<void()> BenchmarkMethod;
    Configuration* config;    

    static Measurement averageOf(const std::vector<Measurement>& runTimes);
    static BenchmarkResult calculateNodeWeights(const std::vector<Measurement>& averageRunTimes);
    static bool slaveShouldRunWith(const BenchmarkResult& nodeWeights);

    size_t iterations;
    size_t warmUps;
    std::unique_ptr<ProblemStatement> fileProblem;
    std::unique_ptr<ProblemStatement> generatedProblem;
    std::unique_ptr<Scheduler> scheduler;

    std::vector<Measurement> benchmarkNodeset(const ProblemStatement& problem, BenchmarkMethod targetMethod) const;
    void benchmarkSlave(BenchmarkMethod targetMethod) const;
    Measurement measureCall(BenchmarkMethod targetMethod) const;
};
