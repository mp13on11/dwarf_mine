#pragma once

#include "BenchmarkResults.h"
#include "Configuration.h"
#include "ProblemStatement.h"
#include "Scheduler.h"

#include <functional>
#include <memory>

class Configuration;

class BenchmarkRunner
{
public:
    explicit BenchmarkRunner(Configuration& config);

    void benchmarkIndividualNodes() const;
    void runBenchmark(const BenchmarkResult& nodeWeights) const;
    void runElf() const;

private:
    typedef std::function<void()> BenchmarkMethod;

    Configuration* config;
    size_t iterations;
    size_t warmUps;
    std::unique_ptr<ProblemStatement> fileProblem;
    std::unique_ptr<ProblemStatement> generatedProblem;
    std::unique_ptr<Scheduler> scheduler;

    void run(BenchmarkMethod targetMethod) const;
    void initializeMaster(const ProblemStatement& problem, const BenchmarkResult& nodeWeights = {{0, 1}}) const;
    void finalizeMaster(const ProblemStatement& problem) const;
};
