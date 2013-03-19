#pragma once

#include "BenchmarkResults.h"
#include "Configuration.h"
#include "ProblemStatement.h"

#include <functional>
#include <memory>

class Configuration;
class Profiler;

class BenchmarkRunner
{
public:
    explicit BenchmarkRunner(Configuration& config);

    void benchmarkNode(int node, Profiler& profiler) const;
    void runBenchmark(const BenchmarkResult& nodeWeights, Profiler& profiler) const;
    void runElf(Profiler& profiler) const;

private:
    typedef std::function<void()> BenchmarkMethod;

    Configuration* config;
    size_t iterations;
    size_t warmUps;
    std::unique_ptr<ProblemStatement> fileProblem;
    std::unique_ptr<ProblemStatement> generatedProblem;

    void run(BenchmarkMethod targetMethod, Profiler& profiler) const;
};
