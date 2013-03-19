#pragma once

#include "BenchmarkResults.h"
#include "Configuration.h"
#include "ProblemStatement.h"

#include <functional>
#include <memory>

class Communicator;
class Configuration;
class Profiler;

class BenchmarkRunner
{
public:
    explicit BenchmarkRunner(Configuration& config);

    void benchmarkNode(const Communicator& communicator, Profiler& profiler) const;
    void runBenchmark(const Communicator& communicator, Profiler& profiler) const;
    void runElf(const Communicator& communicator, Profiler& profiler) const;

private:
    typedef std::function<void()> BenchmarkMethod;

    Configuration* config;
    size_t iterations;
    size_t warmUps;
    std::unique_ptr<ProblemStatement> fileProblem;
    std::unique_ptr<ProblemStatement> generatedProblem;

    void run(BenchmarkMethod targetMethod, Profiler& profiler) const;
};
