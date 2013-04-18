#pragma once

#include "ProblemStatement.h"

#include <memory>

class Communicator;
class Configuration;
class Profiler;
class Scheduler;

class BenchmarkRunner
{
public:
    explicit BenchmarkRunner(const Configuration& config);

    void runBenchmark(const Communicator& communicator, Profiler& profiler) const;
    void runPreBenchmark(const Communicator& communicator, Profiler& profiler) const;

private:
    const Configuration* config;
    size_t iterations;
    size_t warmUps;
    std::unique_ptr<ProblemStatement> fileProblem;
    std::unique_ptr<ProblemStatement> generatedProblem;

    void runBenchmarkInternal(
        const Communicator& communicator, 
        Profiler& profiler,
        const std::unique_ptr<ProblemStatement>& problem
    ) const;
    void run(Scheduler& scheduler, Profiler& profiler) const;
};
