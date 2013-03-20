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

private:
    const Configuration* config;
    size_t iterations;
    size_t warmUps;
    std::unique_ptr<ProblemStatement> fileProblem;
    std::unique_ptr<ProblemStatement> generatedProblem;

    void run(Scheduler& scheduler, Profiler& profiler) const;
};
