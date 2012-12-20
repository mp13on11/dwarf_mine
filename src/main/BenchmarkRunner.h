#pragma once

#include <vector>
#include <chrono>
#include <memory>
#include "ElfFactory.h"
#include "BenchmarkResults.h"
#include "ProblemStatement.h"

class Scheduler;

class BenchmarkRunner
{
private:
    size_t _iterations;
    int _devices;
    BenchmarkResult m_results;

    std::chrono::microseconds measureCall(ProblemStatement& statement, std::shared_ptr<Scheduler> scheduler);
    void benchmarkDevice(DeviceId device, ProblemStatement& statement, std::shared_ptr<Scheduler> scheduler);
    void getBenchmarked(ProblemStatement& statement, std::shared_ptr<Scheduler> scheduler);

public:
    explicit BenchmarkRunner(size_t iterations);
    void runBenchmark(ProblemStatement& statement, const ElfFactory& factory);
    BenchmarkResult getResults();
};
