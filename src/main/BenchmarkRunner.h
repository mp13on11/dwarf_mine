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
    BenchmarkResult _results;

    std::chrono::microseconds measureCall(ProblemStatement& statement, Scheduler& scheduler);
    void benchmarkDevice(DeviceId device, ProblemStatement& statement, Scheduler& scheduler, BenchmarkResult& result);
    void getBenchmarked(ProblemStatement& statement, Scheduler& scheduler);
    void transformToResults(const BenchmarkResult& result);

public:
    explicit BenchmarkRunner(size_t iterations);
    void runBenchmark(ProblemStatement& statement, const ElfFactory& factory);
    BenchmarkResult getResults();
};
