#pragma once

#include "common/ProblemStatement.h"
#include "common/Scheduler.h"

#include <chrono>
#include <memory>
#include <vector>

class Configuration;

class SimpleBenchmarkRunner
{
public:
    typedef std::chrono::microseconds Measurement;

    SimpleBenchmarkRunner(const Configuration& config);
    std::vector<Measurement> run() const;

private:
    std::size_t warmUps;
    std::size_t iterations;
    std::unique_ptr<ProblemStatement> problemStatement;
    std::unique_ptr<Scheduler> scheduler;
    mutable std::chrono::high_resolution_clock::time_point startOfMeasurement;

    void provideData() const;
    void warmUp() const;
    std::vector<Measurement> iterate() const;
    void outputData() const;
    void startMeasurement() const;
    Measurement endMeasurement() const;
};