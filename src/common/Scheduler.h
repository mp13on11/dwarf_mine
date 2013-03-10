#pragma once

#include "BenchmarkResults.h"
#include "Configuration.h"
#include <iosfwd>

struct DataGenerationParameters;

class Scheduler
{
public:
    Scheduler();
    virtual ~Scheduler() = 0;

    void setNodeset(const BenchmarkResult& benchmarkResult);
    void setNodeset(NodeId singleNode);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
    virtual void configureWith(const Configuration& config) {}
#pragma GCC diagnostic pop
    virtual void generateData(const DataGenerationParameters& params) = 0;
    virtual void provideData(std::istream& input) = 0;
    virtual void dispatch() = 0;
    virtual void outputData(std::ostream& output) = 0;
    virtual void dispatchSimple() = 0;
    virtual void dispatchBenchmark(NodeId node) = 0;

protected:

    BenchmarkResult nodeSet;
};
