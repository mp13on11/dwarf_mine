#pragma once

#include "BenchmarkResults.h"

struct ProblemStatement;

class Scheduler
{
public:
    Scheduler();
    virtual ~Scheduler() = 0;

    void setNodeset(const BenchmarkResult& benchmarkResult);
    void setNodeset(NodeId singleNode);

    virtual void provideData(ProblemStatement& statement) = 0;
    virtual void dispatch() = 0;
    virtual void outputData(ProblemStatement& statement) = 0;
    virtual void dispatchSimple() = 0;

protected:
    BenchmarkResult nodeSet;
};
