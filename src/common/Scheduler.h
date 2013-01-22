#pragma once

#include "BenchmarkResults.h"
#include "MpiHelper.h"

struct ProblemStatement;

class Scheduler
{
public:
    Scheduler();
    virtual ~Scheduler() = 0;

    void setNodeset(const BenchmarkResult& benchmarkResult);
    void setNodeset(NodeId singleNode);

    void dispatch(ProblemStatement& statement);
    virtual void provideData(ProblemStatement& statement) = 0;
    virtual void dispatch() = 0;
    virtual void outputData(ProblemStatement& statement) = 0;

protected:
    BenchmarkResult nodeSet;
    NodeId rank;
};
