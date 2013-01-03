#pragma once

#include "BenchmarkResults.h"

struct ProblemStatement;
class Elf;

class Scheduler
{
public:
    Scheduler();
    explicit Scheduler(const BenchmarkResult& benchmarkResult);
    virtual ~Scheduler();

    virtual void provideData(ProblemStatement& statement) = 0;
    virtual void outputData(ProblemStatement& statement) = 0;
    void dispatch();
    void dispatch(ProblemStatement& statement);
    void setNodeset(const BenchmarkResult& benchmarkResult);
    void setNodeset(NodeId singleNode);

    Elf* getElf() const { return elf; }
    void setElf(Elf* val) { elf = val; }

protected:
    virtual void doDispatch() = 0;
    virtual bool hasData() = 0;

    bool nodesHaveRatings;
    BenchmarkResult nodeSet;
    int rank;
    Elf* elf;
};

inline Scheduler::~Scheduler()
{
}
