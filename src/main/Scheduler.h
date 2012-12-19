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
    virtual void dispatch(ProblemStatement& statement) = 0; // to be executed on master rank
    void setNodeset(const BenchmarkResult& benchmarkResult);
    void setNodeset(NodeId singleNode);
    
    Elf* getElf() const { return elf; }
    void setElf(Elf* val) { elf = val; }

protected:
    bool nodesHaveRatings;
    BenchmarkResult nodeSet;
    int rank;
    Elf* elf;    
};

inline Scheduler::~Scheduler()
{
}
