#pragma once

#include "BenchmarkResults.h"

struct ProblemStatement;

class Scheduler
{
public:
    Scheduler(const BenchmarkResult& benchmarkResult);
    virtual ~Scheduler();
    virtual void dispatch(ProblemStatement& statement);
};

inline Scheduler::~Scheduler()
{
}
