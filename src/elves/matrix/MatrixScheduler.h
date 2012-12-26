#pragma once

#include "../main/Scheduler.h"

class MatrixElf;

class MatrixScheduler: public Scheduler
{
public:
    MatrixScheduler() = default;
    explicit MatrixScheduler(const BenchmarkResult& benchmarkResult);
    virtual void doDispatch(ProblemStatement& statement);
};
