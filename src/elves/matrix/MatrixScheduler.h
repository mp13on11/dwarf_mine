#pragma once

#include "../main/Scheduler.h"

class MatrixScheduler: public Scheduler
{
public:
    MatrixScheduler() = default;
    explicit MatrixScheduler(const BenchmarkResult& benchmarkResult);
    virtual void dispatch(ProblemStatement& statement);
};
