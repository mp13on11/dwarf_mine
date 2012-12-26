#pragma once

#include "../main/Scheduler.h"

class MatrixScheduler: public Scheduler
{
public:
    MatrixScheduler() = default;
    explicit MatrixScheduler(const BenchmarkResult& benchmarkResult);
    virtual ~MatrixScheduler();

    virtual void doDispatch(ProblemStatement& statement);

private:
    struct MatrixSchedulerImpl;
    MatrixSchedulerImpl* pImpl;
};
