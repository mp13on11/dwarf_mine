#pragma once

#include "../main/Scheduler.h"

class MatrixScheduler: public Scheduler
{
public:
    MatrixScheduler();
    explicit MatrixScheduler(const BenchmarkResult& benchmarkResult);
    virtual ~MatrixScheduler();

    virtual bool hasData();
    virtual void provideData(ProblemStatement& statement);
    virtual void outputData(ProblemStatement& statement);

protected:
    virtual void doDispatch();

private:
    struct MatrixSchedulerImpl;
    MatrixSchedulerImpl* pImpl;
};
