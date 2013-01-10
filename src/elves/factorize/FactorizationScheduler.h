#pragma once

#include "BigInt.h"
#include "main/Scheduler.h"

class FactorizationScheduler : public Scheduler
{
public:
    FactorizationScheduler();
    explicit FactorizationScheduler(const BenchmarkResult& result);

    virtual void provideData(ProblemStatement& statement);
    virtual void outputData(ProblemStatement& statement);

protected:
    virtual void doDispatch();
    virtual bool hasData();

private:
    BigInt number;
    BigInt a, b;

    void distributeNumber();
    void factorizeNumber();
};
