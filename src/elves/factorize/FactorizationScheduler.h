#pragma once

#include "BigInt.h"
#include "main/Scheduler.h"

#include <future>

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
    typedef std::pair<BigInt, BigInt> BigIntPair;

    BigInt number;
    BigInt a, b;

    void distributeNumber();
    BigIntPair factorizeNumber();
    int distributeFinishedStateRegularly(std::future<BigIntPair>& f) const;
    void sendResultToMaster(int rank, std::future<BigIntPair>& f);
    void stopFactorization();
    BigInt broadcast(const BigInt& number) const;
};
