#pragma once

#include "common-factorization/BigInt.h"
#include "common/SchedulerTemplate.h"

#include <functional>
#include <future>

class MonteCarloFactorizationElf;

class FactorizationScheduler : public SchedulerTemplate<MonteCarloFactorizationElf>
{
public:
    FactorizationScheduler(const std::function<ElfPointer()>& factory);

    virtual void provideData(ProblemStatement& statement);
    virtual void outputData(ProblemStatement& statement);

protected:
    typedef std::pair<BigInt, BigInt> BigIntPair;

    BigInt number;
    BigInt p, q;

    virtual void doDispatch();
    virtual bool hasData() const;

private:
    void distributeNumber();
    int distributeFinishedStateRegularly(std::future<BigIntPair>& f) const;
    void sendResultToMaster(int rank, std::future<BigIntPair>& f);
    BigInt broadcast(const BigInt& number) const;
};
