#pragma once

#include "BigInt.h"
#include "main/SchedulerTemplate.h"

#include <functional>
#include <future>

class FactorizationElf;

class FactorizationScheduler : public SchedulerTemplate<FactorizationElf>
{
public:
    FactorizationScheduler(const std::function<ElfPointer()>& factory);

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
    int distributeFinishedStateRegularly(std::future<BigIntPair>& f) const;
    void sendResultToMaster(int rank, std::future<BigIntPair>& f);
    BigInt broadcast(const BigInt& number) const;
};
