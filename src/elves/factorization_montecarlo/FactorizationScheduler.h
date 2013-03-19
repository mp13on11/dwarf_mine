#pragma once

#include "common-factorization/BigInt.h"
#include "common/SchedulerTemplate.h"

#include <functional>
#include <future>

class MonteCarloFactorizationElf;

class FactorizationScheduler : public SchedulerTemplate<MonteCarloFactorizationElf>
{
public:
    FactorizationScheduler(const Communicator& communicator, const std::function<ElfPointer()>& factory);

    virtual void provideData(std::istream& input);
    virtual void outputData(std::ostream& output);
    virtual void generateData(const DataGenerationParameters& params);

protected:
    typedef std::pair<BigInt, BigInt> BigIntPair;

    BigInt number;
    BigInt p, q;

    virtual void doDispatch();
    virtual void doSimpleDispatch();
    virtual void doBenchmarkDispatch(int node);
    virtual bool hasData() const;

private:
    void distributeNumber();
    int distributeFinishedStateRegularly(std::future<BigIntPair>& f) const;
    void sendResultToMaster(int rank, std::future<BigIntPair>& f);
    BigInt broadcast(const BigInt& number) const;
};
