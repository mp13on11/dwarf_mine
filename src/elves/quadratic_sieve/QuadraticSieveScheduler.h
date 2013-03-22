#pragma once

#include "common/SchedulerTemplate.h"
#include "common-factorization/BigInt.h"
#include <utility>
#include "QuadraticSieve.h"

class QuadraticSieveElf;

class QuadraticSieveScheduler : public SchedulerTemplate<QuadraticSieveElf>
{
public:
    QuadraticSieveScheduler(const Communicator& communicator, const std::function<ElfPointer()>& factory);

    virtual void provideData(std::istream& input);
    virtual void outputData(std::ostream& output);
    virtual void generateData(const DataGenerationParameters& params);

    std::pair<BigInt, BigInt> factor();


    std::vector<double> determineChunkSizes(const BigInt& start, const BigInt& end);

private:
    std::vector<BigInt> sieveDistributed(
        const BigInt& start,
        const BigInt& end,
        const BigInt& number,
        const FactorBase& factorBase
    );
    virtual void doDispatch();
    virtual void doSimpleDispatch();
    virtual bool hasData() const;

    BigInt number;
    BigInt p;
    BigInt q;
};
