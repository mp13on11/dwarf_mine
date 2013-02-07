#pragma once

#include "common/SchedulerTemplate.h"
#include "common-factorization/BigInt.h"

class QuadraticSieveElf;

class QuadraticSieveScheduler : public SchedulerTemplate<QuadraticSieveElf>
{
public:
    QuadraticSieveScheduler(const std::function<ElfPointer()>& factory);

    virtual void provideData(std::istream& input);
    virtual void outputData(std::ostream& output);
    virtual void generateData(const DataGenerationParameters& params);

private:
    virtual void doDispatch();
    virtual void doSimpleDispatch();
    virtual bool hasData() const;

    BigInt number;
    BigInt p;
    BigInt q;
};
