#pragma once

#include "common/SchedulerTemplate.h"
#include "common-factorization/BigInt.h"

class QuadraticSieveElf;

class QuadraticSieveScheduler : public SchedulerTemplate<QuadraticSieveElf>
{
public:
    QuadraticSieveScheduler(const std::function<ElfPointer()>& factory);

    virtual void provideData(ProblemStatement& statement);
    virtual void outputData(ProblemStatement& statement);

private:
    virtual void doDispatch();
    virtual void doSimpleDispatch();
    virtual bool hasData() const;

    BigInt number;
    BigInt p;
    BigInt q;
};
