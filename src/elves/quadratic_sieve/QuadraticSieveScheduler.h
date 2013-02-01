#pragma once

#include "common/SchedulerTemplate.h"

class QuadraticSieveElf;

class QuadraticSieveScheduler : public SchedulerTemplate<QuadraticSieveElf>
{
public:
    QuadraticSieveScheduler(const std::function<ElfPointer()>& factory);

    virtual void provideData(ProblemStatement& statement);
    virtual void outputData(ProblemStatement& statement);

protected:
    virtual void doDispatch();
    virtual void doSimpleDispatch();
    virtual bool hasData() const;
};
