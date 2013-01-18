#pragma once

#include "main/SchedulerTemplate.h"
#include "MonteCarloElf.h"
#include <functional>

class MonteCarloElf;

class MonteCarloScheduler: public SchedulerTemplate<MonteCarloElf>
{
public:
    MonteCarloScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MonteCarloScheduler();

    virtual void provideData(ProblemStatement& statement);
    virtual void outputData(ProblemStatement& statement);

protected:
    virtual bool hasData();
    virtual void doDispatch();

private:
    struct MonteCarloSchedulerImpl;
    MonteCarloSchedulerImpl* pImpl;
};
