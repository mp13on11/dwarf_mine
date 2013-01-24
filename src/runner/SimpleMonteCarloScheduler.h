#pragma once

#include "montecarlo/MonteCarloScheduler.h"

#include <functional>

class SimpleMonteCarloScheduler : public MonteCarloScheduler
{
public:
    SimpleMonteCarloScheduler(const std::function<ElfPointer()>& factory);

protected:
    virtual void doDispatch();
};
