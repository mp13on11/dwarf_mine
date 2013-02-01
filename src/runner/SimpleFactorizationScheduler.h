#pragma once

#include "factorization_montecarlo/FactorizationScheduler.h"

#include <functional>

class SimpleFactorizationScheduler : public FactorizationScheduler
{
public:
    SimpleFactorizationScheduler(const std::function<ElfPointer()>& factory);

protected:
    virtual void doDispatch();
};
