#pragma once

#include "SchedulerFactory.h"

class CudaElfFactory : public SchedulerFactory
{
public:
    CudaElfFactory(const ElfCategory& category);

protected:
    virtual std::unique_ptr<Scheduler> createSchedulerImplementation() const;
};
