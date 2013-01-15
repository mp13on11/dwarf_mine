#pragma once

#include "main/ElfFactory.h"

class CudaElfFactory : public ElfFactory
{
public:
    CudaElfFactory(const ElfCategory& category);

protected:
    virtual std::unique_ptr<Scheduler> createSchedulerImplementation() const;
};
