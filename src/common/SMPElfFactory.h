#pragma once

#include "ElfFactory.h"

class SMPElfFactory : public ElfFactory
{
public:
    SMPElfFactory(const ElfCategory& category);

protected:
    virtual std::unique_ptr<Scheduler> createSchedulerImplementation() const;
};
