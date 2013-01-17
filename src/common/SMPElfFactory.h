#pragma once

#include "SchedulerFactory.h"

class SMPElfFactory : public SchedulerFactory
{
public:
    SMPElfFactory(const ElfCategory& category);

protected:
    virtual std::unique_ptr<Scheduler> createSchedulerImplementation() const;
};
