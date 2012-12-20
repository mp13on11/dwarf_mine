#pragma once

#include "main/ElfFactory.h"

class SMPElfFactory : public ElfFactory
{
public:
    SMPElfFactory(const ElfCategory& category);
protected:
    virtual std::unique_ptr<Elf> createElfImplementation() const;
    virtual std::unique_ptr<Scheduler> createSchedulerImplementation() const;
};
