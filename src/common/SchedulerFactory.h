#pragma once

#include "ElfCategory.h"
#include "Scheduler.h"

#include <memory>

class SchedulerFactory
{
public:
    SchedulerFactory(const ElfCategory& category);
    virtual ~SchedulerFactory() = 0;

    std::unique_ptr<Scheduler> createScheduler() const;

protected:
    ElfCategory _category;

    virtual std::unique_ptr<Scheduler> createSchedulerImplementation() const = 0;

private:
    void validate() const;
};

inline SchedulerFactory::~SchedulerFactory()
{
}

std::unique_ptr<SchedulerFactory> createElfFactory(const std::string& type, const ElfCategory& category);
