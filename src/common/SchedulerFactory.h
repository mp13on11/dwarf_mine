#pragma once

#include "ElfCategory.h"
#include "Scheduler.h"

#include <memory>
#include <string>

class SchedulerFactory
{
public:
    SchedulerFactory(const std::string& type, const ElfCategory& category);
    ~SchedulerFactory();

    std::unique_ptr<Scheduler> createScheduler() const;

private:
    std::string _type;
    ElfCategory _category;

    std::unique_ptr<Scheduler> createCudaScheduler() const;
    std::unique_ptr<Scheduler> createSmpScheduler() const;
};
