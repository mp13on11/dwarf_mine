#pragma once

#include "ElfCategory.h"
#include "Scheduler.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

class Communicator;

class SchedulerFactory
{
public:
    typedef std::function<Scheduler*(const Communicator& communicator)> FactoryFunction;

    static std::vector<std::string> getValidCategories();
    static std::unique_ptr<SchedulerFactory> createFor(const std::string& type, const ElfCategory& category);

    std::unique_ptr<Scheduler> createScheduler(const Communicator& communicator) const;

private:
    SchedulerFactory(const FactoryFunction& factory);
    FactoryFunction factory;
};
