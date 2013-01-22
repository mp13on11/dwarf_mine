#pragma once

#include "common/CommandLineConfiguration.h"

class SimpleConfiguration : public CommandLineConfiguration
{
public:
    SimpleConfiguration(int argc, char** argv);

    virtual std::unique_ptr<SchedulerFactory> createSchedulerFactory() const;
};
