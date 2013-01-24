#pragma once

#include <memory>
#include <string>

class ProblemStatement;
class Scheduler;
class SchedulerFactory;

class Configuration
{
public:
    virtual ~Configuration() = 0;

    std::unique_ptr<Scheduler> createScheduler() const;

    virtual std::unique_ptr<ProblemStatement> createProblemStatement(bool forceGenerated = false) const = 0;
    virtual std::unique_ptr<SchedulerFactory> createSchedulerFactory() const = 0;

    virtual size_t warmUps() const = 0;
    virtual size_t iterations() const = 0;
    virtual bool shouldExportConfiguration() const = 0;
    virtual bool shouldImportConfiguration() const = 0;
    virtual bool shouldSkipBenchmark() const = 0;
    virtual std::string importConfigurationFilename() const = 0;
    virtual std::string exportConfigurationFilename() const = 0;
    virtual bool shouldBeQuiet() const = 0;
    virtual bool shouldBeVerbose() const = 0;
};

inline Configuration::~Configuration()
{
}
