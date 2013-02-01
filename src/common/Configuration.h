#pragma once

#include <boost/program_options.hpp>
#include <iosfwd>
#include <memory>
#include <string>

class ProblemStatement;
class Scheduler;
class SchedulerFactory;

class Configuration
{
public:
    static void printHelp();

    Configuration(int argc, char** argv);

    std::unique_ptr<Scheduler> createScheduler() const;

    std::unique_ptr<ProblemStatement> createProblemStatement(bool forceGenerated = false) const;
    std::unique_ptr<SchedulerFactory> createSchedulerFactory() const;

    size_t warmUps() const;
    size_t iterations() const;
    bool shouldExportConfiguration() const;
    bool shouldImportConfiguration() const;
    bool shouldSkipBenchmark() const;
    std::string importConfigurationFilename() const;
    std::string exportConfigurationFilename() const;
    bool shouldBeQuiet() const;
    bool shouldBeVerbose() const;
    std::string timeOutputFilename() const;

    void validate() const;
    bool shouldPrintHelp() const;

    friend std::ostream& operator<<(std::ostream& s, const Configuration& c);

protected:
    std::string mode() const;
    std::string category() const;

private:
    static boost::program_options::options_description createDescription();

    boost::program_options::options_description description;
    boost::program_options::variables_map variables;

    std::string inputFilename() const;
    std::string outputFilename() const;
    size_t leftMatrixRows() const;
    size_t commonMatrixRowsColumns() const;
    size_t rightMatrixColumns() const;
    bool useFiles() const;
};
