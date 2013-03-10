#pragma once

#include <boost/program_options.hpp>
#include <iosfwd>
#include <memory>
#include <string>

class ProblemStatement;
class Scheduler;
class SchedulerFactory;
struct DataGenerationParameters;

class Configuration
{
public:
    static void printHelp();

    Configuration(int argc, char** argv);

    std::unique_ptr<Scheduler> createScheduler() const;

    std::unique_ptr<ProblemStatement> createProblemStatement() const;
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
    std::string schedulingStrategy() const;

    void validate() const;
    bool shouldPrintHelp() const;

    friend std::ostream& operator<<(std::ostream& s, const Configuration& c);

private:
    std::string mode() const;
    std::string category() const;

    DataGenerationParameters makeDataGenerationParameters() const;

    static boost::program_options::options_description createDescription();

    boost::program_options::options_description description;
    boost::program_options::variables_map variables;

    std::string inputFilename() const;
    std::string outputFilename() const;
    size_t leftMatrixRows() const;
    size_t commonMatrixRowsColumns() const;
    size_t rightMatrixColumns() const;
    size_t leftDigits() const;
    size_t rightDigits() const;
    size_t monteCarloTrials() const;
    bool useFiles() const;
};
