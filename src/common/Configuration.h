#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iosfwd>
#include "ElfCategory.h"
#include "ProblemStatement.h"
#include "SchedulerFactory.h"

class Configuration
{
public:
    Configuration(int argc, char** argv, bool showDescriptionOnError);

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


    friend std::ostream& operator<<(std::ostream& s, const Configuration& c);

private:

    void usageError();
    void printUsage();

    bool _useFiles;
    bool _skipBenchmark;
    bool _quiet;
    bool _verbose;
    size_t _numberOfWarmUps;
    size_t _numberOfIterations;
    std::string _mode;
    std::string _category;
    std::string _inputFile;
    std::string _outputFile;
    std::string _exportConfigurationFile;
    std::string _importConfigurationFile;
    size_t _leftMatrixRows;
    size_t _commonMatrixRowsColumns;
    size_t _rightMatrixColumns;

};
