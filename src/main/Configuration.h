#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iosfwd>
#include "ElfFactory.h"
#include "ElfCategory.h"
#include "ProblemStatement.h"

class Configuration
{
public:
    Configuration(int argc, char** argv);
    std::unique_ptr<ProblemStatement> getProblemStatement(bool forceGenerated = false);
    std::unique_ptr<ElfFactory> getElfFactory();
    std::string getElfCategory() const;
    bool parseArguments(bool showDescription);
    size_t getNumberOfWarmUps();
    size_t getNumberOfIterations();
    bool exportConfiguration() const;
    bool importConfiguration() const;
    bool skipBenchmark() const;
    std::string getImportConfigurationFilename() const;
    std::string getExportConfigurationFilename() const;
    bool getQuiet() const;
    bool getVerbose() const;


    friend std::ostream& operator<<(std::ostream& s, const Configuration& c);

private:

    void usageError();
    void printUsage();

    int argc;
    bool _useFiles;
    char** arguments;
    std::string programName;

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
