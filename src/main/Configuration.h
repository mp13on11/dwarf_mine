#pragma once

#include <vector>
#include <string>
#include <memory>
#include "ElfFactory.h"
#include "ElfCategory.h"
#include "ProblemStatement.h"

class Configuration
{
public:
    Configuration(int argc, char** argv);
    std::unique_ptr<ProblemStatement> createProblemStatement(std::string category);
    std::unique_ptr<ElfFactory> getElfFactory();
    std::string getElfCategory() const;
    bool parseArguments();
    size_t getNumberOfWarmUps();
    size_t getNumberOfIterations();

private:
    
    void usageError();
    void printUsage();    

    int argc;
    bool _useFiles;
    size_t _numberOfWarmUps;
    size_t _numberOfIterations;
    char** arguments;    
    std::string programName, _mode;
    std::string _inputFile, _outputFile;
};
