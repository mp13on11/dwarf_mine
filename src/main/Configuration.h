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
    std::unique_ptr<ProblemStatement> createProblemStatement(std::string category);
    std::unique_ptr<ElfFactory> getElfFactory();
    std::string getElfCategory() const;
    bool parseArguments();
    size_t getNumberOfWarmUps();
    size_t getNumberOfIterations();
    bool preBenchmark() const;
    
    friend std::ostream& operator<<(std::ostream& s, const Configuration& c)
    {
		s 	<< "Configuation: "
			<< "\n\tMode: "<< c._mode
			<< "\n\tWarmUps: " << c._numberOfWarmUps
			<< "\n\tIterations: " << c._numberOfIterations
			<< "\n\tInput: " << c._inputFile
			<< "\n\tOutput: " << c._outputFile
			<< "\n\tPre-Benchmark: " << c._preBenchmark;
		return s;
	}

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
    bool _preBenchmark;
};
