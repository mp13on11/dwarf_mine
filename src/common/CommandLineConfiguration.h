#pragma once

#include "Configuration.h"

#include <iosfwd>
#include <memory>
#include <string>

class CommandLineConfiguration : public Configuration
{
public:
	CommandLineConfiguration(int argc, char** argv, bool showDescriptionOnError);

	virtual std::unique_ptr<ProblemStatement> createProblemStatement(bool forceGenerated = false) const;
	virtual std::unique_ptr<SchedulerFactory> createSchedulerFactory() const;

	virtual size_t warmUps() const;
	virtual size_t iterations() const;
	virtual bool shouldExportConfiguration() const;
	virtual bool shouldImportConfiguration() const;
	virtual bool shouldSkipBenchmark() const;
	virtual std::string importConfigurationFilename() const;
	virtual std::string exportConfigurationFilename() const;
	virtual bool shouldBeQuiet() const;
	virtual bool shouldBeVerbose() const;


    friend std::ostream& operator<<(std::ostream& s, const CommandLineConfiguration& c);

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
