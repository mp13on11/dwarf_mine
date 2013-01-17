#pragma once

#include "Configuration.h"

#include <boost/program_options.hpp>
#include <iosfwd>
#include <memory>
#include <string>

class CommandLineConfiguration : public Configuration
{
public:
	static void printHelp();

	CommandLineConfiguration(int argc, char** argv);

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

	void validate() const;
	bool shouldPrintHelp() const;

	friend std::ostream& operator<<(std::ostream& s, const CommandLineConfiguration& c);

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
