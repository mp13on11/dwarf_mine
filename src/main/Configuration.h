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
    std::unique_ptr<ProblemStatement> createProblemStatement();
    std::unique_ptr<ElfFactory> getElfFactory(const ElfCategory& category);

private:
    void usageError();
    void printUsage();

    std::vector<std::string> arguments;
    std::string programName;
};
