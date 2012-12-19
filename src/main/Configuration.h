#pragma once

#include <vector>
#include <string>
#include <memory>
#include "ElfFactory.h"

class Configuration
{
public:
    Configuration(int argc, char** argv);
    
    std::unique_ptr<ElfFactory> getElfFactory();
    
private:
    void usageError();
    void printUsage();
    
    std::vector<std::string> arguments;
    std::string programName;
};
