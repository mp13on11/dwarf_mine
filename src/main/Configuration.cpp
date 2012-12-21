#include "Configuration.h"
#include "CudaElfFactory.h"
#include "SMPElfFactory.h"

#include <stdexcept>
#include <cstdlib>
#include <iostream>

using namespace std;

Configuration::Configuration(int argc, char** argv)
    : arguments(argv + 1, argv + argc), programName(argv[0])
{
    if (arguments.size() < 1)
    {
        usageError();
    }
}

void Configuration::usageError()
{
    printUsage();
    exit(1);
}

void Configuration::printUsage()
{
    cerr << "Usage: " << programName << " cuda|smp" << endl;
}

unique_ptr<ElfFactory> Configuration::getElfFactory(const ElfCategory& category)
{
    try
    {
        return createElfFactory(arguments[0], category);
    }
    catch (const std::exception&)
    {
        usageError();
    }

    return nullptr; // just to make compiler happy, usageError() terminates anyway
}
