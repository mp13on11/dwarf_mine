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

unique_ptr<ElfFactory> Configuration::getElfFactory()
{
    if (arguments[0] == "smp")
        return unique_ptr<ElfFactory>(new SMPElfFactory());
    else if (arguments[0] == "cuda")
        return unique_ptr<ElfFactory>(new CudaElfFactory());
    else
        usageError();

    return nullptr; // just to make compiler happy, usageError() terminates anyway
}
