#include "Configuration.h"
#include "CudaElfFactory.h"
#include "SMPElfFactory.h"

#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

Configuration::Configuration(int argc, char** argv)
    : arguments(argv + 1, argv + argc), programName(argv[0])
{
    /*
    if (arguments.size() != 1 && arguments.size() != 3)
    {
        usageError();
    }
    */
}

void Configuration::usageError()
{
    printUsage();
    exit(1);
}

void Configuration::printUsage()
{
    cerr << "Usage: " << programName << " cuda|smp" << " [inputFilename outputFilename] " << endl << "\tInput and outputFilename required for master." << endl;
}

unique_ptr<ProblemStatement> Configuration::createProblemStatement(std::string category)
{

    if(arguments.size() < 2)
    {
        return unique_ptr<ProblemStatement>(new ProblemStatement(category));
    }

    return unique_ptr<ProblemStatement>(new ProblemStatement(category, arguments[1], arguments[2]));
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
