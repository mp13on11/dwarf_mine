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

unique_ptr<ProblemStatement> Configuration::createProblemStatement()
{
    ifstream input;
    auto statement = unique_ptr<ProblemStatement>(new ProblemStatement()));
    input.open(argv[2]);
    
    if(!input.is_open())
    {
        throw runtime_error("Failed to open " + string(argv[2]));
    }

    statement->input.rdbuf(input.rdbuf());
    return statement;
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
