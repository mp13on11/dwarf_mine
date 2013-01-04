#include "Configuration.h"
#include "CudaElfFactory.h"
#include "SMPElfFactory.h"

#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp> 

using namespace std;

Configuration::Configuration(int argc, char** argv)
    : argc(argc), _useFiles(false), arguments(argv), programName(argv[0]) 
{

}

bool Configuration::parseArguments()
{ 

    try
    {           
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()    
            ("help", "Print help message")
            ("mode,m", po::value<string>(&_mode), "Mode (smp|cuda)")
            ("numwarmups,w", po::value<size_t>(&_numberOfWarmUps)->default_value(50), "Number of warmup rounds")
            ("numiter,n", po::value<size_t>(&_numberOfIterations)->default_value(100), "Number of benchmark iterations")
            ("input,i", po::value<string>(&_inputFile), "Input file")
            ("output,o", po::value<string>(&_outputFile), "Output file")
            ("prebenchmark,p",po::value<bool>(&_preBenchmark)->default_value(true), "Use prebenchmark - without uniform distribution is enforced");
        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, arguments, desc), vm);            

            if(vm.count("help") || argc == 1) 
            { 
                cout << "Dwarf Mine Benchmark" << endl << desc << endl;  
            }

            po::notify(vm);
        }
        
        catch(po::error& e)
        {
            cerr << "ERROR: " << e.what() << endl << endl; 
            cerr << desc << endl; 
            return false; 
        }
    
        if(vm.count("input") && vm.count("output"))
        {
           _useFiles = true;
        }
        
        if(vm.count("mode") && (_mode != "smp" && _mode != "cuda")){
            cerr << "ERROR: Mode must be smp or cuda" << endl; 
            return false;
        }

    }

    catch(exception& e)
    {
        cerr << "Unhandled Exception in configuration" << endl;
        return false;
    }

    return true;
}

bool Configuration::preBenchmark() const
{
	return _preBenchmark;
}

unique_ptr<ProblemStatement> Configuration::createProblemStatement(std::string category)
{

    if(!_useFiles)
    {
        return unique_ptr<ProblemStatement>(new ProblemStatement(category));
    } 
    
    return unique_ptr<ProblemStatement>(new ProblemStatement(category, _inputFile, _outputFile));
}

unique_ptr<ElfFactory> Configuration::getElfFactory()
{
    return createElfFactory(_mode, getElfCategory());
}

size_t Configuration::getNumberOfIterations()
{
    return _numberOfIterations;
}

size_t Configuration::getNumberOfWarmUps()
{
    return _numberOfWarmUps;
}

// TODO remove
string Configuration::getElfCategory() const
{
    return "matrix";
}
