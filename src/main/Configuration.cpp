#include "Configuration.h"
#include "CudaElfFactory.h"
#include "SMPElfFactory.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>

#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

using namespace std;

Configuration::Configuration(int argc, char** argv) : 
    argc(argc), _useFiles(false), arguments(argv), 
    programName(argv[0]), _category("matrix")
{

}

bool Configuration::parseArguments()
{
    namespace po = boost::program_options;
    po::options_description desc("Options");

    try
    {
        desc.add_options()
            ("help", "Print help message")
            ("mode,m",               po::value<string>(&_mode)->required(), "Mode (smp|cuda)")
            ("numwarmups,w",         po::value<size_t>(&_numberOfWarmUps)->default_value(50), "Number of warmup rounds")
            ("numiter,n",            po::value<size_t>(&_numberOfIterations)->default_value(100), "Number of benchmark iterations")
            ("input,i",              po::value<string>(&_inputFile), "Input file")
            ("output,o",             po::value<string>(&_outputFile), "Output file")
            ("export_configuration", po::value<string>(&_exportConfigurationFile), "Measure cluster and export configuration")
            ("import_configuration", po::value<string>(&_importConfigurationFile), "Run benchmark with given configuration")
            ("skip_benchmark",       "Skip the benchmark run")
            ("quiet,q",              "Do not output anything")
            ("left_rows",            po::value<size_t>(&_leftMatrixRows)->default_value(500), "Number of left rows to be generated (overridden for benchmark by input file)")
            ("common_rows_columns",  po::value<size_t>(&_commonMatrixRowsColumns)->default_value(500), "Number of left columns / right rows to be generated (overridden for benchmark by input file)")
            ("right_columns",        po::value<size_t>(&_rightMatrixColumns)->default_value(500), "Number of right columns to be generated (overridden for benchmark by input file)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, arguments, desc), vm);
        po::notify(vm);

        if(vm.count("help") || argc == 1)
        {
            cout << "Dwarf Mine Benchmark" << endl << desc << endl;
            return false;
        }

        if(vm.count("input") ^ vm.count("output"))
            throw logic_error("Both input and output are needed, if one is given");

        if(vm.count("input") && vm.count("output"))
           _useFiles = true;

        if(vm.count("mode") && (_mode != "smp" && _mode != "cuda"))
            throw logic_error("Mode must be smp or cuda");

        _skipBenchmark = vm.count("skip_benchmark") > 0;

        _quiet = vm.count("quiet") > 0;

    }
    catch(const po::error& e)
    {
        cerr << desc << endl;
        throw;
    }

    return true;
}

unique_ptr<ProblemStatement> generateProblemStatement(string elfCategory, size_t leftRows, size_t commonRowsColumns, size_t rightColumns)
{
    auto statement = unique_ptr<ProblemStatement>(new ProblemStatement(elfCategory));
    Matrix<float> left(leftRows, commonRowsColumns);
    Matrix<float> right(commonRowsColumns, rightColumns);
    auto distribution = uniform_real_distribution<float> (-100, +100);
    auto engine = mt19937(time(nullptr));
    auto generator = bind(distribution, engine);
    MatrixHelper::fill(left, generator);
    MatrixHelper::fill(right, generator);
    MatrixHelper::writeMatrixTo(*(statement->input), left);
    MatrixHelper::writeMatrixTo(*(statement->input), right);
    return statement;
}

unique_ptr<ProblemStatement> Configuration::getProblemStatement(bool forceGenerated)
{

    if(!_useFiles || forceGenerated)
    {
        return generateProblemStatement(_category, _leftMatrixRows, _commonMatrixRowsColumns, _rightMatrixColumns);
    }
    return unique_ptr<ProblemStatement>(new ProblemStatement(getElfCategory(), _inputFile, _outputFile));
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

string Configuration::getElfCategory() const
{
    return _category;
}

bool Configuration::exportConfiguration() const
{
    return _exportConfigurationFile != "";
}

bool Configuration::importConfiguration() const
{
    return _importConfigurationFile != "";
}

bool Configuration::skipBenchmark() const
{
    return _skipBenchmark;
}

bool Configuration::getQuiet() const
{
    return _quiet;
}

std::string Configuration::getExportConfigurationFilename() const
{
    return _exportConfigurationFile;
}

std::string Configuration::getImportConfigurationFilename() const
{
    return _importConfigurationFile;
}

std::ostream& operator<<(std::ostream& s, const Configuration& c)
{
    s   << "Configuation: "
        << "\n\tMode: "<< c._mode
        << "\n\tWarmUps: " << c._numberOfWarmUps
        << "\n\tIterations: " << c._numberOfIterations;
    if (c._useFiles)
    {
        s   << "\n\tInput: " << c._inputFile
            << "\n\tOutput: " << c._outputFile;
    }
    else
    {
        s   << "\n\tMatrices: ("<<c._leftMatrixRows<<" x "<<c._commonMatrixRowsColumns<<") x ("<<c._commonMatrixRowsColumns<<" x "<<c._rightMatrixColumns<<")";
    }
    return s;
}
