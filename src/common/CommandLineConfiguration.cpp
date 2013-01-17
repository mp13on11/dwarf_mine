#include "CommandLineConfiguration.h"
#include "ProblemStatement.h"
#include "SchedulerFactory.h"
#include "matrix/Matrix.h"
#include "matrix/MatrixHelper.h"

#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

using namespace std;

CommandLineConfiguration::CommandLineConfiguration(int argc, char** argv, bool showDescriptionOnError) :
    _useFiles(false)
{
    namespace po = boost::program_options;
    po::options_description desc("Options");

    try
    {
        desc.add_options()
            ("help", "Print help message")
            ("mode,m",               po::value<string>(&_mode)->required(), "Mode (smp|cuda)")
            ("category,c",           po::value<string>(&_category)->default_value("matrix"), "Elf to be run (matrix|factorize)")
            ("numwarmups,w",         po::value<size_t>(&_numberOfWarmUps)->default_value(50), "Number of warmup rounds")
            ("numiter,n",            po::value<size_t>(&_numberOfIterations)->default_value(100), "Number of benchmark iterations")
            ("input,i",              po::value<string>(&_inputFile), "Input file")
            ("output,o",             po::value<string>(&_outputFile), "Output file")
            ("export_configuration", po::value<string>(&_exportConfigurationFile), "Measure cluster and export configuration")
            ("import_configuration", po::value<string>(&_importConfigurationFile), "Run benchmark with given configuration")
            ("skip_benchmark",       "Skip the benchmark run")
            ("quiet,q",              "Do not output anything")
            ("verbose,v",            "Show output from all MPI processes")
            ("left_rows",            po::value<size_t>(&_leftMatrixRows)->default_value(500), "Number of left rows to be generated (overridden for benchmark by input file)")
            ("common_rows_columns",  po::value<size_t>(&_commonMatrixRowsColumns)->default_value(500), "Number of left columns / right rows to be generated (overridden for benchmark by input file)")
            ("right_columns",        po::value<size_t>(&_rightMatrixColumns)->default_value(500), "Number of right columns to be generated (overridden for benchmark by input file)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if(vm.count("help") || argc == 1)
        {
            cout << "Dwarf Mine Benchmark" << endl << desc << endl;
        }

        if(vm.count("input") ^ vm.count("output"))
            throw logic_error("Both input and output are needed, if one is given");

        if(vm.count("input") && vm.count("output"))
           _useFiles = true;

        if(vm.count("mode") && (_mode != "smp" && _mode != "cuda"))
            throw logic_error("Mode must be smp or cuda");

        _skipBenchmark = vm.count("skip_benchmark") > 0;

        _quiet = vm.count("quiet") > 0;

        _verbose = vm.count("verbose") > 0;

    }
    catch(const po::error& e)
    {
        if (showDescriptionOnError)
            cerr << desc << endl;
        throw;
    }
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

unique_ptr<ProblemStatement> CommandLineConfiguration::createProblemStatement(bool forceGenerated) const
{
    if(!_useFiles || forceGenerated)
    {
        return generateProblemStatement(_category, _leftMatrixRows, _commonMatrixRowsColumns, _rightMatrixColumns);
    }
    return unique_ptr<ProblemStatement>(new ProblemStatement(_category, _inputFile, _outputFile));
}

unique_ptr<SchedulerFactory> CommandLineConfiguration::createSchedulerFactory() const
{
    return unique_ptr<SchedulerFactory>(new SchedulerFactory(_mode, _category));
}

size_t CommandLineConfiguration::iterations() const
{
    return _numberOfIterations;
}

size_t CommandLineConfiguration::warmUps() const
{
    return _numberOfWarmUps;
}

bool CommandLineConfiguration::shouldExportConfiguration() const
{
    return _exportConfigurationFile != "";
}

bool CommandLineConfiguration::shouldImportConfiguration() const
{
    return _importConfigurationFile != "";
}

bool CommandLineConfiguration::shouldSkipBenchmark() const
{
    return _skipBenchmark;
}

bool CommandLineConfiguration::shouldBeQuiet() const
{
    return _quiet;
}
bool CommandLineConfiguration::shouldBeVerbose() const
{
    return _verbose;
}

std::string CommandLineConfiguration::exportConfigurationFilename() const
{
    return _exportConfigurationFile;
}

std::string CommandLineConfiguration::importConfigurationFilename() const
{
    return _importConfigurationFile;
}

std::ostream& operator<<(std::ostream& s, const CommandLineConfiguration& c)
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
