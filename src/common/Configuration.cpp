#include "Configuration.h"
#include "ProblemStatement.h"
#include "SchedulerFactory.h"
#include "DataGenerationParameters.h"
#include "matrix/Matrix.h"
#include "matrix/MatrixHelper.h"
#include "matrix/MatrixOnlineSchedulingStrategyFactory.h"

#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace boost::program_options;

Configuration::Configuration(int argc, char** argv) :
    description(createDescription()), variables()
{
    store(parse_command_line(argc, argv, description), variables);
    notify(variables);

    if (mode() != "smp" && mode() != "cuda")
        throw error("Mode must be smp or cuda");
    if ((variables.count("input") > 0) ^ (variables.count("output") > 0))
        throw error("Both input and output are needed, if one is given");
}

unique_ptr<ProblemStatement> Configuration::createProblemStatement() const
{
    if(!useFiles())
        return createGeneratedProblemStatement();

    return unique_ptr<ProblemStatement>(
        new ProblemStatement(category(), inputFilename(), outputFilename())
    );
}

unique_ptr<ProblemStatement> Configuration::createGeneratedProblemStatement() const
{
    return unique_ptr<ProblemStatement>(
        new ProblemStatement(category(), makeDataGenerationParameters())
    );
}

unique_ptr<SchedulerFactory> Configuration::createSchedulerFactory() const
{
    return SchedulerFactory::createFor(mode(), category());
}

DataGenerationParameters Configuration::makeDataGenerationParameters() const
{
    return
        {
            leftMatrixRows(),
            commonMatrixRowsColumns(),
            rightMatrixColumns(),
            schedulingStrategy(),
            leftDigits(),
            rightDigits(),
            monteCarloTrials()
        };
}

size_t Configuration::iterations() const
{
    return variables["numiter"].as<size_t>();
}

size_t Configuration::leftDigits() const
{
    return variables["left_digits"].as<size_t>();
}

size_t Configuration::rightDigits() const
{
    return variables["right_digits"].as<size_t>();
}

size_t Configuration::warmUps() const
{
    return variables["numwarmups"].as<size_t>();
}

bool Configuration::shouldExportConfiguration() const
{
    return exportConfigurationFilename() != "";
}

bool Configuration::shouldImportConfiguration() const
{
    return importConfigurationFilename() != "";
}

bool Configuration::shouldSkipBenchmark() const
{
    return variables.count("skip_benchmark") > 0;
}

bool Configuration::shouldBeQuiet() const
{
    return variables.count("quiet") > 0;
}
bool Configuration::shouldBeVerbose() const
{
    return variables.count("verbose") > 0;
}

string Configuration::exportConfigurationFilename() const
{
    if (variables.count("export_configuration") == 0)
        return "";

    return variables["export_configuration"].as<string>();
}

string Configuration::importConfigurationFilename() const
{
    if (variables.count("import_configuration") == 0)
        return "";

    return variables["import_configuration"].as<string>();
}

bool Configuration::shouldPrintHelp() const
{
    return variables.count("help") > 0;
}

string Configuration::timeOutputFilename() const
{
    return variables["time_output"].as<string>();
}

void Configuration::printHelp()
{
    cout << createDescription() << endl;
}

options_description Configuration::createDescription()
{
    string categories = boost::algorithm::join(SchedulerFactory::getValidCategories(), "\n\t    ");
    string categoriesDescription = "Elf to be run, valid categories:\n\t    " + categories;
    string schedulingStrategies = boost::algorithm::join(
        MatrixOnlineSchedulingStrategyFactory::getStrategies(), "\n\t    ");
    string schedulingStrategiesDescription =
        "Matrix: Scheduling strategy for online scheduling mode:\n\t    " + schedulingStrategies;

    options_description description("Options");
    description.add_options()
        ("help,h",               "Print help message")
        ("mode,m",               value<string>()->required(), "Mode (smp|cuda)")
        ("category,c",           value<string>()->default_value("matrix"), categoriesDescription.c_str())
        ("numwarmups,w",         value<size_t>()->default_value(50), "Number of warmup rounds")
        ("numiter,n",            value<size_t>()->default_value(100), "Number of benchmark iterations")
        ("input,i",              value<string>(), "Input file")
        ("output,o",             value<string>(), "Output file")
        ("export_configuration", value<string>(), "Measure cluster and export configuration")
        ("import_configuration", value<string>(), "Run benchmark with given configuration")
        ("skip_benchmark",       "Skip the benchmark run")
        ("quiet,q",              "Do not output anything")
        ("verbose,v",            "Show output from all MPI processes")
        ("left_rows",            value<size_t>()->default_value(500), "Matrix: Number of left rows to be generated (overridden for benchmark by input file)")
        ("common_rows_columns",  value<size_t>()->default_value(500), "Matrix: Number of left columns / right rows to be generated (overridden for benchmark by input file)")
        ("right_columns",        value<size_t>()->default_value(500), "Matrix: Number of right columns to be generated (overridden for benchmark by input file)")
        ("scheduling,s",         value<string>()->default_value("row-wise"), schedulingStrategiesDescription.c_str())
        ("mpi_thread_multiple",  "Initialize MPI with thread multiple support.")
        ("left_digits",          value<size_t>()->default_value(8), "QuadraticSieve: Digits for product's left operand")
        ("right_digits",         value<size_t>()->default_value(8), "QuadraticSieve: Digits for product's right operand")
        ("time_output",          value<string>()->default_value("/dev/null"), "Output file for time measurements")
        ("montecarlo_trials",    value<size_t>()->default_value(100), "MonteCarlo: Number of trials for monte carlo tree search");

    return description;
}

string Configuration::mode() const
{
    return variables["mode"].as<string>();
}

string Configuration::category() const
{
    return variables["category"].as<string>();
}

bool Configuration::useFiles() const
{
    return inputFilename() != "" && outputFilename() != "";
}

string Configuration::inputFilename() const
{
    if (variables.count("input") == 0)
        return "";

    return variables["input"].as<string>();
}

string Configuration::outputFilename() const
{
    if (variables.count("output") == 0)
        return "";

    return variables["output"].as<string>();
}

size_t Configuration::leftMatrixRows() const
{
    return variables["left_rows"].as<size_t>();
}

size_t Configuration::commonMatrixRowsColumns() const
{
    return variables["common_rows_columns"].as<size_t>();
}

size_t Configuration::rightMatrixColumns() const
{
    return variables["right_columns"].as<size_t>();
}

string Configuration::schedulingStrategy() const
{
    return variables["scheduling"].as<string>();
}

bool Configuration::mpiThreadMultiple() const
{
    return variables.count("mpi_thread_multiple") > 0;
}

size_t Configuration::monteCarloTrials() const
{
    return variables["montecarlo_trials"].as<size_t>();
}

std::ostream& operator<<(std::ostream& s, const Configuration& c)
{
    s << "Configuation: "
        << "\n\tMode: " << c.mode()
        << "\n\tCategory: " << c.category()
        << "\n\tWarmUps: " << c.warmUps()
        << "\n\tIterations: " << c.iterations();

    if (c.useFiles())
    {
        s << "\n\tInput: " << c.inputFilename()
            << "\n\tOutput: " << c.outputFilename();
    }
    else
    {
        // TODO: This depends on the actual category...
        s << "\n\tMatrices: ("
            << c.leftMatrixRows() << " x " <<c.commonMatrixRowsColumns()
            << ") x ("
            << c.commonMatrixRowsColumns() << " x " << c.rightMatrixColumns()
            << ")";
    }
    return s;
}

unique_ptr<Scheduler> Configuration::createScheduler(const Communicator& communicator) const
{
    auto scheduler = createSchedulerFactory()->createScheduler(communicator);
    scheduler->configureWith(*this);
    return scheduler;
}
