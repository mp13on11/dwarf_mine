#include "CommandLineConfiguration.h"
#include "ProblemStatement.h"
#include "SchedulerFactory.h"
#include "matrix/Matrix.h"
#include "matrix/MatrixHelper.h"
#include "montecarlo/OthelloUtil.h"

#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;
using namespace boost::program_options;

CommandLineConfiguration::CommandLineConfiguration(int argc, char** argv) :
    description(createDescription()), variables()
{
    store(parse_command_line(argc, argv, description), variables);
    notify(variables);

    if (mode() != "smp" && mode() != "cuda")
        throw error("Mode must be smp or cuda");
    if (variables.count("input") ^ variables.count("output"))
        throw error("Both input and output are needed, if one is given");
}

unique_ptr<ProblemStatement> generateMatrixProblemStatement(string category, size_t leftRows, size_t commonRowsColumns, size_t rightColumns)
{
    auto statement = unique_ptr<ProblemStatement>(new ProblemStatement(category));
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

unique_ptr<ProblemStatement> generateMonteCarloProblemStatement(string category)
{
    auto statement = unique_ptr<ProblemStatement>(new ProblemStatement(category));
    vector<Field> playfield = {
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, W, B, F, F, F,      
        F, F, F, B, W, F, F, F,      
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F
    };
    // vector<Field> playfield = {
    //     B, B, B, W, W, W, W, W,
    //     B, B, B, B, W, W, W, W,
    //     B, B, B, B, W, W, W, W,
    //     B, B, B, B, W, W, W, W,      
    //     B, B, B, B, W, W, W, W,      
    //     B, B, B, B, W, W, W, W,
    //     B, B, B, B, W, W, W, W,
    //     W, B, B, F, B, B, B, W
    // };
    OthelloHelper::writePlayfieldToStream(*(statement->input), playfield);
    return statement;
}

unique_ptr<ProblemStatement> CommandLineConfiguration::createProblemStatement(bool forceGenerated) const
{
    if(!useFiles() || forceGenerated)
    {
        if (category() == "matrix")
        {
            return generateMatrixProblemStatement(
                    category(),
                    leftMatrixRows(),
                    commonMatrixRowsColumns(),
                    rightMatrixColumns()
                );
        } else if (category() == "montecarlo")
        {
            return generateMonteCarloProblemStatement(category());
        }
        throw error("Generated problem statement for "+category()+" not supported");
    }
    return unique_ptr<ProblemStatement>(
            new ProblemStatement(category(), inputFilename(), outputFilename())
        );
}

unique_ptr<SchedulerFactory> CommandLineConfiguration::createSchedulerFactory() const
{
    return SchedulerFactory::createFor(mode(), category());
}

size_t CommandLineConfiguration::iterations() const
{
    return variables["numiter"].as<size_t>();
}

size_t CommandLineConfiguration::warmUps() const
{
    return variables["numwarmups"].as<size_t>();
}

bool CommandLineConfiguration::shouldExportConfiguration() const
{
    return exportConfigurationFilename() != "";
}

bool CommandLineConfiguration::shouldImportConfiguration() const
{
    return importConfigurationFilename() != "";
}

bool CommandLineConfiguration::shouldSkipBenchmark() const
{
    return variables.count("skip_benchmark") > 0;
}

bool CommandLineConfiguration::shouldBeQuiet() const
{
    return variables.count("quiet") > 0;
}
bool CommandLineConfiguration::shouldBeVerbose() const
{
    return variables.count("verbose") > 0;
}

string CommandLineConfiguration::exportConfigurationFilename() const
{
    if (variables.count("export_configuration") == 0)
        return "";

    return variables["export_configuration"].as<string>();
}

string CommandLineConfiguration::importConfigurationFilename() const
{
    if (variables.count("import_configuration") == 0)
        return "";

    return variables["import_configuration"].as<string>();
}

bool CommandLineConfiguration::shouldPrintHelp() const
{
    return variables.count("help") > 0;
}

string CommandLineConfiguration::timeOutputFilename() const
{
    return variables["time_output"].as<string>();
}

void CommandLineConfiguration::printHelp()
{
    cout << createDescription() << endl;
}

options_description CommandLineConfiguration::createDescription()
{
    options_description description("Options");
    description.add_options()
        ("help,h",               "Print help message")
        ("mode,m",               value<string>()->required(), "Mode (smp|cuda)")
        ("category,c",           value<string>()->default_value("matrix"), "Elf to be run (matrix|factorize|montecarlo)")
        ("numwarmups,w",         value<size_t>()->default_value(50), "Number of warmup rounds")
        ("numiter,n",            value<size_t>()->default_value(100), "Number of benchmark iterations")
        ("input,i",              value<string>(), "Input file")
        ("output,o",             value<string>(), "Output file")
        ("export_configuration", value<string>(), "Measure cluster and export configuration")
        ("import_configuration", value<string>(), "Run benchmark with given configuration")
        ("skip_benchmark",       "Skip the benchmark run")
        ("quiet,q",              "Do not output anything")
        ("verbose,v",            "Show output from all MPI processes")
        ("left_rows",            value<size_t>()->default_value(500), "Number of left rows to be generated (overridden for benchmark by input file)")
        ("common_rows_columns",  value<size_t>()->default_value(500), "Number of left columns / right rows to be generated (overridden for benchmark by input file)")
        ("right_columns",        value<size_t>()->default_value(500), "Number of right columns to be generated (overridden for benchmark by input file)")
        ("time_output",          value<string>()->default_value("/dev/null"), "Output file for time measurements");

    return description;
}

string CommandLineConfiguration::mode() const
{
    return variables["mode"].as<string>();
}

string CommandLineConfiguration::category() const
{
    return variables["category"].as<string>();
}

bool CommandLineConfiguration::useFiles() const
{
    return inputFilename() != "" && outputFilename() != "";
}

string CommandLineConfiguration::inputFilename() const
{
    if (variables.count("input") == 0)
        return "";

    return variables["input"].as<string>();
}

string CommandLineConfiguration::outputFilename() const
{
    if (variables.count("output") == 0)
        return "";

    return variables["output"].as<string>();
}

size_t CommandLineConfiguration::leftMatrixRows() const
{
    return variables["left_rows"].as<size_t>();
}

size_t CommandLineConfiguration::commonMatrixRowsColumns() const
{
    return variables["common_rows_columns"].as<size_t>();
}

size_t CommandLineConfiguration::rightMatrixColumns() const
{
    return variables["right_columns"].as<size_t>();
}

std::ostream& operator<<(std::ostream& s, const CommandLineConfiguration& c)
{
    s << "Configuation: "
            << "\n\tMode: "<< c.mode()
            << "\n\tWarmUps: " << c.warmUps()
            << "\n\tIterations: " << c.iterations();

    if (c.useFiles())
    {
        s << "\n\tInput: " << c.inputFilename()
                << "\n\tOutput: " << c.outputFilename();
    }
    else
    {
        s << "\n\tMatrices: ("
                << c.leftMatrixRows() << " x " <<c.commonMatrixRowsColumns()
                << ") x ("
                << c.commonMatrixRowsColumns() << " x " << c.rightMatrixColumns()
                << ")";
    }
    return s;
}
