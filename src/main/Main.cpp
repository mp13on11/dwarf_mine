#include "common/BenchmarkRunner.h"
#include "common/Communicator.h"
#include "common/Configuration.h"
#include "common/MpiGuard.h"
#include "common/MpiHelper.h"
#include "common/TimingProfiler.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

void exportWeightedCommunicator(const string& filename, const Communicator& communicator)
{
    fstream file(filename, fstream::out);

    if (!file.is_open())
    {
        cerr << "ERROR: Could not write "<< filename <<endl;
        exit(1);
    }

    for (double weight : communicator.weights())
        file << weight << endl;

    file.close();
}

Communicator importWeightedCommunicator(const string& filename)
{
    fstream file(filename, fstream::in);

    if (!file.is_open())
    {
        cerr << "ERROR: Could not read "<<filename<<endl;
        exit(1);
    }

    vector<double> weights;

    while (file.good())
    {
        double weight;
        file >> weight;

        if (file.good())
            weights.push_back(weight);
    }

    file.close();

    return Communicator(weights);
}

void silenceOutputStreams(bool keepErrorStreams = false)
{
    cout.rdbuf(nullptr);

    if (!keepErrorStreams)
    {
        cerr.rdbuf(nullptr);
        clog.rdbuf(nullptr);
    }
}

vector<double> nodeWeightsFrom(const vector<microseconds>& averageExecutionTimes)
{
    microseconds sum(0);
    for (microseconds time : averageExecutionTimes)
    {
        sum += time;
    }

    vector<double> weights;
    for (microseconds time : averageExecutionTimes)
    {
        weights.push_back(static_cast<double>(time.count()) / static_cast<double>(sum.count()));
    }
    return weights;
}

Communicator determineWeightedCommunicator(const BenchmarkRunner& runner, const Communicator& unweightedCommunicator)
{
    TimingProfiler profiler;
    vector<microseconds> averageTimes;
    
    for (size_t i=0; i<unweightedCommunicator.size(); ++i)
    {
        cout << "Creating Communicator ..." << endl;
        Communicator subCommunicator = unweightedCommunicator.createSubCommunicator(
                {Communicator::MASTER_RANK*1, static_cast<int>(i)}
            );
        cout << "Communicator successfully created" << endl;
        runner.benchmarkNode(subCommunicator, profiler);
        averageTimes.push_back(profiler.averageIterationTime());
    }
    cout << "Weight determination session completed" << endl;
    return Communicator(nodeWeightsFrom(averageTimes));
}

void printResults(const TimingProfiler& profiler, ostream& timeFile)
{
    cout << "Execution times (microseconds):" << endl;
    
    for (const auto& iterationTime : profiler.iterationTimes())
    {
        cout << "\t" << iterationTime.count() << endl;

        if (MpiHelper::isMaster())
            timeFile << iterationTime.count() << endl;
    }
}

void benchmarkWith(Configuration& config)
{
    BenchmarkRunner runner(config);
    TimingProfiler profiler;

    ofstream timeFile;

    if (MpiHelper::isMaster())
    {
        timeFile.open(config.timeOutputFilename(), ios::app);

        if (!timeFile.is_open())
            throw runtime_error("Failed to open file \"" + config.timeOutputFilename() + "\"");
    }

    if(config.shouldRunWithoutMPI())
    {
        if (MpiHelper::numberOfNodes() > 1)
            throw runtime_error("Process was told to run without MPI support, but was called via mpirun");

        runner.runElf(Communicator(), profiler);
        printResults(profiler, timeFile);
    }
    else
    {
        Communicator communicator;
        if (config.shouldExportConfiguration() || !config.shouldImportConfiguration())
        {
            cout << "Calculating node weights" << endl;
            communicator = determineWeightedCommunicator(runner, communicator);
        }
        if (config.shouldExportConfiguration())
        {
            cout << "Exporting node weights" << endl;
            exportWeightedCommunicator(config.exportConfigurationFilename(), communicator);
        }
        if (config.shouldImportConfiguration())
        {
            cout << "Importing node weights" << endl;
            communicator = importWeightedCommunicator(config.importConfigurationFilename());
        }
        if (!config.shouldSkipBenchmark())
        {
            cout << "Running benchmark" << endl;
            runner.runBenchmark(communicator, profiler);
            printResults(profiler, timeFile);
        }
    }
}

int main(int argc, char** argv)
{
    // used to ensure MPI::Finalize is called on exit of the application
    MpiGuard guard(argc, argv);

    try
    {
        Configuration config(argc, argv);

        if (config.shouldPrintHelp())
        {
            config.printHelp();
            return 0;
        }

        if (!config.shouldBeVerbose() && (config.shouldBeQuiet() || !MpiHelper::isMaster()))
            silenceOutputStreams(true);

        cout << config << endl;

        benchmarkWith(config);
    }
    catch (const boost::program_options::error& e)
    {
        Configuration::printHelp();
        cerr << e.what() << endl;
        return 1;
    }
    catch (const logic_error& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
    catch (const exception& e)
    {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "FATAL ERROR of unknown type." << endl;
        return 1;
    }

    return 0;
}

