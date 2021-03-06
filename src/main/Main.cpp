/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

#include "common/BenchmarkRunner.h"
#include "common/Communicator.h"
#include "common/Configuration.h"
#include "common/MpiGuard.h"
#include "common/NodeWeightProfiler.h"
#include "common/TimingProfiler.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

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

Communicator createSubCommunicatorFor(const Communicator& communicator, int rank)
{
    if (rank == Communicator::MASTER_RANK)
    {
        return communicator.createSubCommunicator({
                Communicator::Node(rank, 1.0)
            });
    }
    else
    {
        return communicator.createSubCommunicator({
                Communicator::Node(Communicator::MASTER_RANK, 0.0),
                Communicator::Node(rank, 1.0)
            });
    }
}

Communicator determineWeightedCommunicator(const BenchmarkRunner& runner, const Communicator& unweightedCommunicator)
{
    NodeWeightProfiler profiler;

    for (size_t i=0; i<unweightedCommunicator.size(); ++i)
    {
        Communicator subCommunicator = createSubCommunicatorFor(unweightedCommunicator, i);
        runner.runPreBenchmark(subCommunicator, profiler);
        profiler.saveExecutionTime();
    }

    return Communicator(profiler.nodeWeights());
}

void printResults(const Communicator& communicator, const TimingProfiler& profiler, ostream& timeFile)
{
    cout << "Execution times (microseconds):" << endl;
    
    for (const auto& iterationTime : profiler.iterationTimes())
    {
        cout << "\t" << iterationTime.count() << endl;

        if (communicator.isMaster())
            timeFile << iterationTime.count() << endl;
    }

    cout << "Average: " << profiler.averageIterationTime().count() << endl;
}

void benchmarkWith(const Configuration& config)
{
    BenchmarkRunner runner(config);
    TimingProfiler profiler;
    Communicator communicator;
    ofstream timeFile;

    if (config.shouldPrintHelp())
    {
        config.printHelp();
        return;
    }

    if (!config.shouldBeVerbose() && (config.shouldBeQuiet() || !communicator.isMaster()))
        silenceOutputStreams(true);

    cout << config << endl;


    if (communicator.isMaster())
    {
        timeFile.open(config.timeOutputFilename(), ios::app);

        if (!timeFile.is_open())
            throw runtime_error("Failed to open file \"" + config.timeOutputFilename() + "\"");
    }

    if(communicator.size() == 1)
    {
        runner.runBenchmark(communicator, profiler);
        printResults(communicator, profiler, timeFile);
    }
    else
    {
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
            printResults(communicator, profiler, timeFile);
        }
    }
}

int main(int argc, char** argv)
{
    try
    {
        // used to ensure MPI::Finalize is called on exit of the application
        Configuration configuration(argc, argv);
        MpiGuard guard(configuration.mpiThreadMultiple(), argc, argv);

        benchmarkWith(configuration);
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

