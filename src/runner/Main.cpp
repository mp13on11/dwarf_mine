#include "SimpleBenchmarkRunner.h"
#include "SimpleConfiguration.h"
#include "common/MpiGuard.h"

#include <exception>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;

ostream& operator<<(ostream& out, const vector<microseconds>& measurements)
{
    for (const auto& time : measurements)
    {
        out << time.count() << endl;
    }

    return out;
}

int main(int argc, char** argv)
{
    try
    {
        MpiGuard guard(argc, argv);
        SimpleConfiguration config(argc, argv);

        if (config.shouldPrintHelp())
        {
            SimpleConfiguration::printHelp();
            return 0;
        }

        ofstream file(config.timeOutputFilename(), ios::app);

        if (!file.is_open())
        {
            cerr << "Failed to open file " << config.timeOutputFilename() << endl;
            return 1;
        }

        SimpleBenchmarkRunner runner(config);
        file << runner.run();
    }
    catch (const boost::program_options::error& e)
    {
        SimpleConfiguration::printHelp();
        cerr << e.what() << endl;
        return 1;
    }
    catch (const exception& e)
    {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "FATAL ERROR of unknown type" << endl;
        return 1;
    }
    return 0;
}
