#include <iostream>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <random>
#include <map>
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>
#include <boost/lexical_cast.hpp>

using namespace std;

void printUsage();
void printUsage(const string&);

void verifyMatrices(const vector<string>& args)
{
    throw runtime_error("Not implemented yet!");
}

void generateMatrix(const vector<string>& args)
{
    using namespace boost;

    if (args.size() < 3)
        printUsage("<rows> <cols> <out_file>");

    auto rows = lexical_cast<size_t>(args[0]);
    auto cols = lexical_cast<size_t>(args[1]);
    auto outFile = args[2];

    Matrix<float> result(rows, cols);
    auto distribution = uniform_real_distribution<float> (-100, +100);
    auto engine = mt19937(time(nullptr));
    auto generator = bind(distribution, engine);
    MatrixHelper::fill(result, generator);
    MatrixHelper::writeMatrixTo(outFile, result);
}

map<string, function<void(const vector<string>&)>> subCommands =
{
    { "verify", [](const vector<string>& args)
        {
            verifyMatrices(args);
        }
    },
    { "generate", [](const vector<string>& args)
        {
            generateMatrix(args);
        }
    }
};

void printUsage()
{
    cout
        << "Usage: matrixtool <command> [<args>]" << endl
        << "with <command> being one of:" << endl;

    for (const auto& sc : subCommands)
    {
        cout << "\t" << sc.first << endl;
    }

    exit(1);
}

void printUsage(const string& msg)
{
    cerr << "Usage: " << msg << endl;
    exit(1);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printUsage();
    }

    vector<string> arguments(argv+2, argv+argc);

    subCommands[argv[1]](arguments);

    return 0;
}
