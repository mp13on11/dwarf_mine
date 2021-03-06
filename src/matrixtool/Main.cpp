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

#include <iostream>
#include <fstream>
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

void verifyMatrices(const vector<string>&)
{
    throw runtime_error("Not implemented yet!");
}

function<float()> makeGenerator()
{
    auto distribution = uniform_real_distribution<float> (-100, +100);
    auto engine = mt19937(time(nullptr));
    return bind(distribution, engine);
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
    auto generator = makeGenerator();
    MatrixHelper::fill(result, generator);
    MatrixHelper::writeMatrixTo(outFile, result);

    cout << "Matrix generated." << endl;
}

void generateMatrixPair(const vector<string>& args)
{
    using namespace boost;

    if (args.size() < 4)
        printUsage("<rows> <middle> <cols> <out_file>");

    auto rows = lexical_cast<size_t>(args[0]);
    auto middle = lexical_cast<size_t>(args[1]);
    auto cols = lexical_cast<size_t>(args[2]);
    auto outFile = args[3];

    Matrix<float> left(rows, middle);
    Matrix<float> right(middle, cols);
    auto generator = makeGenerator();
    MatrixHelper::fill(left, generator);
    MatrixHelper::fill(right, generator);

    ofstream output(outFile, ios_base::binary);

    MatrixHelper::writeMatrixPairTo(output, { left, right });

    cout << "Matrix pair generated." << endl;
}

void convert(const vector<string>& args, bool toBinary)
{
    if (args.size() < 2)
        printUsage("<in_file> <out_file>");

    auto inFlags = ios_base::in;
    auto outFlags = ios_base::out;

    if (toBinary)
        outFlags |= ios_base::binary;
    else
        inFlags |= ios_base::binary;

    ifstream inFile(args[0], inFlags);
    ofstream outFile(args[1], outFlags);

    size_t count(0);

    while (inFile.peek() != EOF && inFile.good() && !inFile.eof())
    {
        if (toBinary)
        {
            Matrix<float> input(MatrixHelper::readMatrixTextFrom(inFile));
            MatrixHelper::writeMatrixTo(outFile, input);
        }
        else
        {
            Matrix<float> input(MatrixHelper::readMatrixFrom(inFile));
            MatrixHelper::writeMatrixTextTo(outFile, input);
        }
        ++count;
    }

    cout << "Converted " << count << " matrices." << endl;
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
    },
    { "generate_pair", [](const vector<string>& args)
        {
            generateMatrixPair(args);
        }
    },
    {
        "text2bin", [](const vector<string>& args)
        {
            convert(args, true);
        }
    },
    {
        "bin2text", [](const vector<string>& args)
        {
            convert(args, false);
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
