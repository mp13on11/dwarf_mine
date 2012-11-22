#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include "Matrix.h"
#include "MatrixMultiplicationBenchmarkKernel.h"

using namespace std;

void generateMatrix(const string &fileName);

bool isRunArgument(const string &arg);

void printUsage(const string &program);

int main(int argc, const char *argv[])
{
	if (argc < 3)
	{
		printUsage(argv[0]);
		return 1;
	}

	vector<string> arguments(argv + 1, argv + argc);

	if (arguments[0] == "generate")
	{
		generateMatrix(arguments[1]);
	}
	else if (isRunArgument(arguments[0]) && arguments.size() == 4)
	{
		auto kernel = MatrixMultiplicationBenchmarkKernel::create(arguments[0]);
		kernel->startup(vector<string>{arguments[1], arguments[2]});
		kernel->run();
		kernel->shutdown(arguments[3]);
	}
	else
	{
		printUsage(argv[0]);
		return 1;
	}
}

void generateMatrix(const string &fileName)
{
	const size_t size = 1000;
	default_random_engine engine;
	uniform_real_distribution<float> distribution(0, 1000);

	ofstream file;
	file.exceptions(ios_base::failbit);
	file.open(fileName);

	file << size << " " << size << endl;

	for (size_t i=0; i<size; i++)
	{
		if (i % 100 == 0)
			cout << "written " << i << " rows" << endl;
		for (size_t j=0; j<size; j++)
		{
			file << " " << distribution(engine);
		}

		file << endl;
	}
}

bool isRunArgument(const string &arg)
{
	return arg == "cuda" || arg == "mpi" || arg == "smp";
}

void printUsage(const string &program)
{
	cerr << "Usage: " << endl;
	cerr << "\t" << program << " <cuda|mpi|smp> <left matrix file> <right matrix file> <result matrix file>" << endl;
	cerr << "\t" << program << " <generate> <matrix file>" << endl;
}
