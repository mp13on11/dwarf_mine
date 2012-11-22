#include "CudaMatrixKernel.h"

#include <iostream>

using namespace std;

void CudaMatrixKernel::startup(const std::vector<std::string>& arguments)
{

}

void CudaMatrixKernel::run()
{
    cout << "Hello World" << endl;
}

void CudaMatrixKernel::shutdown(const std::string& outputFilename)
{

}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    BenchmarkKernel* kernel = new CudaMatrixKernel();
    return std::shared_ptr<BenchmarkKernel>(kernel);
}
