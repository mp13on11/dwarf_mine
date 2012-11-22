#include "Kernel.h"
#include "../MatrixHelper.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

void Kernel::startup(const std::vector<std::string>& arguments)
{
    for(const auto& arg : arguments)
    std::cout << arg << std::endl;

    auto m = MatrixHelper::readMatrixFrom(arguments[0]);
}
    
void Kernel::run()
{
}

void Kernel::shutdown(const std::string& outputFilename)
{
    std::cout << outputFilename << std::endl;
}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    return std::shared_ptr<BenchmarkKernel>(new Kernel());
}