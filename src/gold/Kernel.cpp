#include "gold/Kernel.h"
#include "tools/MatrixHelper.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

void printMatrix(const Matrix<float>& matrix) 
{
    std::cout << "[";
    for(size_t y=0; y<matrix.rows(); y++)
    {        
        if(y>0)
            std::cout << std::endl;
        std::cout << "[";
        for(size_t x=0; x<matrix.columns(); x++)
        {
            if(x>0)
                std::cout << " ";
            std::cout << matrix(y, x);
        }
        std::cout << "]";
    }
    std::cout << "]" << std::endl;
}

void Kernel::startup(const std::vector<std::string>& arguments)
{
    //for(const auto& arg : arguments)
    //std::cout << arg << std::endl;

    a = MatrixHelper::readMatrixFrom(arguments[0]);
    b = MatrixHelper::readMatrixFrom(arguments[1]);
    c = Matrix<float>(a.rows(), b.columns());
}
    
void Kernel::run()
{

    for(size_t y=0; y<c.rows(); y++)
    {        
        for(size_t x=0; x<c.columns(); x++)
        {
            float val = 0;
            c(y,x) = 0;
            for(size_t i=0; i<a.columns(); i++)
            {
                val += a(y,i) * b(i,x);
            }
            c(y,x) = val;
        }
    }
}

void Kernel::shutdown(const std::string& outputFilename)
{
    //std::cout << outputFilename << std::endl;
    //printMatrix(c);
    MatrixHelper::writeMatrixTo(outputFilename, c);
}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    return std::shared_ptr<BenchmarkKernel>(new Kernel());
}
