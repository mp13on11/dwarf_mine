#include "gold/Kernel.h"
#include "tools/MatrixHelper.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <math.h>
#include <omp.h>

using namespace std;

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
    
size_t div_ceil(size_t x, size_t y)
{
    return (x % y) ? (x / y + 1) : (x / y);
}


void Kernel::run()
{
    size_t workerCount = 5;
    thread* workers = new thread[workerCount];


    size_t elementsPerBlock = 10*10;

    size_t columnsPerBlock = (size_t)ceil(sqrt(elementsPerBlock));
    columnsPerBlock = min(columnsPerBlock, c.columns());

    size_t rowsPerBlock = div_ceil(elementsPerBlock, columnsPerBlock);

    size_t blocksPerRow = div_ceil(c.columns(), columnsPerBlock);
    size_t blocksPerColumns = div_ceil(c.rows(), rowsPerBlock);

    size_t blockCount = blocksPerRow * blocksPerColumns;

    #pragma omp parallel for
    for(size_t index=0; index<blockCount; index++)
    {
        size_t rowStart = (index / blocksPerRow) * rowsPerBlock;
        size_t columnStart = (index % blocksPerRow) * columnsPerBlock;

        size_t columnEnd = min(columnStart+columnsPerBlock, c.columns());
        size_t rowEnd = min(rowStart+rowsPerBlock, c.rows());

        for(size_t y=rowStart; y<rowEnd; y++)
        {        
            for(size_t x=columnStart; x<columnEnd; x++)
            {
                float val = 0;
                for(size_t i=0; i<a.columns(); i++)
                {
                    val += a(y,i) * b(i,x);
                }
                c(y,x) = val;
            }
        }

    }

    delete[] workers;
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
