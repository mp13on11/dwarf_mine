#include "SMPMatrixElf.h"
#include <iostream>
#include <math.h>
#include <random>
#include <omp.h>
#include <functional>
#include "../Matrix.h"

size_t div_ceil(size_t x, size_t y)
{
    return (x % y) ? (x / y + 1) : (x / y);
}

void initRandom(std::function<float()>& generator, uint seed){
    auto distribution = std::uniform_real_distribution<float> (-100, +100);
    auto engine = std::mt19937(seed);
    generator = bind(distribution, engine);
}

void fillMatrix(Matrix<float>& m, size_t rows, size_t columns, const std::function<float()>& generator){
    for(size_t y = 0; y < rows; y++)
    {
        for(size_t x = 0; x < columns; x++)
        {
            m(y, x) = generator();
        }
    }
}

void SMPMatrixElf::run(std::istream& in, std::ostream& out)
{
    using namespace std;

    function<float()> generator;
    initRandom(generator, 1234);

    size_t columns = 106;//0;
    size_t rows = 106;//0;

    Matrix<float> a(rows, columns);
    Matrix<float> b(rows, columns);
    Matrix<float> c(rows, columns);

    fillMatrix(a, rows, columns, generator);
    fillMatrix(b, rows, columns, generator);

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
}