#include "Matrix.h"
#include "SMPMatrixElf.h"
#include "common/Utils.h"

#include <cmath>
#include <functional>
#include <random>

using namespace std;

SMPMatrixElf::MatrixT SMPMatrixElf::multiply(const MatrixT& left, const MatrixT& right)
{
    if (left.empty() || right.empty()) return MatrixT();
    MatrixT c(left.rows(), right.columns());

    size_t elementsPerBlock = 10*10;
    size_t columnsPerBlock = (size_t)ceil(sqrt(elementsPerBlock));
    columnsPerBlock = min(columnsPerBlock, c.columns());

    size_t rowsPerBlock = div_ceil(elementsPerBlock, columnsPerBlock);

    size_t blocksPerRow = div_ceil(c.columns(), columnsPerBlock);
    size_t blocksPerColumns = div_ceil(c.rows(), rowsPerBlock);

    size_t blockCount = blocksPerRow * blocksPerColumns;

    MatrixT transposed = right.transposed();

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
                for(size_t i=0; i<left.columns(); i++)
                {
                    val += left(y,i) * transposed(x,i);
                }
                c(y,x) = val;
            }
        }

    }

    return c;
}
