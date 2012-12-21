#include "GoldMatrixElf.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <matrix/Matrix.h>

MatrixElf::MatrixT GoldMatrixElf::multiply(const Matrix<float>& left, const Matrix<float>& right)
{
    MatrixT result(left.rows(), right.columns());
    for(size_t y=0; y<result.rows(); ++y)
    {
        for(size_t x=0; x<result.columns(); ++x)
        {
            float val = 0;
            for(size_t i=0; i<left.columns(); ++i)
            {
                val += left(y,i) * right(i,x);
            }
            result(y,x) = val;
        }
    }
    return result;
}
