#pragma once

#include <MatrixElf.h>

class CudaMatrixElf : public MatrixElf
{
public:
    virtual MatrixT multiply(const MatrixT& left, const MatrixT& right);
};
