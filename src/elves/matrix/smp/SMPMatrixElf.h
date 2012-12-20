#pragma once

#include <MatrixElf.h>

class SMPMatrixElf : public MatrixElf
{
public:
    virtual MatrixT multiply(const MatrixT& left, const MatrixT& right);
};
