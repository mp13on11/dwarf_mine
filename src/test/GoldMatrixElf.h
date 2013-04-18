#pragma once

#include <MatrixElf.h>

template<typename T>
class Matrix;

class GoldMatrixElf : public MatrixElf
{
public:
    MatrixT multiply(const Matrix<float>& left, const Matrix<float>& right);
};
