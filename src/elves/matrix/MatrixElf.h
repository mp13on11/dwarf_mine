#pragma once

#include <Elf.h>

template<typename T>
class Matrix;

class MatrixElf : public Elf
{
public:
    typedef Matrix<float> MatrixT;

    virtual MatrixT multiply(const MatrixT& left, const MatrixT& right) = 0;
    virtual void run(std::istream& input, std::ostream& output);
};
