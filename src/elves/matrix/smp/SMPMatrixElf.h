#pragma once

#include <Elf.h>

template<typename T>
class Matrix;

class SMPMatrixElf : public Elf
{
public:
    Matrix<float> multiply(const Matrix<float>& left, const Matrix<float>& right);

    virtual void run(std::istream& in, std::ostream& out);
};
