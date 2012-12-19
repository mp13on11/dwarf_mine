#pragma once

#include <Elf.h>

class CudaMatrixElf : public Elf
{
public:
    virtual void run(std::istream& input, std::ostream& output);
};
