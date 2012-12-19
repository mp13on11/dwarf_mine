#pragma once

#include <Elf.h>

class SMPMatrixElf : public Elf
{

public:
    virtual void run(std::istream& in, std::ostream& out);
};
