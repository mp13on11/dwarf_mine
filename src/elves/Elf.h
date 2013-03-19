#pragma once

#include <iosfwd>

class Elf
{
public:
    virtual ~Elf() = 0;
};

inline Elf::~Elf()
{
}
