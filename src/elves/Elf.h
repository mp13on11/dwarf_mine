#pragma once

#include <iosfwd>

class Elf
{
public:
    virtual ~Elf() = 0;
    //virtual void run(std::istream& input, std::ostream& output) = 0;
};

inline Elf::~Elf()
{
}
