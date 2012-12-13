#pragma once

#include <iostream>

class Elf
{
public:
    virtual ~Elf() = 0;
    virtual void run(std::istream& input = std::cin, std::ostream& output = std::cout) = 0;
};

inline Elf::~Elf()
{
}
