#pragma once

class Elf
{
public:
    virtual ~Elf() = 0;

    virtual void run() = 0;
};

inline Elf::~Elf()
{
}
