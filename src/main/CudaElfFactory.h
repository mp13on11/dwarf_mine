#pragma once

#include "main/ElfFactory.h"

class CudaElfFactory : public ElfFactory
{
protected:
    virtual std::unique_ptr<Elf> createElfFrom(const ElfCategory& category) const;
};
