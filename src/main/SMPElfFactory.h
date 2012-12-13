#pragma once

#include "main/ElfFactory.h"

class SMPElfFactory : public ElfFactory
{
protected:
    virtual std::unique_ptr<Elf> createElfFrom(const ElfCategory& category) const;
};
