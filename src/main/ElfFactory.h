#pragma once

#include "Elf.h"
#include "ElfCategory.h"
#include <memory>

class ElfFactory
{
public:
    virtual ~ElfFactory() = 0;

    std::unique_ptr<Elf> createElf(const ElfCategory& category) const;

protected:
    virtual std::unique_ptr<Elf> createElfFrom(const ElfCategory& category) const = 0;

private:
    void validate(const ElfCategory& category) const;
};

inline ElfFactory::~ElfFactory()
{
}
