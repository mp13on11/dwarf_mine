#pragma once

#include "ElfCategory.h"
#include "Elf.h"
#include "Scheduler.h"
#include <memory>

class ElfFactory
{
public:
    ElfFactory(const ElfCategory& category);
    virtual ~ElfFactory() = 0;

    std::unique_ptr<Scheduler> createScheduler() const;

protected:
    ElfCategory _category;

    virtual std::unique_ptr<Scheduler> createSchedulerImplementation() const = 0;

private:
    void validate() const;
};

inline ElfFactory::~ElfFactory()
{
}

std::unique_ptr<ElfFactory> createElfFactory(const std::string& type, const ElfCategory& category);
