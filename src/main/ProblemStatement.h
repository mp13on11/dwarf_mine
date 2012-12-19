#pragma once

#include "elves/ElfCategory.h"

#include <iosfwd>

struct ProblemStatement
{
    std::iostream& input;
    std::iostream& output;
    ElfCategory elfCategory;
};
