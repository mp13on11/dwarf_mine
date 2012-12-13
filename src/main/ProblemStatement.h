#pragma once

#include "elves/ElfCategory.h"

#include <iostream>

struct ProblemStatement
{
    std::istream& input;
    std::ostream& output;
    ElfCategory elfCategory;
};
