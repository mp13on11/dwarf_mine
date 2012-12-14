#pragma once

#include "elves/ElfCategory.h"

#include <iostream>

struct ProblemStatement
{
    std::iostream& input;
    std::iostream& output;
    ElfCategory elfCategory;
};
