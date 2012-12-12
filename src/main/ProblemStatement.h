#pragma once

#include <iostream>

struct ProblemStatement
{
    typedef std::string ElfCategory;

    std::istream& input;
    std::ostream& output;
    ElfCategory elfCategory;
};
