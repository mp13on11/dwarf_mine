#pragma once

#include "elves/ElfCategory.h"

#include <memory>
#include <iosfwd>
#include <fstream>
#include <sstream>
#include <stdexcept>

struct ProblemStatement
{
    std::unique_ptr<std::iostream> input;
    std::unique_ptr<std::iostream> output;
    ElfCategory elfCategory;

    ProblemStatement(ElfCategory elfCategory, std::string inputFilename, std::string outputFilename)
        : elfCategory(elfCategory)       
    {
        auto i = new std::fstream();
        auto o = new std::fstream();
        i->open(inputFilename);

        if (!i->is_open())
            throw std::runtime_error("Failed to open file: " + inputFilename);

        o->open(outputFilename);

        if (!o->is_open())
            throw std::runtime_error("Failed to open file: " + outputFilename);

        input = std::unique_ptr<std::iostream>(i);
        output = std::unique_ptr<std::iostream>(o);

    }

    ProblemStatement(ElfCategory elfCategory) 
        : input(std::unique_ptr<std::iostream>(new std::stringstream())), output(std::unique_ptr<std::iostream>(new std::stringstream())), elfCategory(elfCategory)
    {
        
    }
};
