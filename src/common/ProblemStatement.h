#pragma once

#include "elves/ElfCategory.h"
#include "DataGenerationParameters.h"

#include <memory>
#include <iosfwd>
#include <fstream>
#include <sstream>
#include <stdexcept>

class ProblemStatement
{
public:
    ProblemStatement(ElfCategory elfCategory, std::string inputFilename, std::string outputFilename) :
        input(streamOnFile(inputFilename, std::ios::in)),
        output(streamOnFile(outputFilename, std::ios::out)),
        elfCategory(elfCategory),
        _hasInput(true)
    {
    }

    ProblemStatement(ElfCategory elfCategory, const DataGenerationParameters& parameters) :
        input(std::unique_ptr<std::iostream>(new std::stringstream())),
        output(std::unique_ptr<std::iostream>(new std::stringstream())),
        elfCategory(elfCategory),
        _hasInput(false),
        params(parameters)
    {
    }

    std::istream& getInput()
    {
        return *input;
    }

    std::ostream& getOutput()
    {
        return *output;
    }

    bool hasInput() const
    {
        return _hasInput;
    }

    const DataGenerationParameters& getDataGenerationParameters() const
    {
        if (_hasInput)
            throw std::logic_error("Only input-less ProblemStatements have DataGenerationParameters");

        return params;
    }

private:
    static std::unique_ptr<std::iostream> streamOnFile(const std::string& filename, std::ios::openmode mode);

    std::unique_ptr<std::iostream> input;
    std::unique_ptr<std::iostream> output;
    ElfCategory elfCategory;
    bool _hasInput;
    DataGenerationParameters params;
};
