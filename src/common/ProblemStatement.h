/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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

    std::istream& getInput() const
    {
        return *input;
    }

    std::ostream& getOutput() const
    {
        return *output;
    }

    bool hasInput() const
    {
        return _hasInput;
    }

    const DataGenerationParameters& getDataGenerationParameters() const
    {
        return params;
    }

private:
    static std::unique_ptr<std::iostream> streamOnFile(const std::string& filename, std::ios::openmode mode);

    std::unique_ptr<std::iostream> input;
    std::unique_ptr<std::iostream> output;
    ElfCategory elfCategory;
    bool _hasInput;
    DataGenerationParameters params;
