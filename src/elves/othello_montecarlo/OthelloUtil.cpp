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

#include "OthelloUtil.h"
#include <Result.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

namespace OthelloHelper
{
    size_t generateUniqueSeed(size_t nodeId, size_t threadId, size_t commonSeed)
    {
        std::stringstream buffer;
        buffer << commonSeed << " ~ "<< nodeId << " ~ " << threadId;
        std::hash<std::string> generateHash;
        return generateHash(buffer.str());
    }

    void writeResultToStream(std::ostream& stream, const Result& result)
    {
        stream << result.x 
               << result.y 
               << result.visits 
               << result.wins; 
    }

    void readResultFromStream(std::istream& stream, Result& result)
    {
        stream >> result.x
               >> result.y
               >> result.visits
               >> result.wins;
    }

    void writePlayfieldToStream(std::ostream& stream, const std::vector<Field>& playfield)
    {
        for (const auto& field : playfield)
        {
            stream << field;
        }
    }

    void readPlayfieldFromStream(std::istream& stream, std::vector<Field>& playfield)
    {
        if (stream.good())
        {
            Field field;
            stream >> field;
            while (stream.good())
            {
                playfield.push_back(field);
                stream >> field;
            }
        }
        else
        {
            throw std::runtime_error("Unexpected empty stream");
        }
        if (playfield.size() != 64)
        {
            throw std::runtime_error("Invalid playfield size");
        }
    }
}

std::ostream& operator<<(std::ostream& stream, const Field field)
{
    if (field == Field::Black)
    {
        stream << "B";
    }
    else if (field == Field::White)
    {
        stream << "W";
    }
    else if (field == Field::Free)
    {
        stream << "F";
    }
    else 
    {
        stream << "?";
    }
    return stream;
}

std::istream& operator>>(std::istream& stream, Field& field)
{
    char sign;
    stream >> sign;
    switch (sign)
    {
        case 'B': field = Field::Black;
            break;
        case 'W': field = Field::White;
            break;
        case 'F': field = Field::Free;
            break;
        default:
            field = Field::Illegal;
    }
    return stream;
