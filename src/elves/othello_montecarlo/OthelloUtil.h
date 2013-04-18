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

#include <iosfwd>
#include <vector>
#include <functional>
#include <Result.h>
#include <Move.h>
#include <Field.h>
#include <iostream>

typedef std::function<size_t(size_t)> RandomGenerator;

// shortcuts
const Field F = Field::Free;
const Field W = Field::White;
const Field B = Field::Black;

typedef std::vector<Field> Playfield;

typedef std::vector<Move> MoveList;

std::ostream& operator<<(std::ostream& stream, const Field field);

std::istream& operator>>(std::istream& stream, Field& field);

// operator overloads

inline Move operator+(const Move& first, const Move& second)
{
    return Move{first.x + second.x, first.y + second.y};
}

inline void operator+=(Move& value, const Move& other)
{
    value.x += other.x;
    value.y += other.y;
}

inline bool operator==(const Move& first, const Move& second)
{
    return first.x == second.x && first.y == second.y;
}

inline std::ostream& operator<<(std::ostream& stream, const Move& move)
{
    stream << "{" << move.x << ", " << move.y << "}";
    return stream;
}

// explicit helper methods

namespace OthelloHelper
{
    size_t generateUniqueSeed(size_t nodeId, size_t threadId, size_t commonSeed);

    void writeResultToStream(std::ostream& stream, const Result& result);

    void readResultFromStream(std::istream& stream, Result& result);
    
    void writePlayfieldToStream(std::ostream& stream, const Playfield& playfield);
    
    void readPlayfieldFromStream(std::istream& stream, Playfield& playfield);
}

