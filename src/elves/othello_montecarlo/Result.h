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

#include <iostream>

struct Result
{
    unsigned int x;
    unsigned int y;
    unsigned int visits;
    unsigned int wins;

    Result(size_t _x = 0, size_t _y = 0, size_t _visits = 0, size_t _wins = 0)
    {
        x = _x;
        y = _y;
        visits = _visits;
        wins = _wins;
    }

    bool equalMove(const Result& other) const
    {
        return (x == other.x && y == other.y);
    }

    double successRate() const
    {
        return 1.0 * wins / visits;
    }

    bool operator<(const Result& other) const 
    {
        return this->successRate() < other.successRate();
    }

    std::ostream& operator<<(std::ostream& stream)
    {
        stream << "Move: "<<x<<", "<<y<<" ("<<wins<<"/"<<visits<<")"<<successRate();
        return stream;
    }
