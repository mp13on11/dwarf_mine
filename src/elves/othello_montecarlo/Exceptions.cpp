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

#include "Exceptions.h"

#include <sstream>
#include "OthelloUtil.h"

using namespace std;

InvalidFieldSizeException::InvalidFieldSizeException(size_t sideLength) :
    logic_error(constructMessage(sideLength))
{
}

string InvalidFieldSizeException::constructMessage(size_t sideLength)
{
    stringstream stream;
    stream << "Othello field size must be a multiple of 2 - given: "<<sideLength;
    return stream.str();
}


InvalidMoveException::InvalidMoveException(size_t sideLength, const Move& move) :
    runtime_error(constructMessage(sideLength, move))
{
}

string InvalidMoveException::constructMessage(size_t sideLength, const Move& move)
{
    stringstream stream;
    stream << "Move"<<move<<" outside of playfield ["<<sideLength<<"x"<<sideLength<<"]";
    return stream.str();
}


OccupiedFieldException::OccupiedFieldException( const Move& move) :
    runtime_error(constructMessage(move))
{
}

string OccupiedFieldException::constructMessage(const Move& move)
{
    stringstream stream;
    stream << "Move"<<move<<" on occupied field";
    return stream.str();
