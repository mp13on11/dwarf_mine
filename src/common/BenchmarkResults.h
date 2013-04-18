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

#include <map>
#include <iosfwd>

typedef double Rating; // TODO
typedef std::pair<int, Rating> NodeRating;

class BenchmarkResult
{
public:
    typedef std::map<int, Rating>::const_iterator const_iterator;
    typedef std::map<int, Rating>::iterator iterator;
    typedef std::map<int, Rating>::value_type value_type;

    BenchmarkResult();
    BenchmarkResult(const std::map<int,Rating>& ratings);
    BenchmarkResult(std::initializer_list<value_type> init);

    Rating& operator[](const int& nodeId);
    const_iterator begin() const;
    const_iterator end() const;
    iterator begin();
    iterator end();

    size_t size() const;
    bool empty() const;

private:
    std::map<int, Rating> _ratings;
};

std::ostream& operator<<(std::ostream& o, const BenchmarkResult& results);


inline BenchmarkResult::BenchmarkResult()
{
}

inline BenchmarkResult::BenchmarkResult(const std::map<int,Rating>& ratings) : 
    _ratings(ratings) 
{
}

inline BenchmarkResult::BenchmarkResult(std::initializer_list<value_type> init) :
    _ratings(init) 
{
}

inline BenchmarkResult::const_iterator BenchmarkResult::begin() const
{
    return _ratings.begin();
}

inline BenchmarkResult::const_iterator BenchmarkResult::end() const
{
    return _ratings.end();
}

inline BenchmarkResult::iterator BenchmarkResult::begin()
{
    return _ratings.begin();
}

inline BenchmarkResult::iterator BenchmarkResult::end()
{
    return _ratings.end();
}

inline size_t BenchmarkResult::size() const 
{
    return _ratings.size();
}

inline Rating& BenchmarkResult::operator[](const int& nodeId)
{
    return _ratings[nodeId];
}

inline bool BenchmarkResult::empty() const
{
    return _ratings.empty();
}