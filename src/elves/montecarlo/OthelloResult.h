#pragma once

#include <iosfwd>

struct OthelloResult
{
    int x;
    int y;
    size_t visits;
    size_t wins;

    
};

std::ostream& operator<<(std::ostream& stream, OthelloResult& result);

std::istream& operator>>(std::istream& stream, OthelloResult& result);
