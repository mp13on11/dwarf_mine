#include "OthelloResult.h"

#include <iostream>

std::ostream& operator<<(std::ostream& stream, OthelloResult& result)
{
    stream << result.x << result.y << result.visits << result.wins;
    return stream;
}

std::istream& operator>>(std::istream& stream, OthelloResult& result)
{
    stream >> result.x;
    stream >> result.y;
    stream >> result.visits;
    stream >> result.wins;
    return stream;
}
