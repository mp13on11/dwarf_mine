#pragma once

#include <iostream>

struct Result
{
    size_t x;
    size_t y;
    size_t visits;
    size_t wins;

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
};