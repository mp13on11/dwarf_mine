#pragma once

#include <iosfwd>
#include <vector>
#include <functional>
#include "OthelloMove.h"


typedef std::function<size_t(size_t)> RandomGenerator;

enum class Field { Free, Black, White };

// shortcuts
#define F Field::Free
#define W Field::White
#define B Field::Black

typedef Field Player;
typedef std::vector<Field> Playfield;

struct OthelloResult
{
    size_t x;
    size_t y;
    size_t visits;
    size_t wins;
    size_t iterations;

    bool equalMove(const OthelloResult& other) const
    {
        return (x == other.x && y == other.y);
    }

    double successRate() const
    {
        return 1.0 * wins / visits;
    }
};

namespace OthelloHelper
{
    void writeResultToStream(std::ostream& stream, OthelloResult& result);

    void readResultFromStream(std::istream& stream, OthelloResult& result);
    
    void writePlayfieldToStream(std::ostream& stream, const Playfield& playfield);
    
    void readPlayfieldFromStream(std::istream& stream, Playfield& playfield);
}

std::ostream& operator<<(std::ostream& stream, const Field field);

std::istream& operator>>(std::istream& stream, Field& field);