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

