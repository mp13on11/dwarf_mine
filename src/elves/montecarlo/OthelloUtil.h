#pragma once

#include <iosfwd>
#include <vector>
#include <functional>
#include <OthelloResult.h>
#include <OthelloField.h>
#include <iostream>

typedef std::function<size_t(size_t)> RandomGenerator;


// shortcuts
const Field F = Field::Free;
const Field W = Field::White;
const Field B = Field::Black;

typedef std::vector<Field> Playfield;

std::ostream& operator<<(std::ostream& stream, const Field field);

std::istream& operator>>(std::istream& stream, Field& field);

namespace OthelloHelper
{
    size_t generateUniqueSeed(size_t nodeId, size_t threadId, size_t commonSeed);

    void writeResultToStream(std::ostream& stream, const OthelloResult& result);

    void readResultFromStream(std::istream& stream, OthelloResult& result);
    
    void writePlayfieldToStream(std::ostream& stream, const Playfield& playfield);
    
    void readPlayfieldFromStream(std::istream& stream, Playfield& playfield);
}

