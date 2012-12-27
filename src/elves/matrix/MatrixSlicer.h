#pragma once

#include "MatrixSlice.h"
#include <main/BenchmarkResults.h>
#include <list>
#include <vector>

class MatrixSlicer
{
public:
    typedef std::vector<MatrixSlice> SliceList;
    typedef std::list<NodeRating> RatingList;

    SliceList sliceAndDice(const BenchmarkResult& results, size_t rows, size_t columns);

private:
    void sliceRows(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns);
    void sliceColumns(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns);
    void setup(const BenchmarkResult& results);

    SliceList slices;
    RatingList ratings;
};
