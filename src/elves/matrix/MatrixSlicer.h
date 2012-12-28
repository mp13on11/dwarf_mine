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

    SliceList sliceAndDice(const BenchmarkResult& results, size_t rows, size_t columns) const;

private:
    void sliceRows(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const;
    void sliceColumns(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const;
    void setup(const BenchmarkResult& results) const;

    mutable SliceList slices;
    mutable RatingList ratings;
};
