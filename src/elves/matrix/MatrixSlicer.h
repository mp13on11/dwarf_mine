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
	size_t testDeterminePivot(size_t rowsOrCols, const BenchmarkResult& results) const;
private:
    void sliceRows(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const;
    void sliceColumns(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const;
    void setup(const BenchmarkResult& results) const;
    size_t determinePivot(size_t rowsOrCols) const;
    size_t processRating(size_t y, size_t x, size_t rows, size_t cols, bool colWise) const;

    mutable SliceList slices;
    mutable RatingList ratings;
};
