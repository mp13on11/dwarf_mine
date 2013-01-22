#pragma once

#include "MatrixSlice.h"
#include "common/BenchmarkResults.h"
#include <list>
#include <vector>

class MatrixSlicer
{
public:
    typedef std::vector<MatrixSlice> SliceList;
    typedef std::list<NodeRating> RatingList;

    SliceList sliceAndDice(const BenchmarkResult& results, size_t rows, size_t columns) const;
    SliceList stripLayout(const BenchmarkResult& results, size_t rows, size_t columns) const; // WIP
	size_t testDeterminePivot(size_t rowsOrCols, const BenchmarkResult& results) const;
private:
    void sliceRows(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const;
    void sliceColumns(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const;
    void setup(const BenchmarkResult& results) const;
    size_t determinePivot(size_t rowsOrCols) const;
    size_t processRating(size_t y, size_t x, size_t rows, size_t cols, bool colWise) const;

    size_t layoutStrip(const BenchmarkResult& results, size_t rows, size_t columns, size_t finishedIndex, bool verticalStrips) const; // WIP
    double computeAverageAspectRatio(BenchmarkResult& results, size_t finishedIndex, size_t numItems); // WIP
    int sizeForNode(BenchmarkResult& result); // WIP
    int computeSize(BenchmarkResult& results, size_t finishedIndex, size_t numItems); // WIP


    mutable SliceList slices;
    mutable RatingList ratings;
};
