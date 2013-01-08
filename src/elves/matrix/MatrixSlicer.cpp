#include "MatrixSlicer.h"
#include "MatrixSlice.h"
#include <cmath>
#include <cassert>

using namespace std;

size_t MatrixSlicer::determinePivot(size_t rowsOrCols) const
{
    size_t pivot = rowsOrCols;

    if (ratings.size() > 1)
    {
        Rating overallPercentRatings = 0;
        Rating invertedOverallPercentRatings = 0;
        for (const auto& s : ratings)
        {
            overallPercentRatings += s.second;
            invertedOverallPercentRatings += 100 - s.second;
		}

        if (overallPercentRatings > 0)
        {
			double invertedRuntimePercentRating = 100 - ratings.front().second;

            pivot = ceil(rowsOrCols * invertedRuntimePercentRating * (1.0 / invertedOverallPercentRatings));
		}
        else
            pivot = 0;        
    }    
    ratings.pop_front();
    return pivot;
}

size_t MatrixSlicer::processRating(size_t y, size_t x, size_t rows, size_t cols, bool colWise) const
{
    NodeId processor = ratings.front().first;
    size_t pivot = determinePivot(colWise ? cols : rows);
    
    if (colWise)
        cols = pivot;
    else
        rows = pivot;
    
    slices.push_back(MatrixSlice{processor, x, y, cols, rows});
    return pivot;
}

void MatrixSlicer::sliceColumns(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const
{
    if (ratings.empty()) return;

    size_t pivot = processRating(rowOrigin, columnOrigin, rows, columns, true);
    sliceRows(rowOrigin, columnOrigin + pivot, rows, columns - pivot);
}

void MatrixSlicer::sliceRows(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const
{
    if (ratings.empty()) return;
    
    size_t pivot = processRating(rowOrigin, columnOrigin, rows, columns, false);
    sliceColumns(rowOrigin + pivot, columnOrigin, rows - pivot, columns);
}

vector<MatrixSlice> MatrixSlicer::sliceAndDice(const BenchmarkResult& results, size_t rows, size_t columns) const
{
    setup(results);
    sliceColumns(0, 0, rows, columns);
    return slices;
}

void MatrixSlicer::setup(const BenchmarkResult& results) const
{
    slices.clear();

    RatingList orderedRatings(results.begin(), results.end());
    orderedRatings.sort(
        [](const NodeRating& a, const NodeRating& b)
        {
            return a.second < b.second;
        }
    );
    ratings = move(orderedRatings);
}
