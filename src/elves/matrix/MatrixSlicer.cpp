#include "MatrixSlicer.h"
#include "MatrixSlice.h"
#include <numeric>
#include <cmath>

using namespace std;

void MatrixSlicer::sliceColumns(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns)
{
    if (ratings.size() == 0)
    {
        return;
    }
    int processor = ratings.front().first;

    int pivot = columns;
    if (ratings.size() > 1)
    {
        int overall = 0;
        for (const auto& s : ratings)
        {
            overall += s.second;
        }
        pivot = ceil(columns * ratings.front().second * 1.0 / overall);
    }
    slices.push_back(MatrixSlice{processor, columnOrigin, rowOrigin, columnOrigin + pivot, rows});
    ratings.pop_front();
    sliceRows(rowOrigin, columnOrigin + pivot, rows, columns - pivot);
}

void MatrixSlicer::sliceRows(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns)
{
    if (ratings.size() == 0)
    {
        return;
    }
    int processor = ratings.front().first;

    int pivot = rows;
    if (ratings.size() > 1)
    {
        int overall = 0;
        for (const auto& s : ratings)
        {
            overall += s.second;
        }
        pivot = ceil(rows * ratings.front().second * 1.0 / overall);
    }
    slices.push_back(MatrixSlice{processor, columnOrigin, rowOrigin, columns, rowOrigin + pivot});
    ratings.pop_front();
    sliceColumns(rowOrigin + pivot, columnOrigin, rows - pivot, columns);
}

vector<MatrixSlice> MatrixSlicer::sliceAndDice(const BenchmarkResult& results, size_t rows, size_t columns)
{
    setup(results);
    sliceColumns(0, 0, rows, columns);
    return slices;
}

void MatrixSlicer::setup(const BenchmarkResult& results)
{
    slices.clear();

    RatingList orderedRatings(results.begin(), results.end());
    orderedRatings.sort(
        [](const NodeRating& a, const NodeRating& b)
        {
            return a.second > b.second;
        }
    );
    ratings = move(orderedRatings);
}
