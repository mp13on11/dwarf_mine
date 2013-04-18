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
    int processor = ratings.front().first;
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

// vector<MatrixSlice> MatrixSlicer::stripLayout(const BenchmarkResult& results, size_t rows, size_t columns) const
// {

//     setup(results);
    
//     size_t finishedIndex = 0;
//     size_t numItems = 0;
//     bool verticalStrips = true;

//     while(finishedIndex < results.size())
//     {
//         size_t numItem = layoutStrip(results, rows, columns, finishedIndex, verticalStrips);
        
//         if((finishedIndex + numItems) < results.size())
//         {
//             int numItem2 = layoutStrip(results, rows, columns, finishedIndex + numItems, verticalStrips);
//             double ar2a = computeAverageAspectRatio(results, finishedIndex, numItems + numItem2);
            
//             computeHorizontalLayout(results, rows, columns, finishedIndex, numItems + numItem2, verticalStrips);
//             double ar2b = double ar2a = computeAverageAspectRatio(results, finishedIndex, numItems + numItem2);

//             if(ar2b < ar2a)
//             {
//                 numItems += numItems2;
//             }
//             else
//             {
//                 computeHorizontalLayout(results, rows, columns, finishedIndex, numItems, verticalStrips);
//             }

//         }
//     }

//     return slices;
// }

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

// size_t MatrixSlicer::layoutStrip(BenchmarkResult& results, size_t rows, size_t columns, size_t finishedIndex, bool& verticalStrips) const
// {
//     size_t numItems = 0;
//     double previousAR = 0.0;
//     double ar = std::numeric_limits<double>::max();

//     do
//     {
//         previousAR = ar;
//         ++numItems;
//         computeHorizontalLayout(results, rows, columns, finishedIndex, numItems, verticalStrips);
//         ar = computeAverageAspectRatio(results, finishedIndex, numItems);
//     } while((ar < prevAR) && ((finishedIndex + numItems) < results.size()));

//     if(ar >= previousAR)
//     {
//         --numItems;
//         computeHorizontalLayout(results, rows, columns, finishedIndex, numItems, verticalStrips);
//         ar = computeAverageAspectRatio(results, finishedIndex, numItems);
//     }

//     return numItems;
// }

// void MatrixSlicer::computeHorizontalLayout(BenchmarkResult& results, size_t rows, size_t columns, size_t finishedIndex, size_t numItems, bool verticalStrips)
// {
//     double totalSize = computeSize(results, finishedIndex, numItems);
//     double height = verticalStrips? rows : columns;
//     double width;
//     double x = 0;

//     for(size_t i = index; i < numItems + finishedIndex, i++)
//     {
//         width = sizeForNode(results[i]) / height;
//         if(verticalStrips)
//         {
//             slices.push_back(MatrixSlice{results[i].first, 0, x, height, x + width});
//         }
//         else
//         {
//             slices.push_back(MatrixSlice{results[i].first, x, 0, x + height, width});   
//         }
//         x += width;
//     }
// }

// int MatrixSlicer::computeSize(BenchmarkResult& results, size_t finishedIndex, size_t numItems)
// {
//     int size = 0;
//     for(int i = finishedIndex; i < numItems + finishedIndex; i++)
//     {
//         size += sizeForNode(results[i]);
//     }
// }

// int MatrixSlicer::sizeForNode(BenchmarkResult& result)
// {
//     return result.second;
// }

// double MatrixSlicer::computeAverageAspectRatio(BenchmarkResult& results, size_t finishedIndex, size_t numItems)
// {
//     double total = 0.0;

//     for(int i = finishedIndex; i < numItems + finishedIndex; i++)
//     {
//         total += computeAspectRatio(results[i]);
//     }

//     return total / numItems;

// }
