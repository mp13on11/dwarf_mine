#include "MatrixSlicerSquarified.h"
#include <cmath>
#include <numeric>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace std;

double sum(list<NodeRating>::iterator begin, list<NodeRating>::iterator end, double sum = 0)
{
	for (auto i = begin; i != end; i++)
		sum += (*i).second;
	return sum;
}

void removeAll(list<NodeRating>& ratings, list<NodeRating> toBeRemoved)
{
	for (const auto& rating : toBeRemoved)
		ratings.remove(rating);
}


size_t MatrixSlicerSquarified::getSmallestSide()
{
	return min(_unlayoutedSlice.getColumns(), _unlayoutedSlice.getRows());
}

size_t MatrixSlicerSquarified::getLongestSide()
{
	return max(_unlayoutedSlice.getColumns(), _unlayoutedSlice.getRows());
}

bool MatrixSlicerSquarified::isVerticalStrip()
{
	return _unlayoutedSlice.getColumns() < _unlayoutedSlice.getRows();
}

size_t MatrixSlicerSquarified::getX()
{
	return _unlayoutedSlice.getStartX();
}

size_t MatrixSlicerSquarified::getY()
{
	return _unlayoutedSlice.getStartY();
}

void MatrixSlicerSquarified::setUnlayoutedSlice(const list<NodeRating>& stripRatings, size_t x, size_t y, size_t rows, size_t columns)
{
	_unlayoutedSlice = MatrixSlice(0, x, y, rows, columns);
	removeAll(_unlayoutedRatings, stripRatings);
}

void sortByRatingsAsc(list<NodeRating>& ratings)
{
	if(ratings.size() <= 1)
	{
		return;
	}
	ratings.sort(
        [](const NodeRating& a, const NodeRating& b) -> bool
        {
       		return a.second < b.second;
        }
    );
}

void sortByRatingsAsc(vector<NodeRating>& ratings)
{
	if(ratings.size() <= 1)
	{
		return;
	}
	sort(ratings.begin(), ratings.end(),
        [](const NodeRating& a, const NodeRating& b) -> bool
        {
       		return a.second < b.second;
        }
    );
}


void MatrixSlicerSquarified::addToLayout(list<NodeRating> stripRatings)
{
	size_t longestSide = getLongestSide();
	size_t smallestSide = getSmallestSide();
	size_t x = getX();
	size_t y = getY();

	removeAll(_unlayoutedRatings, stripRatings);

	double sumRatings = sum(stripRatings.begin(), stripRatings.end());
	if (sumRatings == 0)
	{
		sumRatings = 1;
	}
	double sumUnlayoutedRatings = sum(_unlayoutedRatings.begin(), _unlayoutedRatings.end());

	double pivotRatio = sumRatings / (sumUnlayoutedRatings + sumRatings);
	
	size_t pivotSideLength = ceil(longestSide * pivotRatio);
	size_t stripArea = pivotSideLength * smallestSide;
	list<NodeRating> sortedStrips(stripRatings);
	// sort ascending
	sortByRatingsAsc(sortedStrips);
	for (const auto& stripRating : sortedStrips)
	{
		size_t sliceArea = (stripRating.second / sumRatings) * stripArea;
		size_t cols = sliceArea / pivotSideLength;
		size_t rows = pivotSideLength;

		if (isVerticalStrip())
		{
			swap(cols, rows);
		}

		if(stripRating == sortedStrips.back())
		{
			if(isVerticalStrip())
			{
				rows = _unlayoutedSlice.getStartY() + smallestSide - y;
		
			}
			else
			{
				cols = _unlayoutedSlice.getStartX() + smallestSide - x;	
				
			}
		}
		_slices.push_back(MatrixSlice(stripRating.first, x, y, cols, rows));
		if (isVerticalStrip())
		{
			y += rows;
		}
		else
		{
			x += cols;
		}
	}
	if (isVerticalStrip())
	{
		setUnlayoutedSlice(stripRatings, getX() + pivotSideLength, getY(), getSmallestSide(), getLongestSide() - pivotSideLength);
	}
	else
	{
		setUnlayoutedSlice(stripRatings, getX(), getY()  + pivotSideLength, getLongestSide() - pivotSideLength, getSmallestSide());
	}
}

double MatrixSlicerSquarified::calculateRatio(size_t smallestSide, list<NodeRating> strip, NodeRating head)
{
	strip.push_back(head);
	double result = calculateRatio(smallestSide, strip);
	strip.pop_back();
	return result;
}

double MatrixSlicerSquarified::calculateRatio(size_t smallestSide, list<NodeRating> strip)
{
	if (strip.size() == 0)
	{
		return 0;
	}
	double w = smallestSide;
	double s = 0;
	Rating min_r = numeric_limits<Rating>::max();
	Rating max_r = numeric_limits<Rating>::min();
	for (const auto& rating : strip)
	{
		s += rating.second;
		min_r = min(min_r, rating.second);
		max_r = max(max_r, rating.second);
	}
	double w_squared = w * w;
	double s_squared = s * s;
	double worst = min((w_squared * max_r) / s_squared, s_squared / (w_squared * min_r));
	return worst;
}

void MatrixSlicerSquarified::squarify(list<NodeRating>& strip)
{
	if (_unlayoutedRatings.size() == 0)
	{
		addToLayout(strip);
		return;
	}
	auto head = _unlayoutedRatings.front();
	auto smallestSide = getSmallestSide();

	auto aspectRatio = calculateRatio(smallestSide, strip);
	auto aspectRatioWithHead = calculateRatio(smallestSide, strip, head);

	if(aspectRatio <= aspectRatioWithHead)
	{
		strip.push_back(head);
		_unlayoutedRatings.pop_front();
		squarify(strip);
	}
	else 
	{
		addToLayout(strip);
		strip.clear();
		squarify(strip);
	}
}

void MatrixSlicerSquarified::setup(const BenchmarkResult& results, size_t area)
{
    _slices.clear();
    Rating ratingSum = 0;
    Rating ratingMax = 0;
    Rating ratingMin = numeric_limits<Rating>::max();
    // 0.3 0.5 1 1
    // 3, 2, 1, 1
    // 7
    vector<NodeRating> positiveRatings;
    for (const auto& rating : results)
    {
    	ratingMax = max(ratingMax, rating.second);
    	ratingMin = min(ratingMin, rating.second);
    }
    for (const auto& rating : results)
    {
    	Rating positiveRating = ratingMin / rating.second;
    	ratingSum += positiveRating;
    	positiveRatings.emplace_back(rating.first, positiveRating);
    }
    // we shuffle the ratings a bit to make sure that large slices can be placed along small ones
    // - to optimize this, it would result in binpacking which is np
    // sortByRatingsAsc(positiveRatings);
    // size_t n = positiveRatings.size();
    // for (size_t i = 0; i < n / 2; ++i)
    // {
    // 	swap(positiveRatings[i], positiveRatings[n / 2 * i]);
    // }
    for (const auto& rating : positiveRatings)
    {
    	_unlayoutedRatings.emplace_back(rating.first, round(rating.second / ratingSum * area));
    }
    
}

vector<MatrixSlice> MatrixSlicerSquarified::layout(const BenchmarkResult& results, size_t rows, size_t columns)
{
	setup(results, rows * columns);
	
	setUnlayoutedSlice(list<NodeRating>(), 0, 0, rows, columns);
	list<NodeRating> ratings;
	squarify(ratings);
	// for(auto slice : _slices)
	// {
	// 	cout << "("
 //        << slice.getStartX() << ", "
 //        << slice.getStartY() << ", "
 //        << slice.getColumns() << ", "
 //        << slice.getRows() << ")" << endl;
	// }
	return _slices;
}