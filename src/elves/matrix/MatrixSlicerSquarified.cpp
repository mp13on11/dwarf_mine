#include "MatrixSlicerSquarified.h"
#include <cmath>
#include <numeric>

using namespace std;

double sum(list<NodeRating>::iterator begin, list<NodeRating>::iterator end, double sum = 0)
{
	//for (const auto& nodeRating : ratings)
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
	_unlayoutedSlice = MatrixSlice{0, x, y, rows, columns};
	removeAll(_unlayoutedRatings, stripRatings);
}

void MatrixSlicerSquarified::addToLayout(list<NodeRating> stripRatings)
{
	size_t longestSide = getLongestSide();
	size_t smallestSide = getSmallestSide();
	size_t x = getX();
	size_t y = getY();
	removeAll(_unlayoutedRatings, stripRatings);

	double sum_ratings = sum(stripRatings.begin(), stripRatings.end());
	double sum_unlayouted_ratings = sum(_unlayoutedRatings.begin(), _unlayoutedRatings.end());

	double pivot_ratio = sum_ratings / (sum_unlayouted_ratings + sum_ratings);
	
	size_t pivotSideLength = ceil(longestSide * pivot_ratio);
	size_t stripArea = pivotSideLength * smallestSide;
	list<NodeRating> sortedStrips(stripRatings);
	// sort ascending
	sortedStrips.sort(
        [](const NodeRating& a, const NodeRating& b)
        {
            return a.second < b.second;
        }
    );
	
	for (const auto& stripRating : sortedStrips)
	{
		size_t sliceAreaRatio = (stripRating.second / sum_ratings) * stripArea;
		size_t cols = sliceAreaRatio / smallestSide;
		size_t rows = smallestSide;
		if (isVerticalStrip())
		{
			swap(cols, rows);
		}
		_slices.push_back(MatrixSlice{stripRating.first, x, y, cols, rows});
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
		setUnlayoutedSlice(stripRatings, getX() + smallestSide, getY(), getSmallestSide(), getLongestSide() - pivotSideLength);
	}
	else
	{
		setUnlayoutedSlice(stripRatings, getX(), getY()  + smallestSide, getLongestSide() - pivotSideLength, getSmallestSide());
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
	double s_square = smallestSide * smallestSide;
	double sum_weightings = sum(strip.begin(), strip.end());
	double sum_weighting_square = sum_weightings * sum_weightings;

	size_t worst_ratio = 0;

	for(auto weighting : strip)
	{	
		auto ratio = max(sum_weighting_square * weighting.second / s_square, s_square / sum_weighting_square * weighting.second);
		if(ratio > worst_ratio)
		{
			worst_ratio = ratio;
		}
	}

	return worst_ratio;
}

void MatrixSlicerSquarified::squarify(list<NodeRating>& strip)
{
	auto head = _unlayoutedRatings.front();
	auto smallestSide = getSmallestSide();

	auto aspectRatio = calculateRatio(smallestSide, strip);
	auto aspectRatioWithHead = calculateRatio(smallestSide, strip, head);

	if(aspectRatio <= aspectRatioWithHead)
	{
		strip.push_back(head);
		_unlayoutedRatings.pop_front();
	}
	else 
	{
		addToLayout(strip);
		strip.clear();
	}
	squarify(strip);
}

void MatrixSlicerSquarified::setup(const BenchmarkResult& results)
{
    _slices.clear();
    for (const auto& rating : results)
    	_unlayoutedRatings.push_back(rating);
    _ratings = results;
}

vector<MatrixSlice> MatrixSlicerSquarified::layout(const BenchmarkResult& results, size_t rows, size_t columns)
{
	setup(results);
	
	setUnlayoutedSlice(list<NodeRating>(), 0, 0, rows, columns);
	list<NodeRating> ratings;
	squarify(ratings);

	return _slices;
}