#pragma once

#include "MatrixSlice.h"
#include "common/BenchmarkResults.h"
#include <vector>
#include <list>

class MatrixSlicerSquarified
{

	public:
		typedef std::vector<MatrixSlice> SliceList;
    	typedef std::list<NodeRating> RatingList;

    	SliceList layout(const BenchmarkResult& results, size_t rows, size_t columns);

    private:
    	SliceList _slices;
    	MatrixSlice _unlayoutedSlice;
    	std::list<NodeRating> _unlayoutedRatings;

    	double calculateRatio(size_t smallestSide, RatingList strip);
    	double calculateRatio(size_t smallestSide, RatingList strip, NodeRating head);

    	size_t getSmallestSide();
    	size_t getLongestSide();
    	bool isVerticalStrip();
		size_t getX();
		size_t getY();
		void setUnlayoutedSlice(const std::list<NodeRating>& stripRatings, size_t x, size_t y, size_t rows, size_t columns);

		void addToLayout(std::list<NodeRating> stripRatings);
		void squarify(std::list<NodeRating>& strip);

		void setup(const BenchmarkResult& results, size_t area);

};
