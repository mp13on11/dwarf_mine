#pragma once

#include <matrix/MatrixSlice.h>
#include <vector>
#include <cstddef>

typedef std::vector<MatrixSlice> SliceList;
typedef std::vector<size_t> AreaList;

void verifySlice(const MatrixSlice& slice, size_t x, size_t y, size_t columns, size_t rows);
void verifySlices(const SliceList& slices, const AreaList& area);

std::ostream& operator<<(std::ostream& stream, const MatrixSlice& slice);
std::ostream& operator<<(std::ostream& stream, const SliceList& slices);