#pragma once

#include <matrix/MatrixSlice.h>
#include <vector>
#include <cstddef>

typedef std::vector<MatrixSlice> SliceList;

void verifySlice(MatrixSlice& slice, size_t x, size_t y, size_t columns, size_t rows);

std::ostream& operator<<(std::ostream& stream, const MatrixSlice& slice);
std::ostream& operator<<(std::ostream& stream, const SliceList& slices);