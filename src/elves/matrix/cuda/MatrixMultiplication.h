#pragma once

const int DEFAULT_BLOCK_SIZE = 24;

extern void gemm(int m, int n, int k, float* left, float* right, float* out, int blockSize = DEFAULT_BLOCK_SIZE);
