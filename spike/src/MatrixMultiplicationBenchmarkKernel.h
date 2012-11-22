#ifndef MATRIX_MULTIPLICATION_BENCHMARK_KERNEL_H_
#define MATRIX_MULTIPLICATION_BENCHMARK_KERNEL_H_

#include "BenchmarkKernel.h"
#include "Matrix.h"

#include <istream>
#include <string>
#include <vector>

class MatrixMultiplicationBenchmarkKernel : public BenchmarkKernel
{
public:
	virtual void startup(const std::vector<std::string> &arguments);
	virtual void run();
	virtual void shutdown(const std::string &outputFilename);

protected:
	virtual Matrix<float> multiply(const Matrix<float> &a, const Matrix<float> &b) = 0;

private:
	Matrix<float> left;
	Matrix<float> right;
	Matrix<float> result;

	static Matrix<float> readMatrixFrom(const std::string &fileName);
	static void fillMatrixFromStream(Matrix<float> &matrix, std::istream &stream);
	static std::vector<float> getValuesIn(const std::string &line);
};

inline void MatrixMultiplicationBenchmarkKernel::run()
{
	result = multiply(left, right);
}

#endif /* MATRIX_MULTIPLICATION_BENCHMARK_KERNEL_H_ */
