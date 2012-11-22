#pragma once

#include <stdexcept>
#include <cublas_v2.h>

class CublasError: public std::runtime_error
{
public:
	static const std::string ERROR_MESSAGE;

	explicit CublasError(cublasStatus_t status) :
		status(status), std::runtime_error(ERROR_MESSAGE)
	{}

	const cublasStatus_t getStatusCode() const
	{
		return status;
	}

private:
	cublasStatus_t status;
};