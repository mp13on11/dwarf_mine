#pragma once

#include <stdexcept>
#include <boost/lexical_cast.hpp>
#include <cublas_v2.h>

class CublasError: public std::runtime_error
{
public:
	static const std::string ERROR_MESSAGE;

	explicit CublasError(cublasStatus_t status) :
        std::runtime_error(ERROR_MESSAGE + " " + boost::lexical_cast<std::string>(status)), 
        status(status)
	{}

	cublasStatus_t getStatusCode() const
	{
		return status;
	}

private:
	cublasStatus_t status;
};
