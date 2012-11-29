#pragma once

#include <exception>
#include <string>

class InvalidCommandLineException : public std::exception
{
public:
    InvalidCommandLineException();
    InvalidCommandLineException(const std::string& message);
    virtual ~InvalidCommandLineException() throw();

    virtual const char* what() const throw();

private:
    std::string message;
};

inline InvalidCommandLineException::InvalidCommandLineException() :
        message("invalid command line")
{
}

inline InvalidCommandLineException::InvalidCommandLineException(const std::string& message) :
        message(message)
{
}

inline InvalidCommandLineException::~InvalidCommandLineException() throw()
{
}

inline const char* InvalidCommandLineException::what() const throw()
{
    return message.c_str();
}
