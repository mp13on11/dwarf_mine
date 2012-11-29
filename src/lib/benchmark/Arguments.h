#pragma once

#include <iostream>
#include <string>
#include <vector>

class Arguments
{
public:
    static void printUsage(const std::string& program, std::ostream& out = std::cout);

    Arguments();
    Arguments(int argc, const char* argv[]);

    std::size_t iterations() const;
    const std::string& program() const;
    const std::vector<std::string>& inputFileNames() const;
    const std::string& outputFileName() const;
    std::string toString() const;

private:
    std::string _program;
    std::size_t _iterations;
    std::vector<std::string> _inputs;
    std::string _output;
};

inline std::size_t Arguments::iterations() const
{
    return _iterations;
}

inline const std::string& Arguments::program() const
{
    return _program;
}

inline const std::vector<std::string>& Arguments::inputFileNames() const
{
    return _inputs;
}

inline const std::string& Arguments::outputFileName() const
{
    return _output;
}
