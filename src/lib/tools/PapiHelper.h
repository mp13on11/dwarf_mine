#pragma once

#include <memory>
#include <string>

class PapiHelper
{
public:
    static int eventCodeFrom(const std::string& name);

private:
    static void handleResult(int resultCode, const std::string& fileName = "", int lineNumber = 0);
    static void printError(int resultCode, const std::string& fileName, int lineNumber);
    static std::unique_ptr<char[]> convert(const std::string& name);
};
