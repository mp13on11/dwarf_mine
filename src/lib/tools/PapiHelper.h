#pragma once

#include <string>

class PapiHelper
{
public:
    static void handleResult(
        const bool isOk,
        const int resultCode,
        const std::string& fileName = "",
        const int lineNumber = 0);
private:
    static void printError(
        const int resultCode,
        const std::string& fileName,
        const int lineNumber);
};
