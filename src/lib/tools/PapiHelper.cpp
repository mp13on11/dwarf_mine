#include "PapiHelper.h"
#include <iostream>
#include <papi.h>

void PapiHelper::handleResult(
    const bool isOk,
    const int resultCode,
    const std::string& fileName,
    const int lineNumber)
{
    if (isOk)
        return;
    printError(resultCode, fileName, lineNumber);
    exit(1);
}
    
void PapiHelper::printError(
    const int resultCode,
    const std::string& fileName,
    const int lineNumber)
{
    std::cout << "PAPI error "
        << resultCode << ": " << PAPI_strerror(resultCode)
        << std::endl;
    if (fileName != "" && lineNumber != 0)
        std::cout << "(" << fileName << ":" << lineNumber << ")" << std::endl;
}
