#include "PapiHelper.h"
#include <iostream>
#include <papi.h>

using namespace std;

int PapiHelper::eventCodeFrom(const string& name)
{
    int result = 0;
    unique_ptr<char[]> converted = convert(name);
    int status = PAPI_event_name_to_code(converted.get(), &result);
    PapiHelper::handleResult(status, __FILE__, __LINE__);

    return result;
}

void PapiHelper::handleResult(int resultCode, const string& fileName, int lineNumber)
{
    if (resultCode == PAPI_OK)
        return;

    printError(resultCode, fileName, lineNumber);
    exit(1);
}
    
void PapiHelper::printError(int resultCode, const string& fileName, int lineNumber)
{
    cout << "PAPI error " << resultCode << ": " << PAPI_strerror(resultCode);

    if (fileName != "" && lineNumber != 0)
        cout << "(" << fileName << ":" << lineNumber << ")";

    cout << endl;
}

unique_ptr<char[]> PapiHelper::convert(const string &name)
{
    char *converted = new char[name.size() + 1];
    copy(name.cbegin(), name.cend(), converted);
    converted[name.size()] = '\0';

    return unique_ptr<char[]>(converted);
}
