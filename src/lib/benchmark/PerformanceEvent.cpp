#include "benchmark/PerformanceEvent.h"
#include <algorithm>
#include <papi.h>
#include "tools/PapiHelper.h"

PerformanceEvent::PerformanceEvent(const std::string& name)
: name(name)
{
    generatedCode = false;
}

PerformanceEvent::PerformanceEvent(const std::string& name, const int code)
: PerformanceEvent(name)
{
    this->code = code;
    generatedCode = true;
}

PerformanceEvent::~PerformanceEvent()
{
}

const std::string& PerformanceEvent::getName() const
{
    return name;
}

int PerformanceEvent::getCode() const
{
    if (!generatedCode)
        generateCode();
    return code;
}

void PerformanceEvent::generateCode() const
{
    char* convertedName = getNameAsCharArray();
    int result = PAPI_event_name_to_code(convertedName, &code);
    PapiHelper::handleResult(result == PAPI_OK, result, __FILE__, __LINE__);
    generatedCode = true;
    free(convertedName);
}

char* PerformanceEvent::getNameAsCharArray() const
{
    char* convertedName = new char[name.size() + 1];
    std::copy(name.cbegin(), name.cend(), convertedName);
    convertedName[name.size()] = '\0';
    return convertedName;
}
