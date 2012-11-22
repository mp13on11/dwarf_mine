#include <iostream>
#include <climits>
#include <thread>
#include <papi.h>
using namespace std;
unsigned long int currentThreadId = ULONG_MAX;

void hello()
{
    int result;

    result = PAPI_register_thread();
    if (result != PAPI_OK)
        cout << "Couldn't initialize thread " << currentThreadId <<
            ". (Error " << result << ")" << endl;

    std::cout << "Hello, World! " << std::endl;
}

unsigned long int nextThreadId()
{
    return ++currentThreadId;
}

int main()
{
    int result;

    result = PAPI_library_init(PAPI_VER_CURRENT);
    if (result != PAPI_VER_CURRENT && result > 0)
        cout << "Couldn't initialize PAPI library. (Error " << result << ")" << endl;

    result = PAPI_thread_init(nextThreadId);
    if (result != PAPI_OK)
        cout << "Couldn't initialize PAPI's threading support. (Error " << result << ")" << endl;

    std::thread t1(hello), t2(hello);
    t1.join();
    t2.join();
    return 0;
}

