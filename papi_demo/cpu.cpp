#include <iostream>
#include <climits>
#include <thread>
#include <mutex>
#include <papi.h>
using namespace std;

mutex outputSyncMutex;
unsigned long int currentThreadId = ULONG_MAX;

unsigned long int nextThreadId()
{
    return ++currentThreadId;
}

void hello()
{
    int eventSet = PAPI_NULL;
    long long eventCounts[1];
    int presetEvent = PAPI_TOT_INS;

    if (PAPI_register_thread() != PAPI_OK)
        cout << "Couldn't initialize thread." << endl;

    if (PAPI_create_eventset(&eventSet) != PAPI_OK)
        cout << "Couldn't create event set for thread." <<endl;

    if (PAPI_add_event(eventSet, presetEvent) != PAPI_OK)
        cout << "Couldn't add event to set of thread." << endl;
    
    if (PAPI_start(eventSet) != PAPI_OK)
        cout << "Couldn't start measurement for thread." << endl;
    
    outputSyncMutex.lock();
    cout << "Hello, World! " << endl;
    outputSyncMutex.unlock();

    if (PAPI_stop(eventSet, eventCounts) != PAPI_OK)
        cout << "Couldn't stop measurement for thread." << endl;

    outputSyncMutex.lock();
    cout << "PAPI_TOT_INS = " << eventCounts[0] << endl;
    outputSyncMutex.unlock();
}

int main()
{
    int result = PAPI_library_init(PAPI_VER_CURRENT);
    if (result != PAPI_VER_CURRENT && result > 0)
        cout << "Couldn't initialize PAPI library. (Error " << result << ")" << endl;

    if (PAPI_thread_init(nextThreadId) != PAPI_OK)
        cout << "Couldn't initialize PAPI's threading support." << endl;

    std::thread t1(hello), t2(hello);
    t1.join();
    t2.join();

    return 0;
}

