#include <iostream>
#include <cuda.h>
#include <papi.h>
using namespace std;

__global__ void
kernel(char* output)
{
    output[threadIdx.x] = '1';
}

string invokeKernel(char input[])
{
    string result;
    size_t size = sizeof(input);
    char* output;
    cudaMalloc((void**) &output, size);
    cudaMemcpy(output, input, size, cudaMemcpyHostToDevice);
    kernel<<<32, 32>>>(output);
    cudaMemcpy(input, output, size, cudaMemcpyDeviceToHost);
    result = string(input);
    cudaFree(output);
    return result;
}

int main()
{
    int eventSet = PAPI_NULL;
    long long eventCounts[1];
    char nativeEventName[] = "CUDA:::Quadro_4000:domain_d:active_cycles";
    int eventCode;
    char input[] = "12345678901234567890123456789012";
    string output;
    
    int result = PAPI_library_init(PAPI_VER_CURRENT);
    if (result != PAPI_VER_CURRENT && result > 0)
        cout << "Couldn't initialize PAPI library. (Error " << result << ")" << endl;

    if (PAPI_create_eventset(&eventSet) != PAPI_OK)
        cout << "Couldn't create event set" << endl;

    // Since CUDA events are all native and no preset events, generate code
    if (PAPI_event_name_to_code(nativeEventName, &eventCode) != PAPI_OK)
        cout << "Couldn't generate CUDA event code." << endl;

    if (PAPI_add_event(eventSet, eventCode) != PAPI_OK)
        cout << "Couldn't add event to set of thread." << endl;
    
    if (PAPI_start(eventSet) != PAPI_OK)
        cout << "Couldn't start measurement for thread." << endl;
    
    output = invokeKernel(input);

    if (PAPI_stop(eventSet, eventCounts) != PAPI_OK)
        cout << "Couldn't stop measurement for thread." << endl;

    cout << "Input:\t\t" << input
         << "\nOutput:\t\t" << output
         << "\nExpected:\t11111111111111111111111111111111\n"
         << nativeEventName << " = " << eventCounts[0] << endl;

    return 0;
}
