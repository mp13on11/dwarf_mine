#include <iostream>
#include <cuda.h>
#include <papi.h>
using namespace std;

__global__ void
kernel(char* output)
{
    output[threadIdx.x] = 0;
}

int invokeKernel()
{
    int result = 0;
    char input[] = "01234567890123456789012345678901";
    size_t size = sizeof(input);
    char* output;
    cudaMalloc((void**) &output, size);
    cudaMemcpy(output, input, size, cudaMemcpyHostToDevice);
    kernel<<<32, 32>>>(output);
    cudaMemcpy(input, output, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size - 1; ++i)
        result += (int) input[i];
    cudaFree(output);
    return result;
}

int main()
{
    int eventSet = PAPI_NULL;
    long long eventCounts[1];
    int output;
    
    int result = PAPI_library_init(PAPI_VER_CURRENT);
    if (result != PAPI_VER_CURRENT && result > 0)
        cout << "Couldn't initialize PAPI library. (Error " << result << ")" << endl;

    if (PAPI_create_eventset(&eventSet) != PAPI_OK)
        cout << "Couldn't create event set for thread." << endl;

    if (PAPI_add_event(eventSet, PAPI_TOT_INS) != PAPI_OK)
        cout << "Couldn't add event to set of thread." << endl;
    
    if (PAPI_start(eventSet) != PAPI_OK)
        cout << "Couldn't start measurement for thread." << endl;
    
    output = invokeKernel();

    if (PAPI_stop(eventSet, eventCounts) != PAPI_OK)
        cout << "Couldn't stop measurement for thread." << endl;

    cout << "Output: " << output << endl;

    return 0;
}
