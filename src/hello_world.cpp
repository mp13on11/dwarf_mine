#include <iostream>
#include <string>
#include <vector>
#include <thread>

using namespace std;

int main(int argc, char** argv) {
    vector<string> arguments(argv + 1, argv + argc);
    
    if (arguments.empty())
        arguments.push_back("World");
    
    vector<thread> greeters;
    
    for (const string& arg : arguments) {
        greeters.emplace_back([arg]() {
            cout << "Hello " << arg << "!" << endl;
        });
    }
    
    for (thread& greeter : greeters)
        greeter.join();
}
