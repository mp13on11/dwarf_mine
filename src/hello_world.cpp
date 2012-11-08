#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    vector<string> arguments(argv + 1, argv + argc);
    
    if (arguments.empty())
        arguments.push_back("World");
    
    for (const string& arg : arguments)
        cout << "Hello " << arg << "!" << endl;
}
