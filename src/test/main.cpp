#include <ext/stdio_filebuf.h>
#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    std::cout << "Hello World!" << std::endl;
    FILE* x = popen("echo 'fuuuu bar'", "r");

    __gnu_cxx::stdio_filebuf<char> filebuf(x, std::ios::in);
    istream is(&filebuf);

    std::cout << is.rdbuf();
}