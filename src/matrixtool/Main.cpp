#include <iostream>
#include <vector>
#include <matrix/MatrixHelper>

using namespace std;

void printUsage()
{
    cout << "Usage: matrixtool <command> [<args>]" << endl;
    exit(1);
}

void verifyMatrices(const vector<string>& args)
{
    if (args.size() < 2)
        printUsage();

    auto left = args[0];
    auto right = args[1];


}

void generateMatrices(const vector<string>& args)
{

}

map<string, function<void(const vector<string>&)>> subCommands =
{
    { "verify", [](const vector<string>& args)
        {
            verifyMatrices(args);
        }
    },
    { "generate", [](const vector<string>& args)
        {
            generateMatrices(args);
        }
    }
};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printUsage();
    }

    vector<string> arguments(argv+2, argv+argc);

    subCommands[argv[1]](arguments);

    return 0;
}
