#include <cstdio> 

// parser includes
#include "enums.h"
#include "dump.h"
#include "file_helper.h"
#include "node_factory.h"
#include "parser.h"
#include "utilities.h"

using namespace ts;

int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        puts("Usage: print-tester <file path>.ts");
        return 0;
    }

    std::string filePath(argv[1]);

    auto content = readFile(filePath);

    Parser parser;
    auto sourceFile = parser.parseSourceFile(stows(filePath), content, ScriptTarget::Latest);

    stringstream out;
    Printer printer(out);
    //printer.setDeclarationMode(true);
    printer.printNode(sourceFile);

    puts(wstos(out.str()).c_str());

    return 0;
}