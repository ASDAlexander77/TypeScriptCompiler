#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <locale>
#include <codecvt>
#include <string>
#include <io.h>

#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "parser.h"
#include "utilities.h"
#include "file_helper.h"

void printParser(const wchar_t *str)
{
    // TODO:
    ts::Parser parser;
    auto sourceFile = parser.parseSourceFile(S("function f() { let i = 10; }"), ScriptTarget::Latest);
    //auto sourceFile = parser.parseSourceFile(str, ScriptTarget::Latest);

    auto visit = [&](Node child) {

        std::cout << "Node: " << wtoc(parser.syntaxKindString(child->kind).c_str()) << " at [" << child->pos << "-" << child->_end << "]" << std::endl;

        return undefined;
    };

    auto result = ts::forEachChild<Node>(sourceFile, visit);
}

int main(int argc, char **args)
{
    if (argc > 1)
    {
        auto file = args[1];
        auto exists = fs::exists(file);
        if (exists)
        {
            auto str = readFile(std::string(file));
            printParser(str.c_str());
        }
        else
        {
            printParser(ctow(args[1]).c_str());
        }
    }

    return 0;
}