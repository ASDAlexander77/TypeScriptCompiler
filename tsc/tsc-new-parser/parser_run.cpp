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

void printParser(const wchar_t *str, boolean showLineCharPos)
{
    ts::Parser parser;
    //auto sourceFile = parser.parseSourceFile(S("function f() { let i = 10; }"), ScriptTarget::Latest);
    auto sourceFile = parser.parseSourceFile(str, ScriptTarget::Latest);

    FuncT<> visitNode;
    ArrayFuncT<> visitArray;

    auto intent = 0;

    visitNode = [&](Node child) -> Node {

        for (auto i = 0; i < intent; i++)
        {
            std::cout << "\t";
        }

        if (showLineCharPos)
        {
            auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->pos);
            auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->_end);

            std::cout 
                << "Node: " 
                << wtoc(parser.syntaxKindString(child).c_str()) 
                << " @ [ " << child->pos << "(" << posLineChar.line + 1 << ":" << posLineChar.character  << ") - " 
                << child->_end << "(" << endLineChar.line + 1  << ":" << endLineChar.character  << ") ]" << std::endl;
        }
        else
        {
            std::cout << "Node: " << wtoc(parser.syntaxKindString(child).c_str()) << " @ [ " << child->pos << " - " << child->_end << " ]" << std::endl;
        }

        intent++;
        ts::forEachChild(child, visitNode, visitArray);    
        intent--;

        return undefined;
    };

    visitArray = [&](NodeArray<Node> array) -> Node {
        for (auto node : array)
        {
            visitNode(node);
        }

        return undefined;
    };

    auto result = ts::forEachChild(sourceFile.as<Node>(), visitNode, visitArray);
}

boolean hasOption(int argc, char **args, const char* option)
{
    for (auto i = 1; i < argc; i++)
    {
        if (std::strlen(args[i]) < 3)
        {
            continue;
        }

        if (std::strcmp(option, args[i]) == 0)
        {
            return true;
        }
    }

    return false;
}

char* firstNonOption(int argc, char **args)
{
    for (auto i = 1; i < argc; i++)
    {
        if (std::strlen(args[i]) < 3)
        {
            return args[i];
        }

        if (args[i][0] != '-' || args[i][1] != '-')
        {
            return args[i];
        }
    }

    return nullptr;
}

int main(int argc, char **args)
{
    if (argc > 1)
    {
        auto file = firstNonOption(argc, args);
        auto exists = file != nullptr && fs::exists(file);
        if (exists)
        {
            auto str = readFile(std::string(file));
            printParser(str.c_str(), hasOption(argc, args, "--line"));
        }
        else
        {
            printParser(ctow(file).c_str(), hasOption(argc, args, "--line"));
        }
    }

    return 0;
}