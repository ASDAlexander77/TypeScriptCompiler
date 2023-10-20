#include <array>
#include <codecvt>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "file_helper.h"
#include "parser.h"
#include "utilities.h"
#include "dump.h"

using namespace ts;

void printParser(const wchar_t *fileName, const wchar_t *str, boolean showLineCharPos)
{
    ts::Parser parser;
    // auto sourceFile = parser.parseSourceFile(S("function f() { let i = 10; }"), ScriptTarget::Latest);
    auto sourceFile = parser.parseSourceFile(fileName, str, ScriptTarget::Latest);

    ts::FuncT<> visitNode;
    ts::ArrayFuncT<> visitArray;

    auto indent = 0;

    visitNode = [&](ts::Node child) -> ts::Node {
        std::cout << std::string(indent, '\t').c_str();

        if (showLineCharPos)
        {
            auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->pos);
            auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->_end);

            std::cout << "Node: " << wtoc(parser.syntaxKindString(child).c_str()) << " @ [ " << child->pos << "("
                      << posLineChar.line + 1 << ":" << posLineChar.character + 1 << ") - " << child->_end << "("
                      << endLineChar.line + 1 << ":" << endLineChar.character << ") ]" << std::endl;
        }
        else
        {
            std::cout << "Node: " << wtoc(parser.syntaxKindString(child).c_str()) << " @ [ " << child->pos << " - "
                      << child->_end << " ]" << std::endl;
        }

        indent++;
        ts::forEachChild(child, visitNode, visitArray);
        indent--;

        return undefined;
    };

    visitArray = [&](ts::NodeArray<ts::Node> array) -> ts::Node {
        for (auto node : array)
        {
            visitNode(node);
        }

        return undefined;
    };

    auto result = ts::forEachChild(sourceFile.as<ts::Node>(), visitNode, visitArray);
}

void print(const wchar_t *fileName, const wchar_t *str, boolean showLineCharPos)
{
    ts::Parser parser;
    // auto sourceFile = parser.parseSourceFile(S("function f() { let i = 10; }"), ScriptTarget::Latest);
    auto sourceFile = parser.parseSourceFile(fileName, str, ScriptTarget::Latest);

    print(sourceFile);    
}

boolean hasOption(int argc, char **args, const char *option)
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

char *firstNonOption(int argc, char **args)
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
        auto hasLine = hasOption(argc, args, "--line");
        auto hasSource = hasOption(argc, args, "--source");

        auto file = firstNonOption(argc, args);
        auto exists = file != nullptr && fs::exists(file);
        if (exists)
        {
            auto str = readFile(std::string(file));
            if (hasSource)
            {
                print(ctow(file).c_str(), str.c_str(), hasLine);
            }
            else
            {
                printParser(ctow(file).c_str(), str.c_str(), hasLine);
            }
        }
        else
        {
            if (hasSource)
            {
                print(S(""), ctow(file).c_str(), hasLine);
            }
            else
            {
                printParser(S(""), ctow(file).c_str(), hasLine);
            }
        }
    }

    return 0;
}