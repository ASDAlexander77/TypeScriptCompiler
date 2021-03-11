#include "helper.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>

#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "TypeScriptLexerANTLR.h"
#include "TypeScriptParserANTLR.h"

#define T testParse

using l = typescript::TypeScriptLexerANTLR;
using p = typescript::TypeScriptParserANTLR;

void printParse(const char *value)
{
    antlr4::ANTLRInputStream input(value);
    typescript::TypeScriptLexerANTLR lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    typescript::TypeScriptParserANTLR parser(&tokens);    
    auto* main = parser.main();    

    auto s = main->toStringTree(&parser);

    std::cout << s << std::endl;
}

void testParse(const char *value)
{
    antlr4::ANTLRInputStream input(value);
    typescript::TypeScriptLexerANTLR lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    typescript::TypeScriptParserANTLR parser(&tokens);    
    parser.main();    
} 

void testCallExpr()
{
    testParse("function main() { hello(1); }");
}

void testFunctionDecl()
{
    testParse("function defaultArgs1() {}");
    testParse("function defaultArgs2(x: number) {}");
    testParse("function defaultArgs3(y = 3) {}");
    testParse("function defaultArgs3(x: number, y = 3) {}");
}

void testVarDecl()
{
    testParse("function main() { const a = 10; }");
}

int main(int argc, char **args)
{
    try
    {
        if (argc > 1)
        {
            auto file = args[1];
            auto exists = fs::exists(file);
            if (exists)
            {
                std::ifstream f(file);
                std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                std::cout << "Code: " << std::endl << str << std::endl << "Output: " << std::endl;
                printParse(str.c_str());
            }
            else
            {
                std::cout << "Code: " << std::endl << args[1] << std::endl << "Output: " << std::endl;
                printParse(args[1]);
            }
        }
        else
        {
            testCallExpr();
            testFunctionDecl();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }
}