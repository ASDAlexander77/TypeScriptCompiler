#include "helper.h"

#include <vector>
#include <sstream>

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

int main(int, char **)
{
    try
    {
        printParse("hello(1);");
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }
}