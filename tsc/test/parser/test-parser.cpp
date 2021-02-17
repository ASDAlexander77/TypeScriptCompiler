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

int main(int, char **)
{
    try
    {
        testCallExpr();
        testFunctionDecl();

        printParse("function main() { f1(); } function f1(a = 10) { return a; }");
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }
}