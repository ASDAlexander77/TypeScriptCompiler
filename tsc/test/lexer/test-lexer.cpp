#include "helper.h"

#include <vector>

#include "TypeScriptLexerANTLR.h"
#include "TypeScriptParserANTLR.h"

#define T testToken

using l = typescript::TypeScriptLexerANTLR;

void testBasic()
{
    auto foo = 2.0;
    auto bar = 1.0;

    ASSERT_THROW(foo != bar);
    ASSERT_EQUAL(foo, 2.0);
    ASSERT_EQUAL(bar, 1.0);
}

void printTokens(const char *value) 
{
    antlr4::ANTLRInputStream input(value);
    typescript::TypeScriptLexerANTLR lexer(&input);
    auto tokens = lexer.getAllTokens();

    auto print = [](const auto& n) 
    { 
        auto* tokenPtr = n.get();
        std::cout << "TOKEN: type=" << tokenPtr->getType() << " text=" << tokenPtr->getText() << std::endl; 
    };

    std::for_each(tokens.cbegin(), tokens.cend(), print);
}

void testToken(const char *value, size_t tokenExpected)
{
    antlr4::ANTLRInputStream input(value);
    typescript::TypeScriptLexerANTLR lexer(&input);

    auto token = lexer.nextToken();
    //ASSERT_EQUAL(token.get()->getTokenIndex(), tokenExpected);
    ASSERT_EQUAL(token.get()->getType(), tokenExpected);
    ASSERT_THROW(token.get()->getText() != value);
}

void testLexer()
{
    printTokens("123 456\r\n789");
    //T("123", l::NumericLiteral);
}

int main(int, char **)
{
    testLexer();
}