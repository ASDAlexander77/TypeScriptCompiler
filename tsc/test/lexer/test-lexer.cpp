#include "helper.h"

#include <vector>
#include <sstream>

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

void printTokens(typescript::TypeScriptLexerANTLR& lexer, std::vector<std::unique_ptr<antlr4::Token>>& tokens) 
{
    auto print = [&](const auto& n) 
    { 
        //auto* tokenPtr = n.get();
        //std::cout << "TOKEN: type=" << tokenPtr->getType() << " text=" << tokenPtr->getText() << std::endl; 
        std::cout << lexer.getTokenNames()[n.get()->getType()] << ": " << n.get()->getText() << " ..." << n.get()->toString() << std::endl ; 
    };

    std::for_each(tokens.cbegin(), tokens.cend(), print);
}

void printTokens(typescript::TypeScriptLexerANTLR& lexer) 
{
    printTokens(lexer, lexer.getAllTokens());
}

void printTokens(const char *value) 
{
    antlr4::ANTLRInputStream input(value);
    typescript::TypeScriptLexerANTLR lexer(&input);
    printTokens(lexer);
}

void testToken(const char *value, size_t tokenExpected)
{
    antlr4::ANTLRInputStream input(value);
    typescript::TypeScriptLexerANTLR lexer(&input);

    auto tokens = lexer.getAllTokens();

    printTokens(lexer, tokens);

    auto token = tokens.front().get();

    std::ostringstream stringStream;
    stringStream << "Expecting token:" << lexer.getTokenNames()[tokenExpected] << "text: " << value << " but get token:" << lexer.getTokenNames()[token->getType()] << " Text:" << token->getText();
    auto msg = stringStream.str();    

    ASSERT_EQUAL_MSG(token->getType(), tokenExpected, msg);
    ASSERT_THROW_MSG(token->getText().compare(value) == 0, msg);
}

void testLexer()
{
    T("123a", l::NumericLiteral);
}

int main(int, char **)
{
    testLexer();
}