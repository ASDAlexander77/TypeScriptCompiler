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
    std::cout << "Printing tokens:" << std::endl;

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
    stringStream << "Expecting: [" << lexer.getTokenNames()[tokenExpected] << "] \"" << value << "\" but get: [" << lexer.getTokenNames()[token->getType()] << "] \"" << token->getText() << "\".";
    auto msg = stringStream.str();    

    ASSERT_EQUAL_MSG(token->getType(), tokenExpected, msg);
    ASSERT_THROW_MSG(token->getText().compare(value) == 0, msg);
}

void testToken(const char *value, std::vector<size_t> tokensExpected)
{
    antlr4::ANTLRInputStream input(value);
    typescript::TypeScriptLexerANTLR lexer(&input);

    auto tokens = lexer.getAllTokens();

    printTokens(lexer, tokens);

    auto index = 0;    
    for (auto& token : tokens) {
        auto tokenExpected = tokensExpected[index];

        std::ostringstream stringStream;
        stringStream << "Expecting: [" << lexer.getTokenNames()[tokenExpected] << "] in \"" << value << "\" @ " << index << " but get: [" << lexer.getTokenNames()[token->getType()] << "] \"" << token->getText() << "\".";
        auto msg = stringStream.str();    

        ASSERT_EQUAL_MSG(token->getType(), tokenExpected, msg);
        // can't match text
        //ASSERT_THROW_MSG(token->getText().compare(value) == 0, msg);

        index++;
    }
}

void testLexer()
{
    std::cout << "[ Numeric ]" << std::endl;

    T("123", l::NumericLiteral);
}

void testRegex()
{
    std::cout << "[ Regex ]" << std::endl;

    T("/ asdf /", l::RegularExpressionLiteral);
    T("/**// asdf /", { l::MultiLineComment, l::RegularExpressionLiteral });
    T("/**///**/ asdf /       // should be a comment line\r\n1", { l::MultiLineComment, l::SingleLineComment, l::LineTerminatorSequence, l::NumericLiteral });
    T("/**// /**/asdf /", { l::MultiLineComment, l::RegularExpressionLiteral, l::ASTERISKASTERISK_TOKEN, l::RegularExpressionLiteral });// /**/ comment, regex (/ /) power(**) regex(/ asdf /)
    T("/**// asdf/**/ /", { l::MultiLineComment, l::RegularExpressionLiteral, l::ASTERISKASTERISK_TOKEN, l::RegularExpressionLiteral }); 
    T("/(?:)/", l::RegularExpressionLiteral); // empty regular expression
}

int main(int, char **)
{
    try
    {
        testLexer();
        testRegex();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        throw;
    }
}