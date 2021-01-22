#include "helper.h"

#include <vector>
#include <sstream>

#include "TypeScriptLexerANTLR.h"
#include "TypeScriptParserANTLR.h"

#define T testToken

using l = typescript::TypeScriptLexerANTLR;

void printTokens(const typescript::TypeScriptLexerANTLR& lexer, const std::vector<std::unique_ptr<antlr4::Token>>& tokens) 
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

void printTokens(const typescript::TypeScriptLexerANTLR& lexer) 
{
    printTokens(lexer, const_cast<typescript::TypeScriptLexerANTLR&>(lexer).getAllTokens());
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

void testNullLiteral()
{
    T("null", l::NULL_KEYWORD);
}

void testBooleanLiteral()
{
    T("true", l::TRUE_KEYWORD);
    T("false", l::FALSE_KEYWORD);
}

void testBinaryIntegerLiteral()
{
    T("0b11010", l::BinaryIntegerLiteral);
    T("0B11010", l::BinaryIntegerLiteral);
    T("0b11010", l::BinaryIntegerLiteral);
    T("0b1_1010", l::BinaryIntegerLiteral);
    T("0B11_010", l::BinaryIntegerLiteral);
    T("0b110_10", l::BinaryIntegerLiteral);
    T("0B11111111111111111111111111111111111111111111111101001010100000010111110001111111111", l::BinaryIntegerLiteral);
    T("0B111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111101001010100000010111110001111111111", l::BinaryIntegerLiteral);
}

void testBinaryBigIntegerLiteral()
{
    T("0b11010n", l::BinaryBigIntegerLiteral);
}

void testOctalIntegerLiteral()
{
    T("01", l::OctalIntegerLiteral);
    T("0123", l::OctalIntegerLiteral);
    T("045436", l::OctalIntegerLiteral);
    T("0o45436", l::OctalIntegerLiteral);
    T("0O45436", l::OctalIntegerLiteral);
    T("04_5436", l::OctalIntegerLiteral);
    T("0o45_436", l::OctalIntegerLiteral);
    T("0O4543_6", l::OctalIntegerLiteral);    
    T("0o7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777", l::OctalIntegerLiteral);
    T("0o7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777", l::OctalIntegerLiteral);
}

void testHexIntegerLiteral()
{
    T("0x4abcdef", l::HexIntegerLiteral);
    T("0X4abcdef", l::HexIntegerLiteral);
    T("0x4abc_def", l::HexIntegerLiteral);
    T("0X4ab_cdef", l::HexIntegerLiteral);
    T("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", l::HexIntegerLiteral);
    T("0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", l::HexIntegerLiteral);
}

void testHexBigIntegerLiteral()
{
    T("0x4abcdefn", l::HexBigIntegerLiteral);
}

void testDecimalIntegerLiteral()
{
    T("0", l::DecimalIntegerLiteral);
    T("1", l::DecimalIntegerLiteral);
    T("0123888", l::DecimalIntegerLiteral);
    T("0123_888", l::DecimalIntegerLiteral);
    T("123", l::DecimalIntegerLiteral);
    T("-03", l::DecimalIntegerLiteral);
    T("009", l::DecimalIntegerLiteral);    
}

void testDecimalLiteral()
{
    T("0.0", l::DecimalLiteral);
    T("1.0", l::DecimalLiteral);
    T("01.0", l::DecimalLiteral);
    T("123.0", l::DecimalLiteral);
    T("123_4.0", l::DecimalLiteral);
    T("123e1", l::DecimalLiteral);
    T("1e0", l::DecimalLiteral);
    T("1e+0", l::DecimalLiteral);
}

void testDecimalBigIntegerLiteral()
{
    T("0n", l::DecimalBigIntegerLiteral);
    T("1n", l::DecimalBigIntegerLiteral);
    T("123n", l::DecimalBigIntegerLiteral);
}

void testOctalBigIntegerLiteral()
{
    T("0123n", l::OctalBigIntegerLiteral);
}

void testRegex()
{
    T("/ asdf /", l::RegularExpressionLiteral);
    T("/**// asdf /", { l::MultiLineComment, l::RegularExpressionLiteral });
    T("/**///**/ asdf /       // should be a comment line\r\n1", { l::MultiLineComment, l::SingleLineComment, l::DecimalIntegerLiteral });
    T("/**// /**/asdf /", { l::MultiLineComment, l::RegularExpressionLiteral, l::ASTERISKASTERISK_TOKEN, l::RegularExpressionLiteral });// /**/ comment, regex (/ /) power(**) regex(/ asdf /)
    T("/**// asdf/**/ /", { l::MultiLineComment, l::RegularExpressionLiteral, l::ASTERISKASTERISK_TOKEN, l::RegularExpressionLiteral }); 
    T("/(?:)/", l::RegularExpressionLiteral); // empty regular expression
    T("/what/", l::RegularExpressionLiteral);
    T("/\\\\\\\\/", l::RegularExpressionLiteral);
}

void testIdentifier()
{
    T("\\u0061wait", l::IdentifierName);
    T("\\u0079ield", l::IdentifierName);
    T("\\u{0076}ar", l::IdentifierName);
    T("\\u{0079}ield", l::IdentifierName);
    T("typ\\u{0065}", l::IdentifierName);
    T("def\\u0061ult", l::IdentifierName);
}

void testString()
{
    T("''", l::StringLiteral);
    T("\"\"", l::StringLiteral);
    T("'foo\\\r\nbar'", l::StringLiteral);
    T("\"foo\\\r\nbar\"", l::StringLiteral);
    T("'aaa//aaa'", l::StringLiteral);
}

void testTemplateString()
{
    T("` string template`", l::StringLiteral);
    T("`foo ${a}`", { l::TemplateHead, l::IdentifierName, l::TemplateTail });
    T("`\\u0061wait`", l::StringLiteral);
    T("`\\u0079ield`", l::StringLiteral);
    T("`\\u{0076}ar`", l::StringLiteral);
    T("`\\u{0079}ield`", l::StringLiteral);
    T("`typ\\u{0065}`", l::StringLiteral);
    T("`def\\u0061ult`", l::StringLiteral);
}

void testSingleLineComments()
{
    T("//", l::SingleLineComment);
    T("// test", l::SingleLineComment);
    T("// /* test */", l::SingleLineComment);
}

void testMultiLineComments()
{
    T("/**/", l::MultiLineComment);
    T("/* test  */", l::MultiLineComment);
    T("/* //test */", l::MultiLineComment);
    T("/**\r\n * comment\r\n */", l::MultiLineComment);
}

int main(int, char **)
{
    try
    {
        testNullLiteral();
        testBooleanLiteral();
        testBinaryIntegerLiteral();
        testBinaryBigIntegerLiteral();
        testOctalIntegerLiteral();
        testOctalBigIntegerLiteral();
        testHexIntegerLiteral();
        testHexBigIntegerLiteral();
        testDecimalLiteral();
        testDecimalBigIntegerLiteral();
        testIdentifier();
        testString();
        testTemplateString();
        testSingleLineComments();
        testMultiLineComments();
        testRegex();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }
}