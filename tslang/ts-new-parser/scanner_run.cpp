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
#include "scanner.h"

using namespace ts;

string getSafeToken(SyntaxKind token, Scanner scanner)
{
    if (token == SyntaxKind::NoSubstitutionTemplateLiteral)
    {
        return S("");
    }

    return scanner.getTokenText().c_str();
}

void printScanner(const wchar_t *str)
{
    Scanner scanner(ScriptTarget::Latest, true, LanguageVariant::Standard, str);

    auto token = SyntaxKind::Unknown;
    while (token != SyntaxKind::EndOfFileToken)
    {
        token = scanner.scan();
        // std::wcout << scanner.syntaxKindString(token) << "(" << (int)token << S(") @") << scanner.getTokenPos() << S(" '") <<
        // scanner.tokenToString(token) << "':" << scanner.getTokenText() << std::endl; std::wcout << (number)token << S(" ") <<
        // scanner.getTokenText() << std::endl;
        std::wcout << (number)token << " " << getSafeToken(token, scanner) << "$$$" << std::endl;
    }
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
            printScanner(str.c_str());
        }
        else
        {
            printScanner(ctow(args[1]).c_str());
        }
    }

    return 0;
}