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

#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "scanner.h"

void printScanner(const wchar_t* str)
{
    ts::Scanner scanner(ScriptTarget::Latest, true, LanguageVariant::Standard, str);

    auto token = SyntaxKind::Unknown;
    while (token != SyntaxKind::EndOfFileToken)
    {
        token = scanner.scan();
        //std::wcout << scanner.syntaxKindString(token) << "(" << (int)token << S(") @") << scanner.getTokenPos() << S(" '") << scanner.tokenToString(token) << "':" << scanner.getTokenText() << std::endl;
        std::wcout << (number)token << S(" @ ") << scanner.getTokenPos() << " " << scanner.getTokenText() << std::endl;
    }
}

auto wtoc(char* in)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    auto wide = converter.from_bytes(in);
    return wide;
}

int main(int argc, char** args)
{
    if (argc > 1)
    {
        auto file = args[1];
        auto exists = fs::exists(file);
        if (exists)
        {
            std::wifstream f(file);
            string str((std::istreambuf_iterator<char_t>(f)), std::istreambuf_iterator<char_t>());
            printScanner(str.c_str());
        }
        else
        {
            printScanner(wtoc(args[1]).c_str());
        }
    }    

    return 0;
}