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
#include <io.h>

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

void printScanner(const wchar_t *str)
{
    Scanner scanner(ScriptTarget::Latest, true, LanguageVariant::Standard, str);

    auto token = SyntaxKind::Unknown;
    while (token != SyntaxKind::EndOfFileToken)
    {
        token = scanner.scan();
        //std::wcout << scanner.syntaxKindString(token) << "(" << (int)token << S(") @") << scanner.getTokenPos() << S(" '") << scanner.tokenToString(token) << "':" << scanner.getTokenText() << std::endl;
        //std::wcout << (number)token << S(" ") << scanner.getTokenText() << std::endl;
        wprintf_s(S("%d %ls\n"), (number)token, scanner.getTokenText().c_str());
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