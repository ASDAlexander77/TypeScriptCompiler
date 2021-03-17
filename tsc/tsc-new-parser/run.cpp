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

#include "scanner.h"

void printScanner(const wchar_t *str)
{
    ts::Scanner scanner(ScriptTarget::Latest, true, LanguageVariant::Standard, str);

    auto token = SyntaxKind::Unknown;
    while (token != SyntaxKind::EndOfFileToken)
    {
        token = scanner.scan();
        //std::wcout << scanner.syntaxKindString(token) << "(" << (int)token << S(") @") << scanner.getTokenPos() << S(" '") << scanner.tokenToString(token) << "':" << scanner.getTokenText() << std::endl;
        //std::wcout << (number)token << S(" ") << scanner.getTokenText() << std::endl;
        wprintf(S("%d %s\n"), (number)token, scanner.getTokenText().c_str());
    }
}

auto wtoc(char *in)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    auto wide = converter.from_bytes(in);
    return wide;
}

enum Encoding
{
    ENCODING_ASCII,
    ENCODING_UTF16LE,
    ENCODING_UTF16BE,
    ENCODING_UTF8
};

int readEncoding(std::string file)
{
    std::string result;
    std::ifstream ifs(file.c_str(), std::ios::binary);
    int encoding = ENCODING_ASCII;

    if (!ifs.is_open())
    {
        return encoding;
    }
    else if (ifs.eof())
    {
        return encoding;
    }
    else
    {
        int ch1 = ifs.get();
        int ch2 = ifs.get();
        if (ch1 == 0xff && ch2 == 0xfe)
        {
            // The file contains UTF-16LE BOM
            encoding = ENCODING_UTF16LE;
        }
        else if (ch1 == 0xfe && ch2 == 0xff)
        {
            // The file contains UTF-16BE BOM
            encoding = ENCODING_UTF16BE;
        }
        else
        {
            int ch3 = ifs.get();
            if (ch1 == 0xef && ch2 == 0xbb && ch3 == 0xbf)
            {
                // The file contains UTF-8 BOM
                encoding = ENCODING_UTF8;
            }
            else
            {
                // The file does not have BOM
                encoding = ENCODING_ASCII;
            }
        }
    }

    return encoding;
}



string readFile(std::string file)
{
    auto enc = readEncoding(file);

    std::wifstream f(file);

    if (enc == ENCODING_UTF8)
    {
        typedef std::codecvt_utf8_utf16<wchar_t, 0x10ffff, std::codecvt_mode::consume_header> conv;
        std::locale loc(f.getloc(), new conv());
        f.imbue(loc);
    }

    if (enc == ENCODING_UTF16BE)
    {
        typedef std::codecvt_utf16<wchar_t, 0x10ffff, std::consume_header> conv16be;
        std::locale loc16be(f.getloc(), new conv16be());
        f.imbue(loc16be);
    }

    if (enc == ENCODING_UTF16LE)
    {
        typedef std::codecvt_utf16<wchar_t, 0x10ffff, (std::codecvt_mode)(std::consume_header|std::little_endian)> conv16be;
        std::locale loc16be(f.getloc(), new conv16be());
        f.imbue(loc16be);
    }

    string str((std::istreambuf_iterator<char_t>(f)), std::istreambuf_iterator<char_t>());
    return str;
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
            printScanner(wtoc(args[1]).c_str());
        }
    }

    return 0;
}