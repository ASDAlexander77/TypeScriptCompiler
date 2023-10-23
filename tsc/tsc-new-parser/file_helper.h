
#ifndef FILE_HELPER_H_
#define FILE_HELPER_H_

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

#include "config.h"

enum Encoding
{
    ENCODING_ASCII,
    ENCODING_UTF16LE,
    ENCODING_UTF16BE,
    ENCODING_UTF8
};

static auto ctow(const char *in)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    auto wide = converter.from_bytes(in);
    return wide;
}

static auto wtoc(const wchar_t *in)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    auto chars = converter.to_bytes(in);
    return chars;
}

static std::wstring stows(const std::string &s)
{
    std::wstring ws(ctow(s.c_str()));
    return ws;
}

static std::string wstos(const std::wstring &ws)
{
    std::string s(wtoc(ws.c_str()));
    return s;
}

static int readEncoding(std::string file)
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

static std::wstring readFile(std::string file)
{
    auto enc = readEncoding(file);

    std::wifstream f(file);
    std::locale fileLocale;

    if (enc == ENCODING_UTF8)
    {
        typedef std::codecvt_utf8_utf16<wchar_t, 0x10ffff, std::codecvt_mode::consume_header> conv;
        fileLocale = std::locale(f.getloc(), new conv());
    }
    else if (enc == ENCODING_UTF16BE)
    {
        typedef std::codecvt_utf16<wchar_t, 0x10ffff, std::consume_header> conv16be;
        fileLocale = std::locale(f.getloc(), new conv16be());
    }
    else if (enc == ENCODING_UTF16LE)
    {
        typedef std::codecvt_utf16<wchar_t, 0x10ffff, (std::codecvt_mode)(std::consume_header | std::little_endian)> conv16le;
        fileLocale = std::locale(f.getloc(), new conv16le());
    }

    f.imbue(fileLocale);

    std::wstringstream wss;
    wss << f.rdbuf();
    f.close();
    return wss.str();    
}

#endif // FILE_HELPER_H_