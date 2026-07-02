#ifndef CONFIG_H
#define CONFIG_H

#include <cctype>
#include <functional>
#include <regex>
#include <sstream>
#include <string>

#if _MSC_VER
#pragma warning(disable : 4062)
#pragma warning(disable : 4834)
#pragma warning(disable : 4996)
#endif

using boolean = bool;
using number = int;
using string = std::wstring;
using char_t = wchar_t;
using sstream = std::wstringstream;
using regex = std::wregex;
using sregex_iterator = std::wsregex_iterator;
using stringstream = std::wstringstream;
using smatch = std::wsmatch;
using ostream = std::wostream;
#define regex_replace std::regex_replace

#define to_number_base(x, y) std::stoi(x, nullptr, y)
#define _S(x) (L##x)
#define S(x) _S(x)
#define to_string_val(x) std::to_wstring(x)
#define to_bignumber_base(x, y) std::stoull(x, nullptr, y)

#define _E(x) data::DiagnosticMessage(x)

#endif // CONFIG_H