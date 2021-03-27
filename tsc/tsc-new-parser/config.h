#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <regex>
#include <functional>

#pragma warning( disable : 4062 )

using boolean = bool;
using number = int;
using string = std::wstring;
using char_t = wchar_t;
using sstream = std::wstringstream;
using regex = std::wregex;
using sregex_iterator = std::wsregex_iterator;
#define regex_replace std::regex_replace

#define to_number_base(x, y) std::stoi(x, nullptr, y)
#define S(x) L##x
#define to_string(x) std::to_wstring(x)
#define to_number(x) std::stoi(x)
#define to_integer(x) std::stoi(x)
#define to_float(x) std::stod(x)
#define to_bignumber_base(x, y) std::stoll(x, nullptr, y)
#define to_bignumber(x) std::stoll(x)

#endif // CONFIG_H