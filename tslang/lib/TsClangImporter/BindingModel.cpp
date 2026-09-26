#include "TsClangImporter/BindingModel.h"

#include <algorithm>

namespace tsbindgen
{

TsType TsType::builtin(std::string name, std::string comment)
{
    TsType type;
    type.kind = Kind::Builtin;
    type.name = std::move(name);
    type.comment = std::move(comment);
    return type;
}

TsType TsType::named(std::string key)
{
    TsType type;
    type.kind = Kind::Named;
    type.name = std::move(key);
    return type;
}

TsType TsType::reference(TsType pointee)
{
    TsType type;
    type.kind = Kind::Reference;
    type.pointee.push_back(std::move(pointee));
    return type;
}

TsType TsType::skip(std::string reason)
{
    TsType type;
    type.skipReason = std::move(reason);
    return type;
}

std::string renderType(const TsType &type, const std::function<std::string(const std::string &)> &nameOf)
{
    switch (type.kind)
    {
    case TsType::Kind::Builtin:
        return type.name;
    case TsType::Kind::Named:
        return nameOf(type.name);
    case TsType::Kind::Reference:
        return "Reference<" + renderType(type.pointee.front(), nameOf) + ">";
    }

    return type.name;
}

void collectKeys(const TsType &type, std::vector<std::string> &keys)
{
    if (type.kind == TsType::Kind::Named && std::find(keys.begin(), keys.end(), type.name) == keys.end())
    {
        keys.push_back(type.name);
    }

    for (auto &pointee : type.pointee)
    {
        collectKeys(pointee, keys);
    }
}

std::string nameFromKey(const std::string &key)
{
    auto colon = key.find(':');
    return colon == std::string::npos ? key : key.substr(colon + 1);
}

std::string escapeReserved(std::string name)
{
    // ECMAScript reserved words, including the strict-mode ones
    static const char *const reserved[] = {
        "await",   "break",    "case",       "catch",     "class",   "const",   "continue", "debugger",
        "default", "delete",   "do",         "else",      "enum",    "export",  "extends",  "false",
        "finally", "for",      "function",   "if",        "implements", "import", "in",     "instanceof",
        "interface", "let",    "new",        "null",      "package", "private", "protected", "public",
        "return",  "static",   "super",      "switch",    "this",    "throw",   "true",     "try",
        "typeof",  "var",      "void",       "while",     "with",    "yield"};

    for (auto *word : reserved)
    {
        if (name == word)
        {
            return name + "_";
        }
    }

    return name;
}

} // namespace tsbindgen
