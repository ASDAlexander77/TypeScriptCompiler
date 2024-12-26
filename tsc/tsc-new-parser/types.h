#ifndef TYPES_H
#define TYPES_H

#include "config.h"
#include "undefined.h"

#include <map>

namespace ts
{
    struct LineAndCharacter {

        LineAndCharacter() = default;

        /** 0-based. */
        number line;
        /*
            * 0-based. This value denotes the character position in line and is different from the 'column' because of tab characters.
            */
        number character;
    };

    struct DiagnosticMessageStore
    {
        DiagnosticMessageStore() = default;
        DiagnosticMessageStore(int code, DiagnosticCategory category, string label, string message): code(code), category(category), label(label), message(message) {};
        DiagnosticMessageStore(undefined_t): category{DiagnosticCategory::Undefined} {}

        int code;
        DiagnosticCategory category;
        string label;
        string message;

        bool operator !()
        {
            return category == DiagnosticCategory::Undefined;
        }
    };

    enum class PragmaKindFlags {
        None = 0,
        /**
         * Triple slash comment of the form
         * /// <pragma-name argname=S("value") />
         */
        TripleSlashXML = 1 << 0,
        /**
         * Single line comment of the form
         * // @pragma-name argval1 argval2
         * or
         * /// @pragma-name argval1 argval2
         */
        SingleLine = 1 << 1,
        /**
         * Multiline non-jsdoc pragma of the form
         * /* @pragma-name argval1 argval2 * /
         */
        MultiLine = 1 << 2,
        All = TripleSlashXML | SingleLine | MultiLine,
        Default = All,
    };

    struct Arg
    {
        string name;
        bool optional;
        bool captureSpan;
    };

    struct PragmaDefinition
    {
        std::vector<Arg> args;
        PragmaKindFlags kind;
    };

    static std::map<string, PragmaDefinition> commentPragmas = {
        {S("reference"), {
            {
                { S("types"), true,  true },
                { S("lib"), true,  true },
                { S("path"), true,  true },
                { S("no-default-lib"), true },
                { S("resolution-mode"), true },
                { S("preserve"), true },
            },
            PragmaKindFlags::TripleSlashXML
        }},
        {S("amd-dependency"), {
            {{ S("path") }, { S("name"), true }},
            PragmaKindFlags::TripleSlashXML
        }},
        {S("amd-module"), {
            {{ S("name") }},
            PragmaKindFlags::TripleSlashXML
        }},
        {S("ts-check"), {
            {},
            PragmaKindFlags::SingleLine
        }},
        {S("ts-nocheck"), {
            {},
            PragmaKindFlags::SingleLine
        }},
        {S("jsx"), {
            {{ S("factory") }},
            PragmaKindFlags::MultiLine
        }},
        {S("jsxfrag"), {
            {{ S("factory") }},
            PragmaKindFlags::MultiLine
        }},
        {S("jsximportsource"), {
            {{ S("factory") }},
            PragmaKindFlags::MultiLine
        }},
        {S("jsxruntime"), {
            {{ S("factory") }},
            PragmaKindFlags::MultiLine
        }},

        // my addon
        {S("strict-null"), {
            {{ S("option") }},
            PragmaKindFlags::SingleLine
        }},        
    };

} // namespace ts

#endif // ENUMS_H