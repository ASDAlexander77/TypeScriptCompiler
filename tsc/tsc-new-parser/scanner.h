#ifndef SCANNER_H
#define SCANNER_H

#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <regex>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>

#include "config.h"
#include "scanner_enums.h"
#include "diagnostics.h"

// forward declarations
struct CommentDirective;
struct DiagnosticMessage;
struct CommentRange;

struct safe_string
{
    string value;

    safe_string () : value{S("")} {}

    safe_string (string value) : value{value} {}

    safe_string& operator=(string& value_)
    {
        value = value_;
        return *this;
    }

    CharacterCodes operator [](number index)
    {
        if (index >= value.length())
        {
            return CharacterCodes::outOfBoundary;
        }

        return (CharacterCodes) value[index];
    }

    auto substring(number from, number to) -> string
    {
        return value.substr(from, to - from);
    }

    auto length() -> number
    {
        return value.length();
    }

    operator string&()
    {
        return value;
    }
};

template <typename T>
bool operator!(const std::vector<T>& values)
{
    return values.empty();
}

static void debug(bool cond)
{
    assert(cond);
}

static void debug(string msg)
{
    std::wcerr << msg.c_str();
}

static void debug(bool cond, string msg)
{
    if (!cond)
    {
        std::wcerr << msg.c_str();
    }
    
    assert(cond);
}

static void error(string msg)
{
    std::wcerr << msg;
}

class SourceFileLike {
public:
    string text;
    std::vector<number> lineMap;
    /* @internal */
    bool hasGetPositionOfLineAndCharacter;
    auto getPositionOfLineAndCharacter(number line, number character, bool allowEdits = true) -> number;
};

struct LineAndCharacter {

    LineAndCharacter() = default;

    /** 0-based. */
    number line;
    /*
        * 0-based. This value denotes the character position in line and is different from the 'column' because of tab characters.
        */
    number character;
};

struct ScanResult {
    ScanResult() = default;

    SyntaxKind kind;
    string value;
};

template <typename T, typename U>
using cb_type = std::function<U(number, number, SyntaxKind, boolean, T, U)>;

struct DiagnosticMessage;
using ErrorCallback = std::function<void(DiagnosticMessage, number)>;

template <typename T>
auto identity(T x, number i) -> T { return x; }

static auto parsePseudoBigInt(string stringValue) -> string {
    number log2Base;
    switch ((CharacterCodes)stringValue[1]) { // "x" in "0x123"
        case CharacterCodes::b:
        case CharacterCodes::B: // 0b or 0B
            log2Base = 1;
            break;
        case CharacterCodes::o:
        case CharacterCodes::O: // 0o or 0O
            log2Base = 3;
            break;
        case CharacterCodes::x:
        case CharacterCodes::X: // 0x or 0X
            log2Base = 4;
            break;
        default: // already in decimal; omit trailing "n"
            auto nIndex = stringValue.length() - 1;
            // Skip leading 0s
            auto nonZeroStart = 0;
            while ((CharacterCodes)stringValue[nonZeroStart] == CharacterCodes::_0) {
                nonZeroStart++;
            }
            return nonZeroStart > nIndex ? stringValue.substr(nonZeroStart, nIndex-nonZeroStart) : S("0");
    }

    // Omit leading "0b", "0o", or "0x", and trailing "n"
    auto startIndex = 2;
    auto endIndex = stringValue.length() - 1;
    auto bitsNeeded = (endIndex - startIndex) * log2Base;
    // Stores the value specified by the string as a LE array of 16-bit integers
    // using Uint16 instead of Uint32 so combining steps can use bitwise operators
    std::vector<uint16_t> segments((bitsNeeded >> 4) + (bitsNeeded & 15 ? 1 : 0));
    // Add the digits, one at a time
    for (int i = endIndex - 1, bitOffset = 0; i >= startIndex; i--, bitOffset += log2Base) {
        auto segment = bitOffset >> 4;
        auto digitChar = (number)stringValue[i];
        // Find character range: 0-9 < A-F < a-f
        auto digit = digitChar <= (number)CharacterCodes::_9
            ? digitChar - (number)CharacterCodes::_0
            : 10 + digitChar - (digitChar <= (number)CharacterCodes::F ? (number)CharacterCodes::A : (number)CharacterCodes::a);
        auto shiftedDigit = digit << (bitOffset & 15);
        segments[segment] |= shiftedDigit;
        auto residual = shiftedDigit >> 16;
        if (residual) segments[segment + 1] |= residual; // overflows segment
    }
    // Repeatedly divide segments by 10 and add remainder to base10Value
    auto base10Value = string(S(""));
    int firstNonzeroSegment = segments.size() - 1;
    auto segmentsRemaining = true;
    while (segmentsRemaining) {
        auto mod10 = 0;
        segmentsRemaining = false;
        for (auto segment = firstNonzeroSegment; segment >= 0; segment--) {
            auto newSegment = mod10 << 16 | segments[segment];
            auto segmentValue = (newSegment / 10) | 0;
            segments[segment] = segmentValue;
            mod10 = newSegment - segmentValue * 10;
            if (segmentValue && !segmentsRemaining) {
                firstNonzeroSegment = segment;
                segmentsRemaining = true;
            }
        }
        base10Value = to_string(mod10) + base10Value;
    }
    return base10Value;
}

namespace ts
{
    class ScannerImpl;

    class Scanner
    {
        ScannerImpl* impl;
    public:
        Scanner(ScriptTarget, boolean, LanguageVariant = LanguageVariant::Standard, string = string(), ErrorCallback = nullptr, number = 0, number = -1);

        auto setText(string, number = 0, number = -1) -> void;
        auto setOnError(ErrorCallback) -> void;
        auto setScriptTarget(ScriptTarget) -> void;
        auto setLanguageVariant(LanguageVariant) -> void;
        auto scan() -> SyntaxKind;
        auto getToken() -> SyntaxKind;
        auto getTextPos() -> number;
        auto getStartPos() -> number;
        auto getTokenPos() -> number;
        auto getTokenText() -> string;
        auto getTokenValue() -> string;
        auto tokenToString(SyntaxKind) -> string;
        auto syntaxKindString(SyntaxKind) -> string;
        auto setTextPos(number textPos) -> void;
        auto getCommentDirectives() -> std::vector<CommentDirective>;
        auto clearCommentDirectives() -> void;
        auto hasUnicodeEscape() -> boolean;
        auto hasExtendedUnicodeEscape() -> boolean;
        auto hasPrecedingLineBreak() -> boolean;
        auto hasPrecedingJSDocComment() -> boolean;
        auto isIdentifier() -> boolean;
        auto isReservedWord() -> boolean;
        auto isUnterminated() -> boolean;
        auto scanJsDocToken() -> SyntaxKind;
        auto reScanGreaterToken() -> SyntaxKind;
        auto reScanSlashToken() -> SyntaxKind;
        auto reScanTemplateToken(boolean isTaggedTemplate) -> SyntaxKind;
        auto reScanTemplateHeadOrNoSubstitutionTemplate() -> SyntaxKind;
        auto reScanLessThanToken() -> SyntaxKind;
        auto scanJsxIdentifier() -> SyntaxKind;
        auto scanJsxToken() -> SyntaxKind;
        auto scanJsxAttributeValue() -> SyntaxKind;
        auto reScanInvalidIdentifier() -> SyntaxKind;
        auto reScanJsxToken() -> SyntaxKind;
        auto tokenIsIdentifierOrKeyword(SyntaxKind token) -> boolean;
        auto tokenIsIdentifierOrKeywordOrGreaterThan(SyntaxKind token) -> boolean;
        auto getTokenFlags() -> TokenFlags;
        auto getNumericLiteralFlags() -> TokenFlags;
        auto setInJSDocType(boolean inType) -> void;
        auto reScanAsteriskEqualsToken() -> void;
        auto reScanQuestionToken() -> void;
        auto skipTrivia(safe_string &text, number pos, bool stopAfterLineBreak = false, bool stopAtComments = false) -> number;

        template <typename T>
        auto scanRange(number start, number length, std::function<T()> callback) -> T
        {
            return impl->scanRange<T>(start, length, callback);
        }

        template <typename T>
        auto tryScan(std::function<T()> callback) -> T
        {
            // TODO: can't use template method from pointer
            //return impl->tryScan<T>(callback);
            return T();
        }

        template <typename T>
        auto lookAhead(std::function<T()> callback) -> T
        {
            // TODO: can't use template method from pointer
            //return impl->lookAhead<T>(callback);
            return T();
        }

        ~Scanner();
    };
}

#endif // SCANNER_H