#ifndef SCANNER_H
#define SCANNER_H

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>

#include "config.h"
#include "diagnostics.h"
#include "scanner_enums.h"

#include "parser_types.h"

namespace ts
{
struct safe_string
{
    string value;

    safe_string() : value{S("")}
    {
    }

    safe_string(string value) : value{value}
    {
    }

    safe_string &operator=(string &value_)
    {
        value = value_;
        return *this;
    }

    CharacterCodes operator[](number index)
    {
        if ((size_t)index >= value.length())
        {
            return CharacterCodes::outOfBoundary;
        }

        return (CharacterCodes)value[index];
    }

    auto substring(number from, number to) -> string
    {
        return value.substr(from, to - from);
    }

    auto length() -> number
    {
        return value.length();
    }

    operator string &()
    {
        return value;
    }
};

template <typename T> bool operator!(NodeArray<T> &values)
{
    return !values.operator bool();
}

template <typename T> bool operator!(const std::vector<T> &values)
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

template <typename T, typename U> using cb_type = std::function<U(pos_type, number, SyntaxKind, boolean, T, U)>;

using ErrorCallback = std::function<void(DiagnosticMessage, number, string)>;

template <typename T> auto identity(T x, number i) -> T
{
    return x;
}

static auto parsePseudoBigInt(string stringValue) -> string
{
    number log2Base;
    switch ((CharacterCodes)stringValue[1])
    { // "x" in "0x123"
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
        size_t nonZeroStart = 0;
        while ((CharacterCodes)stringValue[nonZeroStart] == CharacterCodes::_0)
        {
            nonZeroStart++;
        }
        return nonZeroStart > nIndex ? stringValue.substr(nonZeroStart, nIndex - nonZeroStart) : S("0");
    }

    // Omit leading "0b", "0o", or "0x", and trailing "n"
    auto startIndex = 2;
    auto endIndex = stringValue.length() - 1;
    auto bitsNeeded = (endIndex - startIndex) * log2Base;
    // Stores the value specified by the string as a LE array of 16-bit integers
    // using Uint16 instead of Uint32 so combining steps can use bitwise operators
    std::vector<uint16_t> segments((bitsNeeded >> 4) + (bitsNeeded & 15 ? 1 : 0));
    // Add the digits, one at a time
    for (int i = endIndex - 1, bitOffset = 0; i >= startIndex; i--, bitOffset += log2Base)
    {
        auto segment = bitOffset >> 4;
        auto digitChar = (number)stringValue[i];
        // Find character range: 0-9 < A-F < a-f
        auto digit =
            digitChar <= (number)CharacterCodes::_9
                ? digitChar - (number)CharacterCodes::_0
                : 10 + digitChar - (digitChar <= (number)CharacterCodes::F ? (number)CharacterCodes::A : (number)CharacterCodes::a);
        auto shiftedDigit = digit << (bitOffset & 15);
        segments[segment] |= shiftedDigit;
        auto residual = shiftedDigit >> 16;
        if (residual)
            segments[segment + 1] |= residual; // overflows segment
    }
    // Repeatedly divide segments by 10 and add remainder to base10Value
    auto base10Value = string(S(""));
    int firstNonzeroSegment = segments.size() - 1;
    auto segmentsRemaining = true;
    while (segmentsRemaining)
    {
        auto mod10 = 0;
        segmentsRemaining = false;
        for (auto segment = firstNonzeroSegment; segment >= 0; segment--)
        {
            auto newSegment = mod10 << 16 | segments[segment];
            auto segmentValue = (newSegment / 10) | 0;
            segments[segment] = segmentValue;
            mod10 = newSegment - segmentValue * 10;
            if (segmentValue && !segmentsRemaining)
            {
                firstNonzeroSegment = segment;
                segmentsRemaining = true;
            }
        }
        base10Value = to_string_val(mod10) + base10Value;
    }
    return base10Value;
}

class Scanner
{
  public:    
    static std::map<string, SyntaxKind> textToKeyword;

    static std::map<string, SyntaxKind> textToToken;

    static std::map<SyntaxKind, string> tokenToText;

    static std::map<SyntaxKind, string> tokenStrings;

  private:
    static std::vector<number> unicodeES3IdentifierStart;

    static std::vector<number> unicodeES3IdentifierPart;

    static std::vector<number> unicodeES5IdentifierStart;

    static std::vector<number> unicodeES5IdentifierPart;

    static std::vector<number> unicodeESNextIdentifierStart;

    static std::vector<number> unicodeESNextIdentifierPart;

    static regex commentDirectiveRegExSingleLine;

    static regex commentDirectiveRegExMultiLine;

    static regex jsDocSeeOrLink;

    static number mergeConflictMarkerLength;

    static regex shebangTriviaRegex;

  protected:
    ScriptKind scriptKind;

    ScriptTarget languageVersion;

    boolean _skipTrivia;

    LanguageVariant languageVariant;

    // scanner text
    safe_string text;

    // Current position (end position of text of current token)
    number pos;

    // end of text
    number end;

    // Start position of whitespace before current token
    number fullStartPos;

    // Start position of text of current token
    number tokenStart;

    SyntaxKind token;
    string tokenValue;
    TokenFlags tokenFlags;

    std::vector<CommentDirective> commentDirectives;
    number inJSDocType = 0;

    JSDocParsingMode jsDocParsingMode = JSDocParsingMode::ParseAll;

    ErrorCallback onError = nullptr;

  public:
    // Creates a scanner over a (possibly unspecified) range of a piece of text.
    Scanner(ScriptTarget languageVersion, boolean skipTrivia, LanguageVariant languageVariant = LanguageVariant::Standard,
            string textInitial = string(), ErrorCallback onError = nullptr, number start = 0, number length = -1);

    auto getToken() -> SyntaxKind;

    auto getTokenFullStart() -> number;

    auto getTokenStart() -> number;
    
    auto getTokenEnd() -> number;

    auto getTokenText() -> string;

    auto getTokenValue() -> string;

    auto hasUnicodeEscape() -> boolean;

    auto hasExtendedUnicodeEscape() -> boolean;

    auto hasPrecedingLineBreak() -> boolean;

    auto hasPrecedingJSDocComment() -> boolean;

    auto isIdentifier() -> boolean;

    auto isReservedWord() -> boolean;

    auto isUnterminated() -> boolean;

    auto getTokenFlags() -> TokenFlags;

    auto getNumericLiteralFlags() -> TokenFlags;

    /* @internal */
    auto tokenIsIdentifierOrKeyword(SyntaxKind token) -> boolean;

    /* @internal */
    auto tokenIsIdentifierOrKeywordOrGreaterThan(SyntaxKind token) -> boolean;

    auto lookupInUnicodeMap(number code, std::vector<number> map) -> boolean;

    /* @internal */ auto isUnicodeIdentifierStart(CharacterCodes code, ScriptTarget languageVersion);

    auto isUnicodeIdentifierPart(CharacterCodes code, ScriptTarget languageVersion);

    static auto makeReverseMap(std::map<string, SyntaxKind> source) -> std::map<SyntaxKind, string>;

    auto tokenToString(SyntaxKind t) -> string;

    auto syntaxKindString(SyntaxKind t) -> string;

    /* @internal */
    auto stringToToken(string s) -> SyntaxKind;

    /* @internal */
    auto computeLineStarts(safe_string text) -> std::vector<number>;

    auto getPositionOfLineAndCharacter(SourceFileLike sourceFile, number line, number character, bool allowEdits = true) -> number;

    /* @internal */
    auto computePositionOfLineAndCharacter(std::vector<number> lineStarts, number line, number character, string debugText,
                                           bool allowEdits = true) -> number;

    /* @internal */
    auto getLineStarts(SourceFileLike sourceFile) -> std::vector<number>;

    /* @internal */
    auto computeLineAndCharacterOfPosition(std::vector<number> lineStarts, number position) -> LineAndCharacter;

    /**
     * @internal
     * We assume the first line starts at position 0 and 'position' is non-negative.
     */
    auto computeLineOfPosition(std::vector<number> lineStarts, number position, number lowerBound = 0) -> number;

    /** @internal */
    auto getLinesBetweenPositions(SourceFileLike sourceFile, number pos1, number pos2);

    auto getLineAndCharacterOfPosition(SourceFileLike sourceFile, number position) -> LineAndCharacter;

    auto isWhiteSpaceLike(CharacterCodes ch) -> boolean;

    /** Does not include line breaks. For that, see isWhiteSpaceLike. */
    auto isWhiteSpaceSingleLine(CharacterCodes ch) -> boolean;

    auto isLineBreak(CharacterCodes ch) -> boolean;

    auto isDigit(CharacterCodes ch) -> boolean;

    auto isHexDigit(CharacterCodes ch) -> boolean;

    auto isCodePoint(number code) -> boolean;

    /* @internal */
    auto isOctalDigit(CharacterCodes ch) -> boolean;

    auto couldStartTrivia(safe_string &text, number pos) -> boolean;

    /* @internal */
    auto skipTrivia(safe_string &text, number pos, bool stopAfterLineBreak = false, bool stopAtComments = false, bool inJSDoc = false) -> number;

    auto isConflictMarkerTrivia(safe_string &text, number pos) -> boolean;

    auto scanConflictMarkerTrivia(safe_string &text, number pos, std::function<void(DiagnosticMessage, number, number, string)> error = nullptr)
        -> number;

    /*@internal*/
    auto isShebangTrivia(string &text, number pos) -> boolean;

    /*@internal*/
    auto scanShebangTrivia(string &text, number pos) -> number;

    /**
     * Invokes a callback for each comment range following the provided position.
     *
     * Single-line comment ranges include the leading double-slash characters but not the ending
     * line break. Multi-line comment ranges include the leading slash-asterisk and trailing
     * asterisk-slash characters.
     *
     * @param reduce If true, accumulates the result of calling the callback in a fashion similar
     *      to reduceLeft. If false, iteration stops when the callback returns a truthy value.
     * @param text The source text to scan.
     * @param pos The position at which to start scanning.
     * @param trailing If false, whitespace is skipped until the first line break and comments
     *      between that location and the next token are returned. If true, comments occurring
     *      between the given position and the next line break are returned.
     * @param cb The callback to execute as each comment range is encountered.
     * @param state A state value to pass to each iteration of the callback.
     * @param initial An initial value to pass when accumulating results (when "reduce" is true).
     * @returns If "reduce" is true, the accumulated value. If "reduce" is false, the first truthy
     *      return value of the callback.
     */
    template <typename T, typename U>
    auto iterateCommentRanges(boolean reduce, safe_string text, pos_type posAdv, boolean trailing, cb_type<T, U> cb, T state, U initial = U())
        -> U
    {
        number pos = posAdv;
        number pendingPos = 0;
        number pendingEnd = 0;
        SyntaxKind pendingKind = SyntaxKind::Unknown;
        boolean pendingHasTrailingNewLine = false;
        auto hasPendingCommentRange = false;
        auto collecting = trailing;
        auto accumulator = initial;
        if (pos == 0)
        {
            collecting = true;
            auto shebang = getShebang(text);
            if (!shebang.empty())
            {
                pos = shebang.length();
            }
        }

        while (pos >= 0 && pos < text.length())
        {
            auto ch = text[pos];
            switch (ch)
            {
            case CharacterCodes::carriageReturn:
                if (text[pos + 1] == CharacterCodes::lineFeed)
                {
                    pos++;
                }
            // falls through
            case CharacterCodes::lineFeed:
                pos++;
                if (trailing)
                {
                    goto scan;
                }

                collecting = true;
                if (hasPendingCommentRange)
                {
                    pendingHasTrailingNewLine = true;
                }

                continue;
            case CharacterCodes::tab:
            case CharacterCodes::verticalTab:
            case CharacterCodes::formFeed:
            case CharacterCodes::space:
                pos++;
                continue;
            case CharacterCodes::slash: {
                auto nextChar = text[pos + 1];
                auto hasTrailingNewLine = false;
                if (nextChar == CharacterCodes::slash || nextChar == CharacterCodes::asterisk)
                {
                    auto kind =
                        nextChar == CharacterCodes::slash ? SyntaxKind::SingleLineCommentTrivia : SyntaxKind::MultiLineCommentTrivia;
                    auto startPos = pos;
                    pos += 2;
                    if (nextChar == CharacterCodes::slash)
                    {
                        while (pos < text.length())
                        {
                            if (isLineBreak(text[pos]))
                            {
                                hasTrailingNewLine = true;
                                break;
                            }
                            pos++;
                        }
                    }
                    else
                    {
                        while (pos < text.length())
                        {
                            if (text[pos] == CharacterCodes::asterisk && text[pos + 1] == CharacterCodes::slash)
                            {
                                pos += 2;
                                break;
                            }
                            pos++;
                        }
                    }

                    if (collecting)
                    {
                        if (hasPendingCommentRange)
                        {
                            accumulator = cb(pendingPos, pendingEnd, pendingKind, pendingHasTrailingNewLine, state, accumulator);
                            if (!reduce && !!accumulator)
                            {
                                // If we are not reducing and we have a truthy result, return it.
                                return accumulator;
                            }
                        }

                        pendingPos = startPos;
                        pendingEnd = pos;
                        pendingKind = kind;
                        pendingHasTrailingNewLine = hasTrailingNewLine;
                        hasPendingCommentRange = true;
                    }

                    continue;
                }
                goto scan;
            }
            default:
                if (ch > CharacterCodes::maxAsciiCharacter && (isWhiteSpaceLike(ch)))
                {
                    if (hasPendingCommentRange && isLineBreak(ch))
                    {
                        pendingHasTrailingNewLine = true;
                    }
                    pos++;
                    continue;
                }
                goto scan;
            }
        }
    scan:

        if (hasPendingCommentRange)
        {
            accumulator = cb(pendingPos, pendingEnd, pendingKind, pendingHasTrailingNewLine, state, accumulator);
        }

        return accumulator;
    }

    template <typename T, typename U> auto forEachLeadingCommentRange(string &text, pos_type pos, cb_type<T, U> cb, T state = T()) -> U
    {
        return iterateCommentRanges(/*reduce*/ false, text, pos, /*trailing*/ false, cb, state);
    }

    template <typename T, typename U> auto forEachTrailingCommentRange(string &text, pos_type pos, cb_type<T, U> cb, T state = T()) -> U
    {
        return iterateCommentRanges(/*reduce*/ false, text, pos, /*trailing*/ true, cb, state);
    }

    template <typename T, typename U> auto reduceEachLeadingCommentRange(string &text, pos_type pos, cb_type<T, U> cb, T state, U initial)
    {
        return iterateCommentRanges(/*reduce*/ true, text, pos, /*trailing*/ false, cb, state, initial);
    }

    template <typename T, typename U> auto reduceEachTrailingCommentRange(string &text, pos_type pos, cb_type<T, U> cb, T state, U initial)
    {
        return iterateCommentRanges(/*reduce*/ true, text, pos, /*trailing*/ true, cb, state, initial);
    }

    auto appendCommentRange(pos_type pos, number end, SyntaxKind kind, boolean hasTrailingNewLine, number state,
                            std::vector<CommentRange> comments) -> std::vector<CommentRange>;

    auto getLeadingCommentRanges(string &text, pos_type pos) -> std::vector<CommentRange>;

    auto getTrailingCommentRanges(string &text, pos_type pos) -> std::vector<CommentRange>;

    /** Optionally, get the shebang */
    auto getShebang(string &text) -> string;

    auto isIdentifierStart(CharacterCodes ch, ScriptTarget languageVersion) -> boolean;

    auto isIdentifierPart(CharacterCodes ch, ScriptTarget languageVersion, LanguageVariant identifierVariant = LanguageVariant::Standard)
        -> boolean;

    /* @internal */
    auto isIdentifierText(safe_string &name, ScriptTarget languageVersion, LanguageVariant identifierVariant = LanguageVariant::Standard)
        -> boolean;

    auto error(DiagnosticMessage message, number errPos = -1, number length = 0, string arg0 = S("")) -> void;

    auto scanNumberFragment() -> string;

    auto scanNumber() -> SyntaxKind;

    auto checkForIdentifierStartAfterNumericLiteral(number numericStart, bool isScientific = false) -> void;

    auto scanDigits() -> bool;

    /**
     * Scans the given number of hexadecimal digits in the text,
     * returning -1 if the given number is unavailable.
     */
    auto scanExactNumberOfHexDigits(number count, boolean canHaveSeparators) -> number;

    /**
     * Scans as many hexadecimal digits as are available in the text,
     * returning string() if the given number of digits was unavailable.
     */
    auto scanMinimumNumberOfHexDigits(number count, boolean canHaveSeparators) -> string;

    auto scanHexDigits(number minCount, boolean scanAsManyAsPossible, boolean canHaveSeparators) -> string;

    auto scanString(boolean jsxAttributeString = false) -> string;

    /**
     * Sets the current 'tokenValue' and returns a NoSubstitutionTemplateLiteral or
     * a literal component of a TemplateExpression.
     */
    auto scanTemplateAndSetTokenValue(boolean scanTemplateAndSetTokenValue) -> SyntaxKind;

    auto scanEscapeSequence(boolean shouldEmitInvalidEscapeError = false) -> string;

    auto scanHexadecimalEscape(number numDigits) -> string;

    auto scanExtendedUnicodeEscape() -> string;

    // Current character is known to be a backslash. Check for Unicode escape of the form '\uXXXX'
    // and return code point value if valid Unicode escape is found. Otherwise return -1.
    auto peekUnicodeEscape() -> CharacterCodes;

    auto peekExtendedUnicodeEscape() -> CharacterCodes;

    auto scanIdentifierParts() -> string;

    auto getIdentifierToken() -> SyntaxKind;

    auto scanBinaryOrOctalDigits(number base) -> string;

    auto checkBigIntSuffix() -> SyntaxKind;

/** @internal */
    auto scanJSDocCommentTextToken(boolean inBackticks) -> SyntaxKind; /*JSDocSyntaxKind | SyntaxKind.JSDocCommentTextToken*/

    auto scan() -> SyntaxKind;

    auto shouldParseJSDoc() -> bool;

    auto reScanInvalidIdentifier() -> SyntaxKind;

    auto scanIdentifier(CharacterCodes startCharacter, ScriptTarget languageVersion) -> SyntaxKind;

    auto reScanGreaterToken() -> SyntaxKind;

    auto reScanAsteriskEqualsToken() -> SyntaxKind;

    auto reScanSlashToken() -> SyntaxKind;

    auto appendIfCommentDirective(std::vector<CommentDirective> commentDirectives, string text, regex commentDirectiveRegEx,
                                  pos_type lineStart) -> std::vector<CommentDirective>;

    auto getDirectiveFromComment(string &text, regex commentDirectiveRegEx) -> CommentDirectiveType;

    auto reScanTemplateToken(boolean isTaggedTemplate) -> SyntaxKind;

    /** @deprecated use {@link reScanTemplateToken}(false) */
    auto reScanTemplateHeadOrNoSubstitutionTemplate() -> SyntaxKind;

    auto reScanJsxToken(boolean allowMultilineJsxText = true) -> SyntaxKind;

    auto reScanLessThanToken() -> SyntaxKind;

    auto reScanHashToken() -> SyntaxKind;

    auto reScanQuestionToken() -> SyntaxKind;

    auto scanJsxToken(boolean allowMultilineJsxText = true) -> SyntaxKind;

    // Scans a JSX identifier; these differ from normal identifiers in that
    // they allow dashes
    auto scanJsxIdentifier() -> SyntaxKind;

    auto scanJsxAttributeValue() -> SyntaxKind;

    auto reScanJsxAttributeValue() -> SyntaxKind;

    auto scanJsDocToken() -> SyntaxKind;

    template <typename T> auto speculationHelper(std::function<T()> callback, boolean isLookahead) -> T
    {
        auto savePos = pos;
        auto saveStartPos = fullStartPos;
        auto saveTokenPos = tokenStart;
        auto saveToken = token;
        auto saveTokenValue = tokenValue;
        auto saveTokenFlags = tokenFlags;
        auto result = callback();

        // If our callback returned something 'falsy' or we're just looking ahead,
        // then unconditionally restore us to where we were.
        if (!result || isLookahead)
        {
            pos = savePos;
            fullStartPos = saveStartPos;
            tokenStart = saveTokenPos;
            token = saveToken;
            tokenValue = saveTokenValue;
            tokenFlags = saveTokenFlags;
        }
        return result;
    }

    template <typename T> auto scanRange(number start, number length, std::function<T()> callback) -> T
    {
        auto saveEnd = end;
        auto savePos = pos;
        auto saveStartPos = fullStartPos;
        auto saveTokenPos = tokenStart;
        auto saveToken = token;
        auto saveTokenValue = tokenValue;
        auto saveTokenFlags = tokenFlags;
        auto saveErrorExpectations = commentDirectives;

        setText(text, start, length);
        auto result = callback();

        end = saveEnd;
        pos = savePos;
        fullStartPos = saveStartPos;
        tokenStart = saveTokenPos;
        token = saveToken;
        tokenValue = saveTokenValue;
        tokenFlags = saveTokenFlags;
        commentDirectives = saveErrorExpectations;

        return result;
    }

    template <typename T> auto lookAhead(std::function<T()> callback) -> T
    {
        return speculationHelper<T>(callback, /*isLookahead*/ true);
    }

    template <typename T> auto tryScan(std::function<T()> callback) -> T
    {
        return speculationHelper<T>(callback, /*isLookahead*/ false);
    }

    auto getText() -> string;

    auto getCommentDirectives() -> std::vector<CommentDirective>;

    auto clearCommentDirectives() -> void;

    auto setText(string newText, number start = 0, number length = -1) -> void;

    auto setOnError(ErrorCallback errorCallback) -> void;

    auto setScriptTarget(ScriptTarget scriptTarget) -> void;

    auto setLanguageVariant(LanguageVariant variant) -> void;

    auto setScriptKind(ScriptKind scriptKind) -> void;
    
    auto setJSDocParsingMode(JSDocParsingMode kind) -> void;

    auto resetTokenState(number pos) -> void;

    /** @internal */
    auto setInJSDocType(boolean inType) -> void;

    /* @internal */
    auto codePointAt(safe_string &str, number i) -> CharacterCodes;

    /* @internal */
    auto charSize(CharacterCodes ch) -> number;

    // Derived from the 10.1.1 UTF16Encoding of the ES6 Spec.
    auto utf16EncodeAsStringFallback(number codePoint) -> string;

    /* @internal */
    auto utf16EncodeAsString(CharacterCodes codePoint) -> string;
};
} // namespace ts

#endif // SCANNER_H