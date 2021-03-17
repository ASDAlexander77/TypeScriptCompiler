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

#include "diagnostics.h"

#pragma warning( disable : 4062 )

using boolean = bool;
using number = int;
using string = std::wstring;
using char_t = wchar_t;
using sstream = std::wstringstream;
using regex = std::wregex;
using sregex_iterator = std::wsregex_iterator;

#define to_number_base(x, y) std::stoull(x, nullptr, y)
#define S(x) L##x
#define to_string(x) std::to_wstring(x)
#define to_number(x) std::stoi(x)

enum class ScriptTarget : number {
    ES3 = 0,
    ES5 = 1,
    ES2015 = 2,
    ES2016 = 3,
    ES2017 = 4,
    ES2018 = 5,
    ES2019 = 6,
    ES2020 = 7,
    ES2021 = 8,
    ESNext = 99,
    JSON = 100,
    Latest = ESNext,
};

enum class  TokenFlags : number {
    None = 0,
    /* @internal */
    PrecedingLineBreak = 1 << 0,
    /* @internal */
    PrecedingJSDocComment = 1 << 1,
    /* @internal */
    Unterminated = 1 << 2,
    /* @internal */
    ExtendedUnicodeEscape = 1 << 3,
    Scientific = 1 << 4,        // e.g. `10e2`
    Octal = 1 << 5,             // e.g. `0777`
    HexSpecifier = 1 << 6,      // e.g. `0x00000000`
    BinarySpecifier = 1 << 7,   // e.g. `0b0110010000000000`
    OctalSpecifier = 1 << 8,    // e.g. `0o777`
    /* @internal */
    ContainsSeparator = 1 << 9, // e.g. `0b1100_0101`
    /* @internal */
    UnicodeEscape = 1 << 10,
    /* @internal */
    ContainsInvalidEscape = 1 << 11,    // e.g. `\uhello`
    /* @internal */
    BinaryOrOctalSpecifier = BinarySpecifier | OctalSpecifier,
    /* @internal */
    NumericLiteralFlags = Scientific | Octal | HexSpecifier | BinaryOrOctalSpecifier | ContainsSeparator,
    /* @internal */
    TemplateLiteralLikeFlags = ContainsInvalidEscape,
};

enum class CommentDirectiveType : number {
    Undefined,
    ExpectError,
    Ignore
};

struct TextRange {
    TextRange() = default;
    number pos;
    number end;
};

struct CommentDirective {
    CommentDirective() = default;
    TextRange range;
    CommentDirectiveType type;
};

enum class LanguageVariant : number {
    Standard,
    JSX
};

enum class SyntaxKind : number {
    Unknown,
    EndOfFileToken,
    SingleLineCommentTrivia,
    MultiLineCommentTrivia,
    NewLineTrivia,
    WhitespaceTrivia,
    // We detect and preserve #! on the first line
    ShebangTrivia,
    // We detect and provide better error recovery when we encounter a git merge marker.  This
    // allows us to edit files with git-conflict markers in them in a much more pleasant manner.
    ConflictMarkerTrivia,
    // Literals
    NumericLiteral,
    BigIntLiteral,
    StringLiteral,
    JsxText,
    JsxTextAllWhiteSpaces,
    RegularExpressionLiteral,
    NoSubstitutionTemplateLiteral,
    // Pseudo-literals
    TemplateHead,
    TemplateMiddle,
    TemplateTail,
    // Punctuation
    OpenBraceToken,
    CloseBraceToken,
    OpenParenToken,
    CloseParenToken,
    OpenBracketToken,
    CloseBracketToken,
    DotToken,
    DotDotDotToken,
    SemicolonToken,
    CommaToken,
    QuestionDotToken,
    LessThanToken,
    LessThanSlashToken,
    GreaterThanToken,
    LessThanEqualsToken,
    GreaterThanEqualsToken,
    EqualsEqualsToken,
    ExclamationEqualsToken,
    EqualsEqualsEqualsToken,
    ExclamationEqualsEqualsToken,
    EqualsGreaterThanToken,
    PlusToken,
    MinusToken,
    AsteriskToken,
    AsteriskAsteriskToken,
    SlashToken,
    PercentToken,
    PlusPlusToken,
    MinusMinusToken,
    LessThanLessThanToken,
    GreaterThanGreaterThanToken,
    GreaterThanGreaterThanGreaterThanToken,
    AmpersandToken,
    BarToken,
    CaretToken,
    ExclamationToken,
    TildeToken,
    AmpersandAmpersandToken,
    BarBarToken,
    QuestionToken,
    ColonToken,
    AtToken,
    QuestionQuestionToken,
    /** Only the JSDoc scanner produces BacktickToken. The normal scanner produces NoSubstitutionTemplateLiteral and related kinds. */
    BacktickToken,
    // Assignments
    EqualsToken,
    PlusEqualsToken,
    MinusEqualsToken,
    AsteriskEqualsToken,
    AsteriskAsteriskEqualsToken,
    SlashEqualsToken,
    PercentEqualsToken,
    LessThanLessThanEqualsToken,
    GreaterThanGreaterThanEqualsToken,
    GreaterThanGreaterThanGreaterThanEqualsToken,
    AmpersandEqualsToken,
    BarEqualsToken,
    BarBarEqualsToken,
    AmpersandAmpersandEqualsToken,
    QuestionQuestionEqualsToken,
    CaretEqualsToken,
    // Identifiers and PrivateIdentifiers
    Identifier,
    PrivateIdentifier,
    // Reserved words
    BreakKeyword,
    CaseKeyword,
    CatchKeyword,
    ClassKeyword,
    ConstKeyword,
    ContinueKeyword,
    DebuggerKeyword,
    DefaultKeyword,
    DeleteKeyword,
    DoKeyword,
    ElseKeyword,
    EnumKeyword,
    ExportKeyword,
    ExtendsKeyword,
    FalseKeyword,
    FinallyKeyword,
    ForKeyword,
    FunctionKeyword,
    IfKeyword,
    ImportKeyword,
    InKeyword,
    InstanceOfKeyword,
    NewKeyword,
    NullKeyword,
    ReturnKeyword,
    SuperKeyword,
    SwitchKeyword,
    ThisKeyword,
    ThrowKeyword,
    TrueKeyword,
    TryKeyword,
    TypeOfKeyword,
    VarKeyword,
    VoidKeyword,
    WhileKeyword,
    WithKeyword,
    // Strict mode reserved words
    ImplementsKeyword,
    InterfaceKeyword,
    LetKeyword,
    PackageKeyword,
    PrivateKeyword,
    ProtectedKeyword,
    PublicKeyword,
    StaticKeyword,
    YieldKeyword,
    // Contextual keywords
    AbstractKeyword,
    AsKeyword,
    AssertsKeyword,
    AnyKeyword,
    AsyncKeyword,
    AwaitKeyword,
    BooleanKeyword,
    ConstructorKeyword,
    DeclareKeyword,
    GetKeyword,
    InferKeyword,
    IntrinsicKeyword,
    IsKeyword,
    KeyOfKeyword,
    ModuleKeyword,
    NamespaceKeyword,
    NeverKeyword,
    ReadonlyKeyword,
    RequireKeyword,
    NumberKeyword,
    ObjectKeyword,
    SetKeyword,
    StringKeyword,
    SymbolKeyword,
    TypeKeyword,
    UndefinedKeyword,
    UniqueKeyword,
    UnknownKeyword,
    FromKeyword,
    GlobalKeyword,
    BigIntKeyword,
    OfKeyword, // LastKeyword and LastToken and LastContextualKeyword

    // Parse tree nodes

    // Names
    QualifiedName,
    ComputedPropertyName,
    // Signature elements
    TypeParameter,
    Parameter,
    Decorator,
    // TypeMember
    PropertySignature,
    PropertyDeclaration,
    MethodSignature,
    MethodDeclaration,
    Constructor,
    GetAccessor,
    SetAccessor,
    CallSignature,
    ConstructSignature,
    IndexSignature,
    // Type
    TypePredicate,
    TypeReference,
    FunctionType,
    ConstructorType,
    TypeQuery,
    TypeLiteral,
    ArrayType,
    TupleType,
    OptionalType,
    RestType,
    UnionType,
    IntersectionType,
    ConditionalType,
    InferType,
    ParenthesizedType,
    ThisType,
    TypeOperator,
    IndexedAccessType,
    MappedType,
    LiteralType,
    NamedTupleMember,
    TemplateLiteralType,
    TemplateLiteralTypeSpan,
    ImportType,
    // Binding patterns
    ObjectBindingPattern,
    ArrayBindingPattern,
    BindingElement,
    // Expression
    ArrayLiteralExpression,
    ObjectLiteralExpression,
    PropertyAccessExpression,
    ElementAccessExpression,
    CallExpression,
    NewExpression,
    TaggedTemplateExpression,
    TypeAssertionExpression,
    ParenthesizedExpression,
    FunctionExpression,
    ArrowFunction,
    DeleteExpression,
    TypeOfExpression,
    VoidExpression,
    AwaitExpression,
    PrefixUnaryExpression,
    PostfixUnaryExpression,
    BinaryExpression,
    ConditionalExpression,
    TemplateExpression,
    YieldExpression,
    SpreadElement,
    ClassExpression,
    OmittedExpression,
    ExpressionWithTypeArguments,
    AsExpression,
    NonNullExpression,
    MetaProperty,
    SyntheticExpression,

    // Misc
    TemplateSpan,
    SemicolonClassElement,
    // Element
    Block,
    EmptyStatement,
    VariableStatement,
    ExpressionStatement,
    IfStatement,
    DoStatement,
    WhileStatement,
    ForStatement,
    ForInStatement,
    ForOfStatement,
    ContinueStatement,
    BreakStatement,
    ReturnStatement,
    WithStatement,
    SwitchStatement,
    LabeledStatement,
    ThrowStatement,
    TryStatement,
    DebuggerStatement,
    VariableDeclaration,
    VariableDeclarationList,
    FunctionDeclaration,
    ClassDeclaration,
    InterfaceDeclaration,
    TypeAliasDeclaration,
    EnumDeclaration,
    ModuleDeclaration,
    ModuleBlock,
    CaseBlock,
    NamespaceExportDeclaration,
    ImportEqualsDeclaration,
    ImportDeclaration,
    ImportClause,
    NamespaceImport,
    NamedImports,
    ImportSpecifier,
    ExportAssignment,
    ExportDeclaration,
    NamedExports,
    NamespaceExport,
    ExportSpecifier,
    MissingDeclaration,

    // Module references
    ExternalModuleReference,

    // JSX
    JsxElement,
    JsxSelfClosingElement,
    JsxOpeningElement,
    JsxClosingElement,
    JsxFragment,
    JsxOpeningFragment,
    JsxClosingFragment,
    JsxAttribute,
    JsxAttributes,
    JsxSpreadAttribute,
    JsxExpression,

    // Clauses
    CaseClause,
    DefaultClause,
    HeritageClause,
    CatchClause,

    // Property assignments
    PropertyAssignment,
    ShorthandPropertyAssignment,
    SpreadAssignment,

    // Enum
    EnumMember,
    // Unparsed
    UnparsedPrologue,
    UnparsedPrepend,
    UnparsedText,
    UnparsedInternalText,
    UnparsedSyntheticReference,

    // Top-level nodes
    SourceFile,
    Bundle,
    UnparsedSource,
    InputFiles,

    // JSDoc nodes
    JSDocTypeExpression,
    JSDocNameReference,
    JSDocAllType, // The * type
    JSDocUnknownType, // The ? type
    JSDocNullableType,
    JSDocNonNullableType,
    JSDocOptionalType,
    JSDocFunctionType,
    JSDocVariadicType,
    JSDocNamepathType, // https://jsdoc.app/about-namepaths.html
    JSDocComment,
    JSDocTypeLiteral,
    JSDocSignature,
    JSDocTag,
    JSDocAugmentsTag,
    JSDocImplementsTag,
    JSDocAuthorTag,
    JSDocDeprecatedTag,
    JSDocClassTag,
    JSDocPublicTag,
    JSDocPrivateTag,
    JSDocProtectedTag,
    JSDocReadonlyTag,
    JSDocCallbackTag,
    JSDocEnumTag,
    JSDocParameterTag,
    JSDocReturnTag,
    JSDocThisTag,
    JSDocTypeTag,
    JSDocTemplateTag,
    JSDocTypedefTag,
    JSDocSeeTag,
    JSDocPropertyTag,

    // Synthesized list
    SyntaxList,

    // Transformation nodes
    NotEmittedStatement,
    PartiallyEmittedExpression,
    CommaListExpression,
    MergeDeclarationMarker,
    EndOfDeclarationMarker,
    SyntheticReferenceExpression,

    // Enum value count
    Count,

    // Markers
    FirstAssignment = EqualsToken,
    LastAssignment = CaretEqualsToken,
    FirstCompoundAssignment = PlusEqualsToken,
    LastCompoundAssignment = CaretEqualsToken,
    FirstReservedWord = BreakKeyword,
    LastReservedWord = WithKeyword,
    FirstKeyword = BreakKeyword,
    LastKeyword = OfKeyword,
    FirstFutureReservedWord = ImplementsKeyword,
    LastFutureReservedWord = YieldKeyword,
    FirstTypeNode = TypePredicate,
    LastTypeNode = ImportType,
    FirstPunctuation = OpenBraceToken,
    LastPunctuation = CaretEqualsToken,
    FirstToken = Unknown,
    LastToken = LastKeyword,
    FirstTriviaToken = SingleLineCommentTrivia,
    LastTriviaToken = ConflictMarkerTrivia,
    FirstLiteralToken = NumericLiteral,
    LastLiteralToken = NoSubstitutionTemplateLiteral,
    FirstTemplateToken = NoSubstitutionTemplateLiteral,
    LastTemplateToken = TemplateTail,
    FirstBinaryOperator = LessThanToken,
    LastBinaryOperator = CaretEqualsToken,
    FirstStatement = VariableStatement,
    LastStatement = DebuggerStatement,
    FirstNode = QualifiedName,
    FirstJSDocNode = JSDocTypeExpression,
    LastJSDocNode = JSDocPropertyTag,
    FirstJSDocTagNode = JSDocTag,
    LastJSDocTagNode = JSDocPropertyTag,
    /* @internal */ FirstContextualKeyword = AbstractKeyword,
    /* @internal */ LastContextualKeyword = OfKeyword,
};

enum class CharacterCodes : number {
    outOfBoundary = -1,
    nullCharacter = 0,
    maxAsciiCharacter = 0x7F,

    lineFeed = 0x0A,              // \n
    carriageReturn = 0x0D,        // \r
    lineSeparator = 0x2028,
    paragraphSeparator = 0x2029,
    nextLine = 0x0085,

    // Unicode 3.0 space characters
    space = 0x0020,   // " "
    nonBreakingSpace = 0x00A0,   //
    enQuad = 0x2000,
    emQuad = 0x2001,
    enSpace = 0x2002,
    emSpace = 0x2003,
    threePerEmSpace = 0x2004,
    fourPerEmSpace = 0x2005,
    sixPerEmSpace = 0x2006,
    figureSpace = 0x2007,
    punctuationSpace = 0x2008,
    thinSpace = 0x2009,
    hairSpace = 0x200A,
    zeroWidthSpace = 0x200B,
    narrowNoBreakSpace = 0x202F,
    ideographicSpace = 0x3000,
    mathematicalSpace = 0x205F,
    ogham = 0x1680,
    _2bytes = 0x10000,

    _ = 0x5F,
    $ = 0x24,

    _0 = 0x30,
    _1 = 0x31,
    _2 = 0x32,
    _3 = 0x33,
    _4 = 0x34,
    _5 = 0x35,
    _6 = 0x36,
    _7 = 0x37,
    _8 = 0x38,
    _9 = 0x39,

    a = 0x61,
    b = 0x62,
    c = 0x63,
    d = 0x64,
    e = 0x65,
    f = 0x66,
    g = 0x67,
    h = 0x68,
    i = 0x69,
    j = 0x6A,
    k = 0x6B,
    l = 0x6C,
    m = 0x6D,
    n = 0x6E,
    o = 0x6F,
    p = 0x70,
    q = 0x71,
    r = 0x72,
    s = 0x73,
    t = 0x74,
    u = 0x75,
    v = 0x76,
    w = 0x77,
    x = 0x78,
    y = 0x79,
    z = 0x7A,

    A = 0x41,
    B = 0x42,
    C = 0x43,
    D = 0x44,
    E = 0x45,
    F = 0x46,
    G = 0x47,
    H = 0x48,
    I = 0x49,
    J = 0x4A,
    K = 0x4B,
    L = 0x4C,
    M = 0x4D,
    N = 0x4E,
    O = 0x4F,
    P = 0x50,
    Q = 0x51,
    R = 0x52,
    S = 0x53,
    T = 0x54,
    U = 0x55,
    V = 0x56,
    W = 0x57,
    X = 0x58,
    Y = 0x59,
    Z = 0x5a,

    ampersand = 0x26,             // &
    asterisk = 0x2A,              // *
    at = 0x40,                    // @
    backslash = 0x5C,             // 
    backtick = 0x60,              // `
    bar = 0x7C,                   // |
    caret = 0x5E,                 // ^
    closeBrace = 0x7D,            // }
    closeBracket = 0x5D,          // ]
    closeParen = 0x29,            // )
    colon = 0x3A,                 // :
    comma = 0x2C,                 // ,
    dot = 0x2E,                   // .
    doubleQuote = 0x22,           // "
    equals = 0x3D,                // =
    exclamation = 0x21,           // !
    greaterThan = 0x3E,           // >
    hash = 0x23,                  // #
    lessThan = 0x3C,              // <
    minus = 0x2D,                 // -
    openBrace = 0x7B,             // {
    openBracket = 0x5B,           // [
    openParen = 0x28,             // (
    percent = 0x25,               // %
    plus = 0x2B,                  // +
    question = 0x3F,              // ?
    semicolon = 0x3B,             // ;
    singleQuote = 0x27,           // '
    slash = 0x2F,                 // /
    tilde = 0x7E,                 // ~

    backspace = 0x08,             // \b
    formFeed = 0x0C,              // \f
    byteOrderMark = 0xFEFF,
    tab = 0x09,                   // \t
    verticalTab = 0x0B,           // \v
};

static TokenFlags& operator|=(TokenFlags& lhv, TokenFlags rhv)
{
    lhv = (TokenFlags) ((number) lhv | (number)rhv);
    return lhv;
}

static TokenFlags operator&(TokenFlags lhv, TokenFlags rhv)
{
    lhv = (TokenFlags) ((number) lhv & (number)rhv);
    return lhv;
}

static bool operator!(TokenFlags lhv)
{
    return (number)lhv == 0;
}

static bool operator!(SyntaxKind lhv)
{
    return (number)lhv == 0;
}

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

struct CommentRange {
    CommentRange() = default;

    SyntaxKind kind;
    number pos;
    number end;
    boolean hasTrailingNewLine;
};

struct ScanResult {
    ScanResult() = default;

    SyntaxKind kind;
    string value;
};

template <typename T, typename U>
using cb_type = std::function<U(number, number, SyntaxKind, boolean, T, U)>;

using ErrorCallback = std::function<void(DiagnosticMessage, number)>;

template <typename T>
auto arraysEqual(const std::vector<T> &a, const std::vector<T> &b) -> boolean {
    return std::equal(a.begin(), a.end(), b.begin());
}

enum class Comparison : number {
    LessThan    = -1,
    EqualTo     = 0,
    GreaterThan = 1
};
template <typename T>
using Comparer = std::function<Comparison(T, T)>;

template <typename T>
auto identity(T x, number i) -> T { return x; }

template <typename T>
auto compareComparableValues(T a, T b) {
    return a == b ? Comparison::EqualTo :
        a < b ? Comparison::LessThan :
        Comparison::GreaterThan;
}

template <typename T>
auto compareValues(T a, T b) -> Comparison {
    return compareComparableValues(a, b);
}

template <typename T, typename U>
auto binarySearch(const std::vector<T> &array, T value, std::function<U(T, number)> keySelector, Comparer<U> keyComparer, number offset = 0) -> number {
    return binarySearchKey<T, U>(array, keySelector(value, -1), keySelector, keyComparer, offset);
}

template <typename T, typename U>
auto binarySearchKey(const std::vector<T> &array, U key, std::function<U(T, number)> keySelector, Comparer<U> keyComparer, number offset = 0) -> number {
    if (!array) {
        return -1;
    }

    auto low = offset;
    auto high = array.size() - 1;
    while (low <= high) {
        auto middle = low + ((high - low) >> 1);
        auto midKey = keySelector(array[middle], middle);
        switch (keyComparer(midKey, key)) {
            case Comparison::LessThan:
                low = middle + 1;
                break;
            case Comparison::EqualTo:
                return middle;
            case Comparison::GreaterThan:
                high = middle - 1;
                break;
        }
    }

    return ~low;
}

static auto positionIsSynthesized(number pos) -> boolean {
    // This is a fast way of testing the following conditions:
    //  pos === undefined || pos === null || isNaN(pos) || pos < 0;
    return !(pos >= 0);
}

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
    auto firstNonzeroSegment = segments.size() - 1;
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
        Scanner(ScriptTarget languageVersion,
                            boolean skipTrivia,
                            LanguageVariant languageVariant = LanguageVariant::Standard,
                            string textInitial = S(""),
                            ErrorCallback onError = nullptr,
                            number start = 0,
                            number length = -1);

        auto scan() -> SyntaxKind;
        auto token() -> SyntaxKind;
        auto tokenPos() -> number;
        auto tokenValue() -> string;
        auto tokenFlags() -> TokenFlags;
        auto tokenToString(SyntaxKind) -> string;

        ~Scanner();
    };
}

#endif // SCANNER_H