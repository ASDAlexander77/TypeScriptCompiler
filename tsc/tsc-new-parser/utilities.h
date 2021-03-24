#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <regex>
#include <functional>

#include "core.h"
#include "enums.h"
#include "types.h"
#include "nodeTest.h"
#include "scanner.h"
#include "parser.h"

namespace Extension
{
    static const string Ts = S(".ts");
    static const string Tsx = S(".tsx");
    static const string Dts = S(".d.ts");
    static const string Js = S(".js");
    static const string Jsx = S(".jsx");
    static const string Json = S(".json");
    static const string TsBuildInfo = S(".tsbuildinfo");
};

inline auto positionIsSynthesized(number pos) -> boolean
{
    // This is a fast way of testing the following conditions:
    //  pos === undefined || pos === null || isNaN(pos) || pos < 0;
    return !(pos >= 0);
}

inline auto getScriptKindFromFileName(string fileName) -> ScriptKind
{
    auto ext = fileName.substr(fileName.find(S('.')));
    std::transform(ext.begin(), ext.end(), ext.begin(), [](char_t c) { return std::tolower(c); });
    if (ext == S("js"))
        return ScriptKind::JS;
    if (ext == S("jsx"))
        return ScriptKind::JSX;
    if (ext == S("ts"))
        return ScriptKind::TS;
    if (ext == S("tsx"))
        return ScriptKind::TSX;
    if (ext == S("json"))
        return ScriptKind::JSON;
    return ScriptKind::Unknown;
}

inline auto ensureScriptKind(string fileName, ScriptKind scriptKind = ScriptKind::Unknown) -> ScriptKind
{
    // Using scriptKind as a condition handles both:
    // - 'scriptKind' is unspecified and thus it is `undefined`
    // - 'scriptKind' is set and it is `Unknown` (0)
    // If the 'scriptKind' is 'undefined' or 'Unknown' then we attempt
    // to get the ScriptKind from the file name. If it cannot be resolved
    // from the file name then the default 'TS' script kind is returned.
    return scriptKind != ScriptKind::Unknown ? scriptKind : scriptKind = getScriptKindFromFileName(fileName), scriptKind != ScriptKind::Unknown ? scriptKind : ScriptKind::TS;
}

inline auto isDiagnosticWithDetachedLocation(DiagnosticRelatedInformation diagnostic) -> boolean
{
    return diagnostic.start != -1 && diagnostic.length != -1 && diagnostic.fileName != S("");
}

auto attachFileToDiagnostic(DiagnosticRelatedInformation diagnostic, SourceFile file) -> DiagnosticWithLocation
{
    auto fileName = file->fileName;
    auto length = file->text.length();
    Debug::assertEqual(diagnostic.fileName, fileName);
    Debug::assertLessThanOrEqual(diagnostic.start, length);
    Debug::assertLessThanOrEqual(diagnostic.start + diagnostic.length, length);
    DiagnosticWithLocation diagnosticWithLocation;
    diagnosticWithLocation.file = file;
    diagnosticWithLocation.start = diagnostic.start;
    diagnosticWithLocation.length = diagnostic.length;
    diagnosticWithLocation.messageText = diagnostic.messageText;
    diagnosticWithLocation.category = diagnostic.category;
    diagnosticWithLocation.code = diagnostic.code;

    return diagnosticWithLocation;
}

auto attachFileToDiagnostic(Diagnostic diagnostic, SourceFile file) -> DiagnosticWithLocation
{
    auto fileName = file->fileName;
    auto length = file->text.length();
    Debug::assertEqual(diagnostic.fileName, fileName);
    Debug::assertLessThanOrEqual(diagnostic.start, length);
    Debug::assertLessThanOrEqual(diagnostic.start + diagnostic.length, length);
    DiagnosticWithLocation diagnosticWithLocation;
    diagnosticWithLocation.file = file;
    diagnosticWithLocation.start = diagnostic.start;
    diagnosticWithLocation.length = diagnostic.length;
    diagnosticWithLocation.messageText = diagnostic.messageText;
    diagnosticWithLocation.category = diagnostic.category;
    diagnosticWithLocation.code = diagnostic.code;
    diagnosticWithLocation.reportsUnnecessary = diagnostic.reportsUnnecessary;

    if (!diagnostic.relatedInformation.empty())
    {
        for (auto &related : diagnostic.relatedInformation)
        {
            if (isDiagnosticWithDetachedLocation(related) && related.fileName == fileName)
            {
                Debug::assertLessThanOrEqual(related.start, length);
                Debug::assertLessThanOrEqual(related.start + related.length, length);
                diagnosticWithLocation.relatedInformation.push_back(attachFileToDiagnostic(related, file));
            }
            else
            {
                diagnosticWithLocation.relatedInformation.push_back(related);
            }
        }
    }

    return diagnosticWithLocation;
}

static auto attachFileToDiagnostics(std::vector<DiagnosticWithDetachedLocation> diagnostics, SourceFile file) -> std::vector<DiagnosticWithLocation>
{
    std::vector<DiagnosticWithLocation> diagnosticsWithLocation;
    for (auto &diagnostic : diagnostics)
    {
        diagnosticsWithLocation.push_back(attachFileToDiagnostic(diagnostic, file));
    }
    return diagnosticsWithLocation;
}

static auto assertDiagnosticLocation(SourceFile file, number start, number length)
{
    Debug::assertGreaterThanOrEqual(start, 0);
    Debug::assertGreaterThanOrEqual(length, 0);

    if (!!file)
    {
        Debug::assertLessThanOrEqual(start, file->text.length());
        Debug::assertLessThanOrEqual(start + length, file->text.length());
    }
}

static auto getLocaleSpecificMessage(DiagnosticMessage message) -> string
{
    return string(message.message);
}

static auto createDetachedDiagnostic(string fileName, number start, number length, DiagnosticMessage message) -> DiagnosticWithDetachedLocation
{
    assertDiagnosticLocation(/*file*/ SourceFile(), start, length);
    auto text = getLocaleSpecificMessage(message);

    /*
    if (arguments.length > 4) {
        text = formatStringFromArgs(text, arguments, 4);
    }
    */

    DiagnosticWithDetachedLocation d;
    d.start = start;
    d.length = length;

    d.messageText = text;
    d.category = message.category;
    d.code = message.code;
    //diagnosticWithDetachedLocation.reportsUnnecessary = message.reportsUnnecessary;
    d.fileName = fileName;

    return d;
}

static auto createDetachedDiagnostic(string fileName, number start, number length, DiagnosticMessage message, string arg0, ...) -> DiagnosticWithDetachedLocation
{
    assertDiagnosticLocation(/*file*/ SourceFile(), start, length);
    auto text = getLocaleSpecificMessage(message);

    // TODO:
    /*
    if (arguments.length > 4) {
        text = formatStringFromArgs(text, arguments, 4);
    }
    */

    DiagnosticWithDetachedLocation d;
    d.start = start;
    d.length = length;

    d.messageText = text;
    d.category = message.category;
    d.code = message.code;
    //diagnosticWithDetachedLocation.reportsUnnecessary = message.reportsUnnecessary;
    d.fileName = fileName;

    return d;
}

inline auto normalizePath(string path) -> string
{
    // TODO: finish it
    return path;
}

inline auto getLanguageVariant(ScriptKind scriptKind) -> LanguageVariant
{
    // .tsx and .jsx files are treated as jsx language variant.
    return scriptKind == ScriptKind::TSX || scriptKind == ScriptKind::JSX || scriptKind == ScriptKind::JS || scriptKind == ScriptKind::JSON ? LanguageVariant::JSX : LanguageVariant::Standard;
}

inline auto endsWith(string str, string suffix) -> boolean
{
    auto expectedPos = str.length() - suffix.length();
    return expectedPos >= 0 && str.find(suffix, expectedPos) == expectedPos;
}

inline auto fileExtensionIs(string path, string extension) -> boolean
{
    return path.length() > extension.length() && endsWith(path, extension);
}

template <typename T>
inline auto setTextRangePos(T range, number pos)
{
    range->pos = pos;
    return range;
}

template <typename T>
inline auto setTextRangeEnd(T range, number end) -> T
{
    range->end = end;
    return range;
}

template <typename T>
inline auto setTextRangePosEnd(T range, number pos, number end)
{
    return setTextRangeEnd(setTextRangePos(range, pos), end);
}

template <typename T>
inline auto setTextRangePosWidth(T range, number pos, number width)
{
    return setTextRangePosEnd(range, pos, pos + width);
}

template <typename T>
inline auto setTextRange(T range, TextRange location) -> T
{
    return location ? setTextRangePosEnd(range, location.pos, location.end) : range;
}

template <typename T>
auto setParentRecursive(T rootNode, boolean incremental) -> T
{

    auto bindParentToChildIgnoringJSDoc = [&](Node child, Node parent) -> boolean /*true is skip*/ {
        if (incremental &&child.parent == = parent)
        {
            return true;
        }
        setParent(child, parent);
        return false;
    };

    auto bindJSDoc = [&](Node child) {
        if (hasJSDocNodes(child))
        {
            for (const doc : child.jsDoc)
            {
                bindParentToChildIgnoringJSDoc(doc, child);
                forEachChildRecursively(doc, bindParentToChildIgnoringJSDoc);
            }
        }
    };

    auto bindParentToChild = [&](Node child, Node parent) {
        return bindParentToChildIgnoringJSDoc(child, parent) || bindJSDoc(child);
    };

    if (!rootNode)
        return rootNode;
    forEachChildRecursively(rootNode, isJSDocNode(rootNode) ? bindParentToChildIgnoringJSDoc : bindParentToChild);
    return rootNode;
}

inline auto isKeyword(SyntaxKind token) -> boolean
{
    return SyntaxKind::FirstKeyword <= token && token <= SyntaxKind::LastKeyword;
}

inline auto isTemplateLiteralKind(SyntaxKind kind) -> boolean
{
    return SyntaxKind::FirstTemplateToken <= kind && kind <= SyntaxKind::LastTemplateToken;
}

inline auto isModifierKind(SyntaxKind token) -> boolean
{
    switch (token)
    {
    case SyntaxKind::AbstractKeyword:
    case SyntaxKind::AsyncKeyword:
    case SyntaxKind::ConstKeyword:
    case SyntaxKind::DeclareKeyword:
    case SyntaxKind::DefaultKeyword:
    case SyntaxKind::ExportKeyword:
    case SyntaxKind::PublicKeyword:
    case SyntaxKind::PrivateKeyword:
    case SyntaxKind::ProtectedKeyword:
    case SyntaxKind::ReadonlyKeyword:
    case SyntaxKind::StaticKeyword:
        return true;
    }
    return false;
}

inline auto nodeIsMissing(Node node) -> boolean
{
    if (node == undefined)
    {
        return true;
    }

    return node->pos == node->end && node->pos >= 0 && node->kind != SyntaxKind::EndOfFileToken;
}

inline auto nodeIsPresent(Node node) -> boolean
{
    return !nodeIsMissing(node);
}

inline auto containsParseError(Node node) -> boolean
{
    return (node->flags & NodeFlags::ThisNodeOrAnySubNodesHasError) != NodeFlags::None;
}

inline auto isLiteralKind(SyntaxKind kind) -> boolean
{
    return SyntaxKind::FirstLiteralToken <= kind && kind <= SyntaxKind::LastLiteralToken;
}

inline auto getFullWidth(Node node) -> number
{
    return node->end - node->pos;
}

inline auto isOuterExpression(Node node, OuterExpressionKinds kinds = OuterExpressionKinds::All) -> boolean
{
    switch (node->kind)
    {
    case SyntaxKind::ParenthesizedExpression:
        return (kinds & OuterExpressionKinds::Parentheses) != OuterExpressionKinds::None;
    case SyntaxKind::TypeAssertionExpression:
    case SyntaxKind::AsExpression:
        return (kinds & OuterExpressionKinds::TypeAssertions) != OuterExpressionKinds::None;
    case SyntaxKind::NonNullExpression:
        return (kinds & OuterExpressionKinds::NonNullAssertions) != OuterExpressionKinds::None;
    case SyntaxKind::PartiallyEmittedExpression:
        return (kinds & OuterExpressionKinds::PartiallyEmittedExpressions) != OuterExpressionKinds::None;
    }
    return false;
}

inline auto skipOuterExpressions(Node node, OuterExpressionKinds kinds = OuterExpressionKinds::All)
{
    while (isOuterExpression(node, kinds))
    {
        switch (node->kind)
        {
        case SyntaxKind::ParenthesizedExpression:
            node = node.as<ParenthesizedExpression>()->expression;
            break;
        case SyntaxKind::TypeAssertionExpression:
            node = node.as<TypeAssertion>()->expression;
            break;
        case SyntaxKind::AsExpression:
            node = node.as<AsExpression>()->expression;
            break;
        case SyntaxKind::NonNullExpression:
            node = node.as<NonNullExpression>()->expression;
            break;
        case SyntaxKind::PartiallyEmittedExpression:
            node = node.as<PartiallyEmittedExpression>()->expression;
            break;
        }
    }
    return node;
}

inline auto skipPartiallyEmittedExpressions(Node node)
{
    return skipOuterExpressions(node, OuterExpressionKinds::PartiallyEmittedExpressions);
}

inline auto isLeftHandSideExpressionKind(SyntaxKind kind) -> boolean
{
    switch (kind)
    {
    case SyntaxKind::PropertyAccessExpression:
    case SyntaxKind::ElementAccessExpression:
    case SyntaxKind::NewExpression:
    case SyntaxKind::CallExpression:
    case SyntaxKind::JsxElement:
    case SyntaxKind::JsxSelfClosingElement:
    case SyntaxKind::JsxFragment:
    case SyntaxKind::TaggedTemplateExpression:
    case SyntaxKind::ArrayLiteralExpression:
    case SyntaxKind::ParenthesizedExpression:
    case SyntaxKind::ObjectLiteralExpression:
    case SyntaxKind::ClassExpression:
    case SyntaxKind::FunctionExpression:
    case SyntaxKind::Identifier:
    case SyntaxKind::RegularExpressionLiteral:
    case SyntaxKind::NumericLiteral:
    case SyntaxKind::BigIntLiteral:
    case SyntaxKind::StringLiteral:
    case SyntaxKind::NoSubstitutionTemplateLiteral:
    case SyntaxKind::TemplateExpression:
    case SyntaxKind::FalseKeyword:
    case SyntaxKind::NullKeyword:
    case SyntaxKind::ThisKeyword:
    case SyntaxKind::TrueKeyword:
    case SyntaxKind::SuperKeyword:
    case SyntaxKind::NonNullExpression:
    case SyntaxKind::MetaProperty:
    case SyntaxKind::ImportKeyword: // technically this is only an Expression if it's in a CallExpression
        return true;
    default:
        return false;
    }
}

inline auto isLeftHandSideExpression(Node node) -> boolean
{
    return isLeftHandSideExpressionKind(skipPartiallyEmittedExpressions(node)->kind);
}

inline auto isAssignmentOperator(SyntaxKind token) -> boolean {
    return token >= SyntaxKind::FirstAssignment && token <= SyntaxKind::LastAssignment;
}

inline auto getBinaryOperatorPrecedence(SyntaxKind kind) -> OperatorPrecedence {
    switch (kind) {
        case SyntaxKind::QuestionQuestionToken:
            return OperatorPrecedence::Coalesce;
        case SyntaxKind::BarBarToken:
            return OperatorPrecedence::LogicalOR;
        case SyntaxKind::AmpersandAmpersandToken:
            return OperatorPrecedence::LogicalAND;
        case SyntaxKind::BarToken:
            return OperatorPrecedence::BitwiseOR;
        case SyntaxKind::CaretToken:
            return OperatorPrecedence::BitwiseXOR;
        case SyntaxKind::AmpersandToken:
            return OperatorPrecedence::BitwiseAND;
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
            return OperatorPrecedence::Equality;
        case SyntaxKind::LessThanToken:
        case SyntaxKind::GreaterThanToken:
        case SyntaxKind::LessThanEqualsToken:
        case SyntaxKind::GreaterThanEqualsToken:
        case SyntaxKind::InstanceOfKeyword:
        case SyntaxKind::InKeyword:
        case SyntaxKind::AsKeyword:
            return OperatorPrecedence::Relational;
        case SyntaxKind::LessThanLessThanToken:
        case SyntaxKind::GreaterThanGreaterThanToken:
        case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
            return OperatorPrecedence::Shift;
        case SyntaxKind::PlusToken:
        case SyntaxKind::MinusToken:
            return OperatorPrecedence::Additive;
        case SyntaxKind::AsteriskToken:
        case SyntaxKind::SlashToken:
        case SyntaxKind::PercentToken:
            return OperatorPrecedence::Multiplicative;
        case SyntaxKind::AsteriskAsteriskToken:
            return OperatorPrecedence::Exponentiation;
    }

    // -1 is lower than all other precedences.  Returning it will cause binary expression
    // parsing to stop.
    return OperatorPrecedence::Invalid;
}

static auto findAncestor(Node node, std::function<boolean(Node)> callback) -> Node {
    while (node) {
        auto result = callback(node);
        if (result) {
            return node;
        }
        node = node->parent;
    }
    return undefined;
}

static auto isJSDocTypeExpressionOrChild(Node node) -> boolean {
    return !!findAncestor(node, [](Node n) { return isJSDocTypeExpression(n); });
}

static auto getTextOfNodeFromSourceText(safe_string sourceText, Node node, boolean includeTrivia = false, ts::Scanner* scanner = nullptr) -> string {
    if (nodeIsMissing(node)) {
        return string();
    }

    auto text = sourceText.substring(includeTrivia ? node->pos : scanner->skipTrivia(sourceText, node->pos), node->end);

    if (isJSDocTypeExpressionOrChild(node)) {
        // strip space + asterisk at line start
        auto reg = regex(S("(^|\\r?\\n|\\r)\\s*\\*\\s*"), std::regex_constants::extended);
        text = regex_replace(text, reg, S("$1"));
    }

    return text;
}

static auto isStringLiteralLike(Node node) -> boolean {
    return node->kind == SyntaxKind::StringLiteral || node->kind == SyntaxKind::NoSubstitutionTemplateLiteral;
}

static auto isStringOrNumericLiteralLike(Node node) -> boolean {
    return isStringLiteralLike(node) || isNumericLiteral(node);
}

template <typename T>
static auto addRelatedInfo(T diagnostic, std::vector<DiagnosticRelatedInformation> relatedInformation): T {
    if (!relatedInformation.size()) {
        return diagnostic;
    }
    if (diagnostic.relatedInformation.size() > 0) {
        diagnostic.relatedInformation.clear();
    }
    Debug::_assert(diagnostic.relatedInformation.size() != 0, S("Diagnostic had empty array singleton for related info, but is still being constructed!"));
    for (auto &item : relatedInformation)
    {
        diagnostic.relatedInformation.push_back(item);
    }

    return diagnostic;
}

#endif // UTILITIES_H