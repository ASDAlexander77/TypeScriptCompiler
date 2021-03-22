#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <regex>
#include <functional>

#include "core.h"
#include "enums.h"
#include "types.h"
#include "parser.h"

namespace Extension {
    static const string Ts = S(".ts");
    static const string Tsx = S(".tsx");
    static const string Dts = S(".d.ts");
    static const string Js = S(".js");
    static const string Jsx = S(".jsx");
    static const string Json = S(".json");
    static const string TsBuildInfo = S(".tsbuildinfo");
};

inline auto positionIsSynthesized(number pos) -> boolean {
    // This is a fast way of testing the following conditions:
    //  pos === undefined || pos === null || isNaN(pos) || pos < 0;
    return !(pos >= 0);
}

inline auto getScriptKindFromFileName(string fileName) -> ScriptKind
{
    auto ext = fileName.substr(fileName.find(S('.')));
    std::transform(ext.begin(), ext.end(), ext.begin(), [](char_t c){ return std::tolower(c); });
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

inline auto ensureScriptKind(string fileName, ScriptKind scriptKind = ScriptKind::Unknown) -> ScriptKind {
    // Using scriptKind as a condition handles both:
    // - 'scriptKind' is unspecified and thus it is `undefined`
    // - 'scriptKind' is set and it is `Unknown` (0)
    // If the 'scriptKind' is 'undefined' or 'Unknown' then we attempt
    // to get the ScriptKind from the file name. If it cannot be resolved
    // from the file name then the default 'TS' script kind is returned.
    return scriptKind != ScriptKind::Unknown ? scriptKind : scriptKind = getScriptKindFromFileName(fileName), scriptKind != ScriptKind::Unknown ? scriptKind : ScriptKind::TS;
}

inline auto isDiagnosticWithDetachedLocation(DiagnosticRelatedInformation diagnostic) -> boolean {
    return diagnostic.start != -1
        && diagnostic.length != -1
        && diagnostic.fileName != S("");
}

template <typename T>
auto attachFileToDiagnostic(T diagnostic, SourceFile file) -> DiagnosticWithLocation {
    auto fileName = file.fileName;
    auto length = file.text.length();
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

    if (!diagnostic.relatedInformation.empty()) {
        for (auto &related : diagnostic.relatedInformation) {
            if (isDiagnosticWithDetachedLocation(related) && related.fileName == fileName) {
                Debug::assertLessThanOrEqual(related.start, length);
                Debug::assertLessThanOrEqual(related.start + related.length, length);
                diagnosticWithLocation.relatedInformation.push_back(attachFileToDiagnostic(related, file));
            }
            else {
                diagnosticWithLocation.relatedInformation.push_back(related);
            }
        }
    }

    return diagnosticWithLocation;
}

static auto attachFileToDiagnostics(std::vector<DiagnosticWithDetachedLocation> diagnostics, SourceFile file) -> std::vector<DiagnosticWithLocation> {
    std::vector<DiagnosticWithLocation> diagnosticsWithLocation;
    for (auto &diagnostic : diagnostics) {
        diagnosticsWithLocation.push_back(attachFileToDiagnostic(diagnostic, file));
    }
    return diagnosticsWithLocation;
}

static auto  assertDiagnosticLocation(SourceFile file, number start, number length) {
    Debug::assertGreaterThanOrEqual(start, 0);
    Debug::assertGreaterThanOrEqual(length, 0);

    if (!!file) {
        Debug::assertLessThanOrEqual(start, file.text.length());
        Debug::assertLessThanOrEqual(start + length, file.text.length());
    }
}

static auto getLocaleSpecificMessage(DiagnosticMessage message) -> string {
    return string(message.message);
}

static auto createDetachedDiagnostic(string fileName, number start, number length, DiagnosticMessage message) -> DiagnosticWithDetachedLocation {
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

static auto createDetachedDiagnostic(string fileName, number start, number length, DiagnosticMessage message, string arg0, ...) -> DiagnosticWithDetachedLocation {
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

inline auto normalizePath(string path) -> string {
    // TODO: finish it
    return path;
}

inline auto getLanguageVariant(ScriptKind scriptKind) -> LanguageVariant {
    // .tsx and .jsx files are treated as jsx language variant.
    return scriptKind == ScriptKind::TSX || scriptKind == ScriptKind::JSX || scriptKind == ScriptKind::JS || scriptKind == ScriptKind::JSON ? LanguageVariant::JSX : LanguageVariant::Standard;
}

inline auto  endsWith(string str, string suffix) -> boolean {
    auto expectedPos = str.length() - suffix.length();
    return expectedPos >= 0 && str.find(suffix, expectedPos) == expectedPos;
}

inline auto fileExtensionIs(string path, string extension) -> boolean {
    return path.length() > extension.length() && endsWith(path, extension);
}

template <typename T>
inline auto setTextRangePos(T range, number pos) {
    range->pos = pos;
    return range;
}

template <typename T>
inline auto setTextRangeEnd(T range, number end) -> T {
    range->end = end;
    return range;
}

template <typename T>
inline auto setTextRangePosEnd(T range, number pos, number end) {
    return setTextRangeEnd(setTextRangePos(range, pos), end);
}

template <typename T>
inline auto setTextRangePosWidth(T range, number pos, number width) {
    return setTextRangePosEnd(range, pos, pos + width);
}

template <typename T>
inline auto setTextRange(T range, TextRange location) -> T {
    return location ? setTextRangePosEnd(range, location.pos, location.end) : range;
}

template <typename T>
auto setParentRecursive(T rootNode, boolean incremental) -> T {

    auto bindParentToChildIgnoringJSDoc = [&](Node child, Node parent) -> boolean /*true is skip*/ {
        if (incremental && child.parent === parent) {
            return true;
        }
        setParent(child, parent);
        return false;
    }

    auto bindJSDoc = [&](Node child) {
        if (hasJSDocNodes(child)) {
            for (const doc : child.jsDoc) {
                bindParentToChildIgnoringJSDoc(doc, child);
                forEachChildRecursively(doc, bindParentToChildIgnoringJSDoc);
            }
        }
    }

    auto bindParentToChild = [&](Node child, Node parent) {
        return bindParentToChildIgnoringJSDoc(child, parent) || bindJSDoc(child);
    }

    if (!rootNode) return rootNode;
    forEachChildRecursively(rootNode, isJSDocNode(rootNode) ? bindParentToChildIgnoringJSDoc : bindParentToChild);
    return rootNode;
}

inline auto isKeyword(SyntaxKind token) -> boolean {
    return SyntaxKind::FirstKeyword <= token && token <= SyntaxKind::LastKeyword;
}

inline auto isTemplateLiteralKind(SyntaxKind kind) -> boolean {
    return SyntaxKind::FirstTemplateToken <= kind && kind <= SyntaxKind::LastTemplateToken;
}

inline auto isModifierKind(SyntaxKind token) -> boolean {
    switch (token) {
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

inline auto nodeIsMissing(Node node) -> boolean {
    if (node == undefined) {
        return true;
    }

    return node->pos == node->end && node->pos >= 0 && node.kind != SyntaxKind::EndOfFileToken;
}

inline auto nodeIsPresent(Node node) -> boolean {
    return !nodeIsMissing(node);
}

inline auto containsParseError(Node node) -> boolean {
    return (node->flags & NodeFlags::ThisNodeOrAnySubNodesHasError) != NodeFlags::None;
}

#endif // UTILITIES_H