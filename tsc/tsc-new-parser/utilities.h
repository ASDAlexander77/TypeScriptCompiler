#ifndef UTILITIES_H
#define UTILITIES_H

#include <functional>
#include <regex>
#include <string>

#include "core.h"
#include "enums.h"
#include "node_test.h"
#include "parser.h"
#include "scanner.h"
#include "types.h"

namespace ts
{
namespace Extension
{
static const string Ts = S(".ts");
static const string Tsx = S(".tsx");
static const string Dts = S(".d.ts");
static const string Js = S(".js");
static const string Jsx = S(".jsx");
static const string Json = S(".json");
static const string TsBuildInfo = S(".tsbuildinfo");
}; // namespace Extension

inline auto positionIsSynthesized(number pos) -> boolean
{
    // This is a fast way of testing the following conditions:
    //  pos === undefined || pos === null || isNaN(pos) || pos < 0;
    return !(pos >= 0);
}

inline auto getScriptKindFromFileName(string fileName) -> ScriptKind
{
    auto pos = fileName.rfind(S('.'));
    if (pos == string::npos)
    {
        return ScriptKind::Unknown;
    }

    auto ext = fileName.substr(pos + 1);
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
    return scriptKind != ScriptKind::Unknown ? scriptKind : scriptKind = getScriptKindFromFileName(fileName),
                                                            scriptKind != ScriptKind::Unknown ? scriptKind : ScriptKind::TS;
}

inline auto isDiagnosticWithDetachedLocation(DiagnosticRelatedInformation diagnostic) -> boolean
{
    // TODO: sort out
    // return diagnostic->start != -1 && diagnostic->length != -1 && diagnostic->fileName != S("");
    return false;
}

static auto attachFileToDiagnostic(Diagnostic diagnostic, SourceFile file) -> DiagnosticWithLocation
{
    auto fileName = file->fileName;
    auto length = file->text.length();
    // TODO: review it
    // Debug::assertEqual(diagnostic->fileName, fileName);
    Debug::assertLessThanOrEqual(diagnostic->start, length);
    Debug::assertLessThanOrEqual(diagnostic->start + diagnostic->length, length);
    DiagnosticWithLocation diagnosticWithLocation{data::DiagnosticWithLocation()};
    // TODO: review it
    // diagnosticWithLocation->file = file;
    diagnosticWithLocation->start = diagnostic->start;
    diagnosticWithLocation->length = diagnostic->length;
    diagnosticWithLocation->messageText = diagnostic->messageText;
    diagnosticWithLocation->category = diagnostic->category;
    diagnosticWithLocation->code = diagnostic->code;
    // TODO: review it
    /*
diagnosticWithLocation->reportsUnnecessary = diagnostic->reportsUnnecessary;

if (!diagnostic->relatedInformation.empty())
{
    for (auto &related : diagnostic->relatedInformation)
    {
        if (isDiagnosticWithDetachedLocation(related) && related.fileName == fileName)
        {
            Debug::assertLessThanOrEqual(related.start, length);
            Debug::assertLessThanOrEqual(related.start + related.length, length);
            diagnosticWithLocation->relatedInformation.push_back(attachFileToDiagnostic(related, file));
        }
        else
        {
            diagnosticWithLocation->relatedInformation.push_back(related);
        }
    }
}
*/

    return diagnosticWithLocation;
}

static auto attachFileToDiagnostics(std::vector<DiagnosticWithDetachedLocation> diagnostics, SourceFile file)
    -> std::vector<DiagnosticWithLocation>
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
    return string(message->message);
}

static auto createDetachedDiagnostic(string fileName, string sourceText, number start, number length, DiagnosticMessage message)
    -> DiagnosticWithDetachedLocation
{
    assertDiagnosticLocation(/*file*/ SourceFile(), start, length);
    auto text = getLocaleSpecificMessage(message);

    DiagnosticWithDetachedLocation d{data::DiagnosticWithDetachedLocation()};
    d->start = start;
    d->length = length;

    d->messageText = text;
    d->category = message->category;
    d->code = message->code;
    // d.reportsUnnecessary = message.reportsUnnecessary;
    d->fileName = fileName;

    return d;
}

// TODO: finish it with detecting index
static auto formatStringFromArgs(int& replaceIndex, string text, string arg0, string arg1) -> string {
    auto pos = text.find('{', replaceIndex);
    if (pos != std::string::npos)
    {
        auto end = text.find('}', replaceIndex + 1);
        if (end != std::string::npos)
        {
            auto index = to_number_base(text.substr(pos + 1, end - 1), 10);

            if (end != std::string::npos)
            {
                auto subText = index == 0 ? arg0 : index == 1 ? arg1 : S("");
                text.replace(pos, end - pos + 1, subText);
                replaceIndex = pos + subText.size() + 1;
            }
        }
        else
        {
            replaceIndex = pos + 1;
        }
    }

    return text;
}

// TODO: finish sourceText
static auto createDetachedDiagnostic(string fileName, string sourceText, number start, number length, DiagnosticMessage message, string arg0, string arg1 = S(""))
    -> DiagnosticWithDetachedLocation
{
    if ((start + length) > sourceText.size()) {
        length = sourceText.size() - start;
    }

    assertDiagnosticLocation(/*file*/ SourceFile(), start, length);
    auto text = getLocaleSpecificMessage(message);

    auto replaceIndex = 0;
    while (text.find('{', replaceIndex) != std::string::npos) {
        text = formatStringFromArgs(replaceIndex, text, arg0, arg1);
    }

    DiagnosticWithDetachedLocation d{data::DiagnosticWithDetachedLocation()};
    d->start = start;
    d->length = length;

    d->messageText = text;
    d->category = message->category;
    d->code = message->code;
    // diagnosticWithDetachedLocation.reportsUnnecessary = message.reportsUnnecessary;
    d->fileName = fileName;

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
    return scriptKind == ScriptKind::TSX || scriptKind == ScriptKind::JSX || scriptKind == ScriptKind::JS || scriptKind == ScriptKind::JSON
               ? LanguageVariant::JSX
               : LanguageVariant::Standard;
}

inline auto startsWith(string str, string prefix) -> boolean
{
    auto expectedPos = 0;
    return str.find(prefix, expectedPos) == expectedPos;
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

template <typename T> inline auto setTextRangePos(T range, number pos) -> T
{
    range->pos = pos;
    return range;
}

template <typename T> inline auto setTextRangeEnd(T range, number end) -> T
{
    range->_end = end;
    return range;
}

template <typename T> inline auto setTextRangePosEnd(T range, number pos, number end) -> T
{
    return setTextRangeEnd(setTextRangePos(range, pos), end);
}

template <typename T> inline auto setTextRangePosWidth(T range, number pos, number width) -> T
{
    return setTextRangePosEnd(range, pos, pos + width);
}

template <typename T> inline auto setTextRange(T range, TextRange location) -> T
{
    return !!location ? setTextRangePosEnd(range, location->pos, location->_end) : range;
}

inline static auto hasJSDocNodes(Node node) -> boolean
{
    auto jsDoc = node.template as<JSDocContainer>()->jsDoc;
    return !!jsDoc && jsDoc.size() > 0;
}

// JSDoc

/** True if node is of some JSDoc syntax kind. */
/* @internal */
inline static auto isJSDocNode(Node node) -> boolean
{
    return node >= SyntaxKind::FirstJSDocNode && node <= SyntaxKind::LastJSDocNode;
}

template <typename R = Node, typename T = Node> static auto visitNode(FuncT<R, T> cbNode, T node) -> R
{
    return node ? cbNode(node) : T{};
}

template <typename R = Node, typename T = Node>
static auto visitNodes(FuncT<R, T> cbNode, ArrayFuncT<R, T> cbNodes, NodeArray<T> nodes) -> R
{
    if (!!nodes)
    {
        if (cbNodes)
        {
            return cbNodes(nodes);
        }
        for (auto node : nodes)
        {
            auto result = cbNode(node);
            if (result)
            {
                return result;
            }
        }
    }

    return undefined;
}

template <typename R, typename T, typename U> static auto visitNodes(FuncT<R, T> cbNode, ArrayFuncT<R, T> cbNodes, NodeArray<U> nodes) -> R
{
    if (!!nodes)
    {
        if (cbNodes)
        {
            return cbNodes(NodeArray<T>(nodes));
        }
        for (auto node : nodes)
        {
            auto result = cbNode(node.template as<T>());
            if (result)
            {
                return result;
            }
        }
    }

    return undefined;
}

/**
 * Invokes a callback for each child of the given node-> The 'cbNode' callback is invoked for all child nodes
 * stored in properties. If a 'cbNodes' callback is specified, it is invoked for embedded arrays; otherwise,
 * embedded arrays are flattened and the 'cbNode' callback is invoked for each element. If a callback returns
 * a truthy value, iteration stops and that value is returned. Otherwise, undefined is returned.
 *
 * @param node a given node to visit its children
 * @param cbNode a callback to be invoked for all child nodes
 * @param cbNodes a callback to be invoked for embedded array
 *
 * @remarks `forEachChild` must visit the children of a node in the order
 * that they appear in the source code. The language service depends on this property to locate nodes by position.
 */
template <typename R = Node, typename T = Node>
static auto forEachChild(T node, FuncT<R, T> cbNode, ArrayFuncT<R, T> cbNodes = nullptr) -> R
{
    if (!node || node <= SyntaxKind::LastToken)
    {
        return R{};
    }

    // fake positive result to allow to run first command
    R result;
    auto kind = (SyntaxKind)node;
    switch (kind)
    {
    case SyntaxKind::QualifiedName:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<QualifiedName>()->left);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<QualifiedName>()->right);
        return result;
    case SyntaxKind::TypeParameter:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeParameterDeclaration>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeParameterDeclaration>()->constraint);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeParameterDeclaration>()->_default);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeParameterDeclaration>()->expression);
        return result;
    case SyntaxKind::ShorthandPropertyAssignment:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ShorthandPropertyAssignment>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ShorthandPropertyAssignment>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ShorthandPropertyAssignment>()->exclamationToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ShorthandPropertyAssignment>()->equalsToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ShorthandPropertyAssignment>()->objectAssignmentInitializer);
        return result;
    case SyntaxKind::SpreadAssignment:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<SpreadAssignment>()->expression);
        return result;
    case SyntaxKind::Parameter:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ParameterDeclaration>()->dotDotDotToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ParameterDeclaration>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ParameterDeclaration>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ParameterDeclaration>()->type);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ParameterDeclaration>()->initializer);
        return result;
    case SyntaxKind::PropertyDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyDeclaration>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyDeclaration>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyDeclaration>()->exclamationToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyDeclaration>()->type);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyDeclaration>()->initializer);
        return result;
    case SyntaxKind::PropertySignature:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertySignature>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertySignature>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertySignature>()->type);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertySignature>()->initializer);
        return result;
    case SyntaxKind::PropertyAssignment:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyAssignment>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyAssignment>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyAssignment>()->initializer);
        return result;
    case SyntaxKind::VariableDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<VariableDeclaration>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<VariableDeclaration>()->exclamationToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<VariableDeclaration>()->type);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<VariableDeclaration>()->initializer);
        return result;
    case SyntaxKind::BindingElement:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<BindingElement>()->dotDotDotToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<BindingElement>()->propertyName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<BindingElement>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<BindingElement>()->initializer);
        return result;
    case SyntaxKind::FunctionType:
    case SyntaxKind::ConstructorType:
    case SyntaxKind::CallSignature:
    case SyntaxKind::ConstructSignature:
    case SyntaxKind::IndexSignature:
    case SyntaxKind::MethodSignature:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (kind == SyntaxKind::MethodSignature && !result)
            result = visitNode<R, T>(cbNode, node.template as<SignatureDeclarationBase>()->name);
        if (kind == SyntaxKind::MethodSignature && !result)
            result = visitNode<R, T>(cbNode, node.template as<SignatureDeclarationBase>()->questionToken);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<SignatureDeclarationBase>()->typeParameters);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<SignatureDeclarationBase>()->parameters);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<SignatureDeclarationBase>()->type);
        return result;
    case SyntaxKind::MethodDeclaration:
    case SyntaxKind::Constructor:
    case SyntaxKind::GetAccessor:
    case SyntaxKind::SetAccessor:
    case SyntaxKind::FunctionExpression:
    case SyntaxKind::FunctionDeclaration:
    case SyntaxKind::ArrowFunction:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<FunctionLikeDeclarationBase>()->asteriskToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<FunctionLikeDeclarationBase>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<FunctionLikeDeclarationBase>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<FunctionLikeDeclarationBase>()->exclamationToken);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<FunctionLikeDeclarationBase>()->typeParameters);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<FunctionLikeDeclarationBase>()->parameters);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<FunctionLikeDeclarationBase>()->type);
        if (kind == SyntaxKind::ArrowFunction && !result)
            result = visitNode<R, T>(cbNode, node.template as<ArrowFunction>()->equalsGreaterThanToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<FunctionLikeDeclarationBase>()->body);
        return result;
    case SyntaxKind::ClassStaticBlockDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);        
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ClassStaticBlockDeclaration>()->body);
    case SyntaxKind::TypeReference:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeReferenceNode>()->typeName);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<TypeReferenceNode>()->typeArguments);
        return result;
    case SyntaxKind::TypePredicate:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypePredicateNode>()->assertsModifier);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypePredicateNode>()->parameterName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypePredicateNode>()->type);
        return result;
    case SyntaxKind::TypeQuery:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeQueryNode>()->exprName);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<TypeQueryNode>()->typeArguments);
        return result;
    case SyntaxKind::TypeLiteral:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<TypeLiteralNode>()->members);
        return result;
    case SyntaxKind::ArrayType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ArrayTypeNode>()->elementType);
        return result;
    case SyntaxKind::TupleType:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<TupleTypeNode>()->elements);
        return result;
    case SyntaxKind::UnionType:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<UnionTypeNode>()->types);
        return result;
    case SyntaxKind::IntersectionType:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<IntersectionTypeNode>()->types);
        return result;
    case SyntaxKind::ConditionalType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalTypeNode>()->checkType);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalTypeNode>()->extendsType);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalTypeNode>()->trueType);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalTypeNode>()->falseType);
        return result;
    case SyntaxKind::InferType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<InferTypeNode>()->typeParameter);
        return result;
    case SyntaxKind::ImportType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportTypeNode>()->argument);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportTypeNode>()->attributes);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportTypeNode>()->qualifier);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ImportTypeNode>()->typeArguments);
        return result;
    case SyntaxKind::ParenthesizedType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ParenthesizedTypeNode>()->type);
        return result;
    case SyntaxKind::TypeOperator:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeOperatorNode>()->type);
        return result;
    case SyntaxKind::IndexedAccessType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<IndexedAccessTypeNode>()->objectType);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<IndexedAccessTypeNode>()->indexType);
        return result;
    case SyntaxKind::MappedType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<MappedTypeNode>()->readonlyToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<MappedTypeNode>()->typeParameter);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<MappedTypeNode>()->nameType);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<MappedTypeNode>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<MappedTypeNode>()->type);
        return result;
    case SyntaxKind::LiteralType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<LiteralTypeNode>()->literal);
        return result;
    case SyntaxKind::NamedTupleMember:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NamedTupleMember>()->dotDotDotToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NamedTupleMember>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NamedTupleMember>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NamedTupleMember>()->type);
        return result;
    case SyntaxKind::ObjectBindingPattern:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ObjectBindingPattern>()->elements);
        return result;
    case SyntaxKind::ArrayBindingPattern:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ArrayBindingPattern>()->elements);
        return result;
    case SyntaxKind::ArrayLiteralExpression:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ArrayLiteralExpression>()->elements);
        return result;
    case SyntaxKind::ObjectLiteralExpression:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ObjectLiteralExpression>()->properties);
        return result;
    case SyntaxKind::PropertyAccessExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyAccessExpression>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyAccessExpression>()->questionDotToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PropertyAccessExpression>()->name);
        return result;
    case SyntaxKind::ElementAccessExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ElementAccessExpression>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ElementAccessExpression>()->questionDotToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ElementAccessExpression>()->argumentExpression);
        return result;
    case SyntaxKind::CallExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<CallExpression>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<CallExpression>()->questionDotToken);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<CallExpression>()->typeArguments);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<CallExpression>()->arguments);
        return result;
    case SyntaxKind::NewExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NewExpression>()->expression);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<NewExpression>()->typeArguments);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<NewExpression>()->arguments);
        return result;
    case SyntaxKind::TaggedTemplateExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TaggedTemplateExpression>()->tag);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TaggedTemplateExpression>()->questionDotToken);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<TaggedTemplateExpression>()->typeArguments);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TaggedTemplateExpression>()->_template);
        return result;
    case SyntaxKind::TypeAssertionExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeAssertion>()->type);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeAssertion>()->expression);
        return result;
    case SyntaxKind::ParenthesizedExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ParenthesizedExpression>()->expression);
        return result;
    case SyntaxKind::DeleteExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<DeleteExpression>()->expression);
        return result;
    case SyntaxKind::TypeOfExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeOfExpression>()->expression);
        return result;
    case SyntaxKind::VoidExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<VoidExpression>()->expression);
        return result;
    case SyntaxKind::PrefixUnaryExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PrefixUnaryExpression>()->operand);
        return result;
    case SyntaxKind::YieldExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<YieldExpression>()->asteriskToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<YieldExpression>()->expression);
        return result;
    case SyntaxKind::AwaitExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<AwaitExpression>()->expression);
        return result;
    case SyntaxKind::PostfixUnaryExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PostfixUnaryExpression>()->operand);
        return result;
    case SyntaxKind::BinaryExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<BinaryExpression>()->left);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<BinaryExpression>()->operatorToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<BinaryExpression>()->right);
        return result;
    case SyntaxKind::AsExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<AsExpression>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<AsExpression>()->type);
        return result;
    case SyntaxKind::NonNullExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NonNullExpression>()->expression);
        return result;
    case SyntaxKind::SatisfiesExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<SatisfiesExpression>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<SatisfiesExpression>()->type);
        return result;
    case SyntaxKind::MetaProperty:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<MetaProperty>()->name);
        return result;
    case SyntaxKind::ConditionalExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalExpression>()->condition);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalExpression>()->questionToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalExpression>()->whenTrue);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalExpression>()->colonToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ConditionalExpression>()->whenFalse);
        return result;
    case SyntaxKind::SpreadElement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<SpreadElement>()->expression);
        return result;
    case SyntaxKind::Block:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<Block>()->statements);
        return result;
    case SyntaxKind::ModuleBlock:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ModuleBlock>()->statements);
        return result;
    case SyntaxKind::SourceFile:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<SourceFile>()->statements);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<SourceFile>()->endOfFileToken);
        return result;
    case SyntaxKind::VariableStatement:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<VariableStatement>()->declarationList);
        return result;
    case SyntaxKind::VariableDeclarationList:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<VariableDeclarationList>()->declarations);
        return result;
    case SyntaxKind::ExpressionStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExpressionStatement>()->expression);
        return result;
    case SyntaxKind::IfStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<IfStatement>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<IfStatement>()->thenStatement);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<IfStatement>()->elseStatement);
        return result;
    case SyntaxKind::DoStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<DoStatement>()->statement);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<DoStatement>()->expression);
        return result;
    case SyntaxKind::WhileStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<WhileStatement>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<WhileStatement>()->statement);
        return result;
    case SyntaxKind::ForStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForStatement>()->initializer);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForStatement>()->condition);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForStatement>()->incrementor);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForStatement>()->statement);
        return result;
    case SyntaxKind::ForInStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForInStatement>()->initializer);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForInStatement>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForInStatement>()->statement);
        return result;
    case SyntaxKind::ForOfStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForOfStatement>()->awaitModifier);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForOfStatement>()->initializer);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForOfStatement>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ForOfStatement>()->statement);
        return result;
    case SyntaxKind::ContinueStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ContinueStatement>()->label);
        return result;
    case SyntaxKind::BreakStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<BreakStatement>()->label);
        return result;
    case SyntaxKind::ReturnStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ReturnStatement>()->expression);
        return result;
    case SyntaxKind::WithStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<WithStatement>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<WithStatement>()->statement);
        return result;
    case SyntaxKind::SwitchStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<SwitchStatement>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<SwitchStatement>()->caseBlock);
        return result;
    case SyntaxKind::CaseBlock:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<CaseBlock>()->clauses);
        return result;
    case SyntaxKind::CaseClause:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<CaseClause>()->expression);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<CaseClause>()->statements);
        return result;
    case SyntaxKind::DefaultClause:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<DefaultClause>()->statements);
        return result;
    case SyntaxKind::LabeledStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<LabeledStatement>()->label);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<LabeledStatement>()->statement);
        return result;
    case SyntaxKind::ThrowStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ThrowStatement>()->expression);
        return result;
    case SyntaxKind::TryStatement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TryStatement>()->tryBlock);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TryStatement>()->catchClause);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TryStatement>()->finallyBlock);
        return result;
    case SyntaxKind::CatchClause:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<CatchClause>()->variableDeclaration);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<CatchClause>()->block);
        return result;
    case SyntaxKind::Decorator:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<Decorator>()->expression);
        return result;
    case SyntaxKind::ClassDeclaration:
    case SyntaxKind::ClassExpression:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ClassLikeDeclaration>()->name);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ClassLikeDeclaration>()->typeParameters);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ClassLikeDeclaration>()->heritageClauses);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ClassLikeDeclaration>()->members);
        return result;
    case SyntaxKind::InterfaceDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<InterfaceDeclaration>()->name);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<InterfaceDeclaration>()->typeParameters);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<InterfaceDeclaration>()->heritageClauses);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<InterfaceDeclaration>()->members);
        return result;
    case SyntaxKind::TypeAliasDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeAliasDeclaration>()->name);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<TypeAliasDeclaration>()->typeParameters);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TypeAliasDeclaration>()->type);
        return result;
    case SyntaxKind::EnumDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<EnumDeclaration>()->name);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<EnumDeclaration>()->members);
        return result;
    case SyntaxKind::EnumMember:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<EnumMember>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<EnumMember>()->initializer);
        return result;
    case SyntaxKind::ModuleDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ModuleDeclaration>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ModuleDeclaration>()->body);
        return result;
    case SyntaxKind::ImportEqualsDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportEqualsDeclaration>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportEqualsDeclaration>()->moduleReference);
        return result;
    case SyntaxKind::ImportDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportDeclaration>()->importClause);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportDeclaration>()->moduleSpecifier);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportDeclaration>()->attributes);            
        return result;
    case SyntaxKind::ImportClause:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportClause>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportClause>()->namedBindings);
        return result;
    case SyntaxKind::NamespaceExportDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);    
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NamespaceExportDeclaration>()->name);
        return result;

    case SyntaxKind::NamespaceImport:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NamespaceImport>()->name);
        return result;
    case SyntaxKind::NamespaceExport:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<NamespaceExport>()->name);
        return result;
    case SyntaxKind::NamedImports:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<NamedImports>()->elements);
        return result;
    case SyntaxKind::NamedExports:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<NamedExports>()->elements);
        return result;
    case SyntaxKind::ExportDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExportDeclaration>()->exportClause);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExportDeclaration>()->moduleSpecifier);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExportDeclaration>()->attributes);            
        return result;
    case SyntaxKind::ImportSpecifier:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportSpecifier>()->propertyName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ImportSpecifier>()->name);
        return result;
    case SyntaxKind::ExportSpecifier:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExportSpecifier>()->propertyName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExportSpecifier>()->name);
        return result;
    case SyntaxKind::ExportAssignment:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExportAssignment>()->expression);
        return result;
    case SyntaxKind::TemplateExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TemplateExpression>()->head);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<TemplateExpression>()->templateSpans);
        return result;
    case SyntaxKind::TemplateSpan:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TemplateSpan>()->expression);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TemplateSpan>()->literal);
        return result;
    case SyntaxKind::TemplateLiteralType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TemplateLiteralTypeNode>()->head);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<TemplateLiteralTypeNode>()->templateSpans);
        return result;
    case SyntaxKind::TemplateLiteralTypeSpan:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TemplateLiteralTypeSpan>()->type);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<TemplateLiteralTypeSpan>()->literal);
        return result;
    case SyntaxKind::ComputedPropertyName:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ComputedPropertyName>()->expression);
        return result;
    case SyntaxKind::HeritageClause:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<HeritageClause>()->types);
        return result;
    case SyntaxKind::ExpressionWithTypeArguments:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExpressionWithTypeArguments>()->expression);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<ExpressionWithTypeArguments>()->typeArguments);
        return result;
    case SyntaxKind::ExternalModuleReference:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<ExternalModuleReference>()->expression);
        return result;
    case SyntaxKind::MissingDeclaration:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node->modifiers);
        return result;
    case SyntaxKind::CommaListExpression:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<CommaListExpression>()->elements);
        return result;

    case SyntaxKind::JsxElement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxElement>()->openingElement);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<JsxElement>()->children);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxElement>()->closingElement);
        return result;
    case SyntaxKind::JsxFragment:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxFragment>()->openingFragment);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<JsxFragment>()->children);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxFragment>()->closingFragment);
        return result;
    case SyntaxKind::JsxSelfClosingElement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxSelfClosingElement>()->tagName);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<JsxSelfClosingElement>()->typeArguments);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxSelfClosingElement>()->attributes);
        return result;
    case SyntaxKind::JsxOpeningElement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxOpeningElement>()->tagName);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<JsxOpeningElement>()->typeArguments);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxOpeningElement>()->attributes);
        return result;
    case SyntaxKind::JsxAttributes:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<JsxAttributes>()->properties);
        return result;
    case SyntaxKind::JsxAttribute:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxAttribute>()->name);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxAttribute>()->initializer);
        return result;
    case SyntaxKind::JsxSpreadAttribute:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxSpreadAttribute>()->expression);
        return result;
    case SyntaxKind::JsxExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxExpression>()->dotDotDotToken);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxExpression>()->expression);
        return result;
    case SyntaxKind::JsxClosingElement:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JsxClosingElement>()->tagName);
        return result;

    case SyntaxKind::OptionalType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<OptionalTypeNode>()->type);
        return result;
    case SyntaxKind::RestType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<RestTypeNode>()->type);
        return result;
    case SyntaxKind::JSDocTypeExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTypeExpression>()->type);
        return result;
    case SyntaxKind::JSDocNonNullableType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocNonNullableType>()->type);
        return result;
    case SyntaxKind::JSDocNullableType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocNullableType>()->type);
        return result;
    case SyntaxKind::JSDocOptionalType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocOptionalType>()->type);
        return result;
    case SyntaxKind::JSDocVariadicType:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocVariadicType>()->type);
        return result;
    case SyntaxKind::JSDocFunctionType:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<JSDocFunctionType>()->parameters);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocFunctionType>()->type);
        return result;
    case SyntaxKind::JSDocComment:
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<JSDoc>()->tags);
        return result;
    case SyntaxKind::JSDocSeeTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocSeeTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocSeeTag>()->name);
        return result;
    case SyntaxKind::JSDocNameReference:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocNameReference>()->name);
        return result;
    case SyntaxKind::JSDocParameterTag:
    case SyntaxKind::JSDocPropertyTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (node.template as<JSDocPropertyLikeTag>()->isNameFirst)
        {
            if (!result)
                result = visitNode<R, T>(cbNode, node.template as<JSDocPropertyLikeTag>()->name);
            if (!result)
                result = visitNode<R, T>(cbNode, node.template as<JSDocPropertyLikeTag>()->typeExpression);
        }
        else
        {
            if (!result)
                result = visitNode<R, T>(cbNode, node.template as<JSDocPropertyLikeTag>()->typeExpression);
            if (!result)
                result = visitNode<R, T>(cbNode, node.template as<JSDocPropertyLikeTag>()->name);
        }
        return result;
    case SyntaxKind::JSDocAuthorTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        return result;
    case SyntaxKind::JSDocImplementsTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocImplementsTag>()->_class);
        return result;
    case SyntaxKind::JSDocAugmentsTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocAugmentsTag>()->_class);
        return result;
    case SyntaxKind::JSDocTemplateTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTemplateTag>()->constraint);
        if (!result)
            result = visitNodes(cbNode, cbNodes, node.template as<JSDocTemplateTag>()->typeParameters);
        return result;
    case SyntaxKind::JSDocTypedefTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (node.template as<JSDocTypedefTag>()->typeExpression &&
            node.template as<JSDocTypedefTag>()->typeExpression == SyntaxKind::JSDocTypeExpression)
        {
            if (!result)
                result = visitNode<R, T>(cbNode, node.template as<JSDocTypedefTag>()->typeExpression);
            if (!result)
                result = visitNode<R, T>(cbNode, node.template as<JSDocTypedefTag>()->fullName);
        }
        else
        {
            if (!result)
                result = visitNode<R, T>(cbNode, node.template as<JSDocTypedefTag>()->fullName);
            if (!result)
                result = visitNode<R, T>(cbNode, node.template as<JSDocTypedefTag>()->typeExpression);
        }
        return result;
    case SyntaxKind::JSDocCallbackTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocCallbackTag>()->fullName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocCallbackTag>()->typeExpression);
        return result;
    case SyntaxKind::JSDocReturnTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocReturnTag>()->typeExpression);
        return result;
    case SyntaxKind::JSDocTypeTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTypeTag>()->typeExpression);
        return result;
    case SyntaxKind::JSDocThisTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocThisTag>()->typeExpression);
        return result;
    case SyntaxKind::JSDocEnumTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocEnumTag>()->typeExpression);
        return result;
    case SyntaxKind::JSDocSignature:
        return forEach<decltype(node.template as<JSDocSignature>()->typeParameters), T>(node.template as<JSDocSignature>()->typeParameters,
                                                                                        cbNode);
        forEach<decltype(node.template as<JSDocSignature>()->parameters), T>(node.template as<JSDocSignature>()->parameters, cbNode);
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocSignature>()->type);
        return result;
    case SyntaxKind::JSDocTypeLiteral:
        return forEach<decltype(node.template as<JSDocTypeLiteral>()->jsDocPropertyTags), T>(
            node.template as<JSDocTypeLiteral>()->jsDocPropertyTags, cbNode);
        return result;
    case SyntaxKind::JSDocTag:
    case SyntaxKind::JSDocClassTag:
    case SyntaxKind::JSDocPublicTag:
    case SyntaxKind::JSDocPrivateTag:
    case SyntaxKind::JSDocProtectedTag:
    case SyntaxKind::JSDocReadonlyTag:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<JSDocTag>()->tagName);
        return result;
    case SyntaxKind::PartiallyEmittedExpression:
        if (!result)
            result = visitNode<R, T>(cbNode, node.template as<PartiallyEmittedExpression>()->expression);
        return result;
    }

    return R{};
}

static auto gatherPossibleChildren(Node node) -> NodeArray<Node>
{
    NodeArray<Node> children;

    auto addWorkItem = [&](Node n) -> Node {
        children.emplace(children.begin(), n);
        return undefined;
    };

    auto addWorkItems = [&](NodeArray<Node> arr) -> Node {
        for (auto n : arr)
        {
            children.emplace(children.begin(), n);
        }

        return undefined;
    };

    forEachChild<Node, Node>(node, addWorkItem,
                             addWorkItems); // By using a stack above and `unshift` here, we emulate a depth-first preorder traversal
    return children;
}

/** @internal */
/**
 * Invokes a callback for each child of the given node-> The 'cbNode' callback is invoked for all child nodes
 * stored in properties. If a 'cbNodes' callback is specified, it is invoked for embedded arrays; additionally,
 * unlike `forEachChild`, embedded arrays are flattened and the 'cbNode' callback is invoked for each element.
 *  If a callback returns a truthy value, iteration stops and that value is returned. Otherwise, undefined is returned.
 *
 * @param node a given node to visit its children
 * @param cbNode a callback to be invoked for all child nodes
 * @param cbNodes a callback to be invoked for embedded array
 *
 * @remarks Unlike `forEachChild`, `forEachChildRecursively` handles recursively invoking the traversal on each child node found,
 * and while doing so, handles traversing the structure without relying on the callstack to encode the tree structure.
 */
template <typename R = Node, typename T = Node>
static auto forEachChildRecursively(T rootNode, FuncWithParentT<R, T> cbNode, ArrayFuncWithParentT<R, T> cbNodes = nullptr) -> R
{
    auto queue = gatherPossibleChildren(rootNode);
    NodeArray<Node> parents; // tracks parent references for elements in queue
    while (parents.size() < queue.size())
    {
        parents.push_back(rootNode);
    }
    while (queue.size() != 0)
    {
        auto current = queue.pop();
        auto parent = parents.pop();
        // TODO: review it
        /*
    if (isArray(current)) {
        if (cbNodes) {
            auto res = cbNodes(current.asArray<boolean>(), parent);
            if (res) {
                //TODO: review it
                //if (res == "skip") continue;
                return res;
            }
        }
        for (int i = current.size() - 1; i >= 0; --i) {
            queue.push_back(current[i]);
            parents.push_back(parent);
        }
    }
    else {
    */
        auto res = cbNode(current, parent);
        if (res)
        {
            // TODO: review it
            // if (res == "skip") continue;
            return res;
        }
        if (current >= SyntaxKind::FirstNode)
        {
            // add children in reverse order to the queue, so popping gives the first child
            for (auto child : gatherPossibleChildren(current))
            {
                queue.push_back(child);
                parents.push_back(current);
            }
        }
        //}
    }

    return false;
}

inline static auto setParent(Node child, Node parent) -> Node
{
    if (!!child && !!parent)
    {
        child->parent = parent;
    }

    return child;
}

template <typename R, typename T> static auto setParentRecursive(T rootNode, boolean incremental) -> R
{
    auto bindParentToChildIgnoringJSDoc = [&](auto child, auto parent) /*true is skip*/ {
        if (incremental && child->parent == parent)
        {
            return true;
        }
        setParent(child, parent);
        return false;
    };

    auto bindJSDoc = [&](auto child) {
        if (hasJSDocNodes(child))
        {
            for (auto &doc : child.template as<JSDocContainer>()->jsDoc)
            {
                bindParentToChildIgnoringJSDoc(doc, child);
                forEachChildRecursively<boolean, T>(doc, bindParentToChildIgnoringJSDoc);
            }
        }

        return false;
    };

    auto bindParentToChild = [&](Node child, Node parent) { return bindParentToChildIgnoringJSDoc(child, parent) || bindJSDoc(child); };

    if (!rootNode)
    {
        return rootNode;
    }

    if (isJSDocNode(rootNode))
        forEachChildRecursively<boolean, T>(rootNode, bindParentToChildIgnoringJSDoc);
    else
        forEachChildRecursively<boolean, T>(rootNode, bindParentToChild);
    return rootNode;
}

inline auto isKeyword(SyntaxKind token) -> boolean
{
    return SyntaxKind::FirstKeyword <= token && token <= SyntaxKind::LastKeyword;
}

inline auto isPunctuation(SyntaxKind token) -> boolean {
    return SyntaxKind::FirstPunctuation <= token && token <= SyntaxKind::LastPunctuation;
}

inline auto isKeywordOrPunctuation(SyntaxKind token) -> boolean {
    return isKeyword(token) || isPunctuation(token);
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
        case SyntaxKind::AccessorKeyword:
        case SyntaxKind::AsyncKeyword:
        case SyntaxKind::ConstKeyword:
        case SyntaxKind::DeclareKeyword:
        case SyntaxKind::DefaultKeyword:
        case SyntaxKind::ExportKeyword:
        case SyntaxKind::InKeyword:
        case SyntaxKind::PublicKeyword:
        case SyntaxKind::PrivateKeyword:
        case SyntaxKind::ProtectedKeyword:
        case SyntaxKind::ReadonlyKeyword:
        case SyntaxKind::StaticKeyword:
        case SyntaxKind::OutKeyword:
        case SyntaxKind::OverrideKeyword:
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

    return node->pos == node->_end && node->pos >= 0 && node != SyntaxKind::EndOfFileToken;
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
    return node->_end - node->pos;
}

inline auto isOuterExpression(Node node, OuterExpressionKinds kinds = OuterExpressionKinds::All) -> boolean
{
    switch ((SyntaxKind)node)
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
        switch ((SyntaxKind)node)
        {
        case SyntaxKind::ParenthesizedExpression:
            node = node.template as<ParenthesizedExpression>()->expression;
            break;
        case SyntaxKind::TypeAssertionExpression:
            node = node.template as<TypeAssertion>()->expression;
            break;
        case SyntaxKind::AsExpression:
            node = node.template as<AsExpression>()->expression;
            break;
        case SyntaxKind::NonNullExpression:
            node = node.template as<NonNullExpression>()->expression;
            break;
        case SyntaxKind::PartiallyEmittedExpression:
            node = node.template as<PartiallyEmittedExpression>()->expression;
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
        case SyntaxKind::PrivateIdentifier: // technically this is only an Expression if it's in a `#field in expr` BinaryExpression
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
        case SyntaxKind::ExpressionWithTypeArguments:
        case SyntaxKind::MetaProperty:
        case SyntaxKind::ImportKeyword: // technically this is only an Expression if it's in a CallExpression
        case SyntaxKind::MissingDeclaration:
        return true;
    default:
        return false;
    }
}

inline auto isLeftHandSideExpression(Node node) -> boolean
{
    return isLeftHandSideExpressionKind(skipPartiallyEmittedExpressions(node));
}

inline auto isAssignmentOperator(SyntaxKind token) -> boolean
{
    return token >= SyntaxKind::FirstAssignment && token <= SyntaxKind::LastAssignment;
}

inline auto getBinaryOperatorPrecedence(SyntaxKind kind) -> OperatorPrecedence
{
    switch (kind)
    {
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
    case SyntaxKind::SatisfiesKeyword:
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

static auto findAncestor(Node node, std::function<boolean(Node)> callback) -> Node
{
    while (node)
    {
        auto result = callback(node);
        if (result)
        {
            return node;
        }
        node = node->parent;
    }
    return undefined;
}

static auto isJSDocTypeExpressionOrChild(Node node) -> boolean
{
    return !!findAncestor(node, [](Node n) { return isJSDocTypeExpression(n); });
}

static auto getTextOfNodeFromSourceText(safe_string sourceText, Node node, boolean includeTrivia = false, ts::Scanner *scanner = nullptr)
    -> string
{
    if (nodeIsMissing(node))
    {
        return string();
    }

    auto text = sourceText.substring(
        includeTrivia ? static_cast<number>(node->pos) : scanner->skipTrivia(sourceText, static_cast<number>(node->pos)), node->_end);

    if (isJSDocTypeExpressionOrChild(node))
    {
        // strip space + asterisk at line start
        auto reg = regex(S("(^|\\r?\\n|\\r)\\s*\\*\\s*"), std::regex_constants::extended);
        text = regex_replace(text, reg, S("$1"));
    }

    return text;
}

static auto isStringLiteralLike(Node node) -> boolean
{
    return node == SyntaxKind::StringLiteral || node == SyntaxKind::NoSubstitutionTemplateLiteral;
}

static auto isStringOrNumericLiteralLike(Node node) -> boolean
{
    return isStringLiteralLike(node) || isNumericLiteral(node);
}

template <typename T> static auto addRelatedInfo(T diagnostic, std::vector<DiagnosticRelatedInformation> relatedInformation) -> T
{
    if (!relatedInformation.size())
    {
        return diagnostic;
    }
    if (diagnostic.relatedInformation.size() > 0)
    {
        diagnostic.relatedInformation.clear();
    }
    Debug::_assert(diagnostic.relatedInformation.size() != 0,
                   S("Diagnostic had empty array singleton for related info, but is still being constructed!"));
    for (auto &item : relatedInformation)
    {
        diagnostic.relatedInformation.push_back(item);
    }

    return diagnostic;
}

template <typename T> auto addRelatedInfo(T diagnostic, DiagnosticRelatedInformation relatedInformation) -> T
{
    if (!relatedInformation)
    {
        return diagnostic;
    }
    if (diagnostic->relatedInformation.size() > 0)
    {
        diagnostic->relatedInformation.clear();
    }
    // TODO: review next assert
    // Debug::_assert(diagnostic->relatedInformation.size() != 0, S("Diagnostic had empty array singleton for related info, but is still
    // being constructed!"));
    diagnostic->relatedInformation.push_back(relatedInformation);

    return diagnostic;
}

static auto modifierToFlag(SyntaxKind token) -> ModifierFlags
{
    switch (token)
    {
    case SyntaxKind::StaticKeyword:
        return ModifierFlags::Static;
    case SyntaxKind::PublicKeyword:
        return ModifierFlags::Public;
    case SyntaxKind::ProtectedKeyword:
        return ModifierFlags::Protected;
    case SyntaxKind::PrivateKeyword:
        return ModifierFlags::Private;
    case SyntaxKind::AbstractKeyword:
        return ModifierFlags::Abstract;
    case SyntaxKind::ExportKeyword:
        return ModifierFlags::Export;
    case SyntaxKind::DeclareKeyword:
        return ModifierFlags::Ambient;
    case SyntaxKind::ConstKeyword:
        return ModifierFlags::Const;
    case SyntaxKind::DefaultKeyword:
        return ModifierFlags::Default;
    case SyntaxKind::AsyncKeyword:
        return ModifierFlags::Async;
    case SyntaxKind::ReadonlyKeyword:
        return ModifierFlags::Readonly;
    }
    return ModifierFlags::None;
}

static auto modifiersToFlags(ModifiersLikeArray modifiers) -> ModifierFlags
{
    auto flags = ModifierFlags::None;
    if (!!modifiers)
    {
        for (auto &modifier : modifiers)
        {
            flags |= modifierToFlag(modifier);
        }
    }
    return flags;
}

inline auto isParameterPropertyModifier(SyntaxKind kind) -> boolean
{
    return !!(modifierToFlag(kind) & ModifierFlags::ParameterPropertyModifier);
}

inline auto isClassMemberModifier(SyntaxKind idToken) -> boolean
{
    return isParameterPropertyModifier(idToken) || idToken == SyntaxKind::StaticKeyword;
}

inline static auto isNamedDeclaration(Node node) -> boolean
{
    return !!node.is<NamedDeclaration>(); // A 'name' property should always be a DeclarationName.
}

inline static auto isPropertyName(Node node) -> boolean
{
    auto kind = (SyntaxKind)node;
    return kind == SyntaxKind::Identifier || kind == SyntaxKind::PrivateIdentifier || kind == SyntaxKind::StringLiteral ||
           kind == SyntaxKind::NumericLiteral || kind == SyntaxKind::ComputedPropertyName;
}

inline static auto escapeLeadingUnderscores(string identifier) -> string
{
    return (identifier.size() >= 2 && (CharacterCodes)identifier[0] == CharacterCodes::_ &&
                    (CharacterCodes)identifier[1] == CharacterCodes::_
                ? S("_") + identifier
                : identifier);
}

inline static auto isCommaSequence(Expression node) -> boolean
{
    return node == SyntaxKind::BinaryExpression && node.template as<BinaryExpression>()->operatorToken == SyntaxKind::CommaToken ||
           node == SyntaxKind::CommaListExpression;
}

inline static auto isIdentifierTypePredicate(TypePredicateNode predicate) -> boolean
{
    return predicate && predicate == SyntaxKind::Identifier;
}

inline static auto identifierIsThisKeyword(Identifier id) -> boolean
{
    return id->originalKeywordKind == SyntaxKind::ThisKeyword;
}

inline static auto isThisIdentifier(Node node) -> boolean
{
    return !!node && node == SyntaxKind::Identifier && identifierIsThisKeyword(node.template as<Identifier>());
}

inline static auto isInJSFile(Node node) -> boolean
{
    return !!node && !!(node->flags & NodeFlags::JavaScriptFile);
}

inline static auto getSyntacticModifierFlagsNoCache(Node node) -> ModifierFlags
{
    auto flags = modifiersToFlags(node->modifiers);
    if (!!(node->flags & NodeFlags::NestedNamespace) ||
        (node == SyntaxKind::Identifier && node.template as<Identifier>()->isInJSDocNamespace))
    {
        flags |= ModifierFlags::Export;
    }
    return flags;
}

inline static auto getJSDocTagsWorker(Node node, boolean noCache = false) -> NodeArray<JSDocTag>
{
    auto tags = node.template as<JSDocContainer>()->jsDocCache;
    // If cache is 'null', that means we did the work of searching for JSDoc tags and came up with nothing.
    if (tags == undefined || noCache)
    {
        // TODO: finish it
        /*
        auto comments = getJSDocCommentsAndTags(node, noCache);
        Debug::_assert(comments.length < 2 || comments[0] != comments[1]);
        tags = flatMap(comments, j => isJSDoc(j) ? j.tags : j);
        */
        if (!noCache)
        {
            node.template as<JSDocContainer>()->jsDocCache = tags;
        }
    }
    return tags;
}

inline static auto getFirstJSDocTag(Node node, std::function<boolean(Node)> predicate_, boolean noCache = false) -> Node
{
    return find(getJSDocTagsWorker(node, noCache), predicate_);
}

inline static auto getJSDocPublicTagNoCache(Node node) -> JSDocPublicTag
{
    return getFirstJSDocTag(
        node, [](Node node) { return isJSDocPublicTag(node); }, /*noCache*/ true);
}

inline static auto getJSDocPrivateTagNoCache(Node node) -> JSDocPublicTag
{
    return getFirstJSDocTag(
        node, [](Node node) { return isJSDocPrivateTag(node); }, /*noCache*/ true);
}

inline static auto getJSDocProtectedTagNoCache(Node node) -> JSDocProtectedTag
{
    return getFirstJSDocTag(
        node, [](Node node) { return isJSDocProtectedTag(node); }, /*noCache*/ true);
}

inline static auto getJSDocReadonlyTagNoCache(Node node) -> JSDocReadonlyTag
{
    return getFirstJSDocTag(
        node, [](Node node) { return isJSDocReadonlyTag(node); }, /*noCache*/ true);
}

inline static auto getJSDocDeprecatedTagNoCache(Node node) -> JSDocDeprecatedTag
{
    return getFirstJSDocTag(
        node, [](Node node) { return isJSDocDeprecatedTag(node); }, /*noCache*/ true);
}

inline static auto getJSDocModifierFlagsNoCache(Node node) -> ModifierFlags
{
    auto flags = ModifierFlags::None;
    if (!!node->parent && !isParameter(node))
    {
        if (isInJSFile(node))
        {
            if (getJSDocPublicTagNoCache(node))
                flags |= ModifierFlags::Public;
            if (getJSDocPrivateTagNoCache(node))
                flags |= ModifierFlags::Private;
            if (getJSDocProtectedTagNoCache(node))
                flags |= ModifierFlags::Protected;
            if (getJSDocReadonlyTagNoCache(node))
                flags |= ModifierFlags::Readonly;
        }
        if (getJSDocDeprecatedTagNoCache(node))
            flags |= ModifierFlags::Deprecated;
    }

    return flags;
}

/** @internal */
inline static auto canHaveJSDoc(Node node) -> boolean {
    switch ((SyntaxKind)node) {
        case SyntaxKind::ArrowFunction:
        case SyntaxKind::BinaryExpression:
        case SyntaxKind::Block:
        case SyntaxKind::BreakStatement:
        case SyntaxKind::CallSignature:
        case SyntaxKind::CaseClause:
        case SyntaxKind::ClassDeclaration:
        case SyntaxKind::ClassExpression:
        case SyntaxKind::ClassStaticBlockDeclaration:
        case SyntaxKind::Constructor:
        case SyntaxKind::ConstructorType:
        case SyntaxKind::ConstructSignature:
        case SyntaxKind::ContinueStatement:
        case SyntaxKind::DebuggerStatement:
        case SyntaxKind::DoStatement:
        case SyntaxKind::ElementAccessExpression:
        case SyntaxKind::EmptyStatement:
        case SyntaxKind::EndOfFileToken:
        case SyntaxKind::EnumDeclaration:
        case SyntaxKind::EnumMember:
        case SyntaxKind::ExportAssignment:
        case SyntaxKind::ExportDeclaration:
        case SyntaxKind::ExportSpecifier:
        case SyntaxKind::ExpressionStatement:
        case SyntaxKind::ForInStatement:
        case SyntaxKind::ForOfStatement:
        case SyntaxKind::ForStatement:
        case SyntaxKind::FunctionDeclaration:
        case SyntaxKind::FunctionExpression:
        case SyntaxKind::FunctionType:
        case SyntaxKind::GetAccessor:
        case SyntaxKind::Identifier:
        case SyntaxKind::IfStatement:
        case SyntaxKind::ImportDeclaration:
        case SyntaxKind::ImportEqualsDeclaration:
        case SyntaxKind::IndexSignature:
        case SyntaxKind::InterfaceDeclaration:
        case SyntaxKind::JSDocFunctionType:
        case SyntaxKind::JSDocSignature:
        case SyntaxKind::LabeledStatement:
        case SyntaxKind::MethodDeclaration:
        case SyntaxKind::MethodSignature:
        case SyntaxKind::ModuleDeclaration:
        case SyntaxKind::NamedTupleMember:
        case SyntaxKind::NamespaceExportDeclaration:
        case SyntaxKind::ObjectLiteralExpression:
        case SyntaxKind::Parameter:
        case SyntaxKind::ParenthesizedExpression:
        case SyntaxKind::PropertyAccessExpression:
        case SyntaxKind::PropertyAssignment:
        case SyntaxKind::PropertyDeclaration:
        case SyntaxKind::PropertySignature:
        case SyntaxKind::ReturnStatement:
        case SyntaxKind::SemicolonClassElement:
        case SyntaxKind::SetAccessor:
        case SyntaxKind::ShorthandPropertyAssignment:
        case SyntaxKind::SpreadAssignment:
        case SyntaxKind::SwitchStatement:
        case SyntaxKind::ThrowStatement:
        case SyntaxKind::TryStatement:
        case SyntaxKind::TypeAliasDeclaration:
        case SyntaxKind::TypeParameter:
        case SyntaxKind::VariableDeclaration:
        case SyntaxKind::VariableStatement:
        case SyntaxKind::WhileStatement:
        case SyntaxKind::WithStatement:
            return true;
        default:
            return false;
    }
}

inline static auto getModifierFlagsWorker(Node node, boolean includeJSDoc, boolean alwaysIncludeJSDoc = false) -> ModifierFlags
{
    if (node >= SyntaxKind::FirstToken && node <= SyntaxKind::LastToken)
    {
        return ModifierFlags::None;
    }

    if (!(node->modifierFlagsCache & ModifierFlags::HasComputedFlags))
    {
        node->modifierFlagsCache = getSyntacticModifierFlagsNoCache(node) | ModifierFlags::HasComputedFlags;
    }

    if (includeJSDoc && !(node->modifierFlagsCache & ModifierFlags::HasComputedJSDocModifiers) &&
        (alwaysIncludeJSDoc || isInJSFile(node)) && node->parent)
    {
        node->modifierFlagsCache |= getJSDocModifierFlagsNoCache(node) | ModifierFlags::HasComputedJSDocModifiers;
    }

    return node->modifierFlagsCache & ~(ModifierFlags::HasComputedFlags | ModifierFlags::HasComputedJSDocModifiers);
}

inline static auto getSyntacticModifierFlags(Node node) -> ModifierFlags
{
    return getModifierFlagsWorker(node, /*includeJSDoc*/ false);
}

inline static auto hasSyntacticModifiers(Node node) -> boolean
{
    return getSyntacticModifierFlags(node) != ModifierFlags::None;
}

inline static auto getSelectedSyntacticModifierFlags(Node node, ModifierFlags flags) -> ModifierFlags
{
    return getSyntacticModifierFlags(node) & flags;
}

inline static auto hasSyntacticModifier(Node node, ModifierFlags flags) -> boolean
{
    return !!getSelectedSyntacticModifierFlags(node, flags);
}

inline static auto hasStaticModifier(Node node) -> boolean
{
    return hasSyntacticModifier(node, ModifierFlags::Static);
}

inline static auto hasConstModifier(Node node) -> boolean
{
    return hasSyntacticModifier(node, ModifierFlags::Const);
}

inline static auto isSuperProperty(Node node) -> boolean
{
    auto kind = (SyntaxKind)node;
    if (kind == SyntaxKind::PropertyAccessExpression)
        return node.template as<PropertyAccessExpression>()->expression == SyntaxKind::SuperKeyword;
    if (kind == SyntaxKind::ElementAccessExpression)
        return node.template as<ElementAccessExpression>()->expression == SyntaxKind::SuperKeyword;
    return false;
}

inline static auto hasInvalidEscape(TemplateLiteral _template) -> boolean
{
    return _template && !!(isNoSubstitutionTemplateLiteral(_template)
                               ? !!_template->templateFlags
                               : (!!_template->head->templateFlags ||
                                  some(_template->templateSpans, [](TemplateSpan span) { return !!span->literal->templateFlags; })));
}

inline static auto isAssignmentPattern(Node node) -> boolean
{
    auto kind = (SyntaxKind)node;
    return kind == SyntaxKind::ArrayLiteralExpression || kind == SyntaxKind::ObjectLiteralExpression;
}

inline static auto isDeclarationBindingElement(Node bindingElement) -> boolean
{
    switch ((SyntaxKind)bindingElement)
    {
    case SyntaxKind::VariableDeclaration:
    case SyntaxKind::Parameter:
    case SyntaxKind::BindingElement:
        return true;
    }

    return false;
}

inline static auto isObjectLiteralElementLike(Node node) -> boolean
{
    auto kind = (SyntaxKind)node;
    return kind == SyntaxKind::PropertyAssignment || kind == SyntaxKind::ShorthandPropertyAssignment ||
           kind == SyntaxKind::SpreadAssignment || kind == SyntaxKind::MethodDeclaration || kind == SyntaxKind::GetAccessor ||
           kind == SyntaxKind::SetAccessor;
}

inline static auto getElementsOfBindingOrAssignmentPattern(Node name) -> NodeArray<BindingElement>
{
    switch ((SyntaxKind)name)
    {
    case SyntaxKind::ObjectBindingPattern:
        // `a` in `{a}`
        // `a` in `[a]`
        return name.as<ObjectBindingPattern>()->elements;
    case SyntaxKind::ArrayBindingPattern:
        // `a` in `{a}`
        // `a` in `[a]`
        return NodeArray<BindingElement>(name.as<ArrayBindingPattern>()->elements);
    case SyntaxKind::ArrayLiteralExpression:
        // `a` in `{a}`
        // `a` in `[a]`
        return NodeArray<BindingElement>(name.as<ArrayLiteralExpression>()->elements);

    case SyntaxKind::ObjectLiteralExpression:
        // `a` in `{a}`
        return NodeArray<BindingElement>(name.as<ObjectLiteralExpression>()->properties);
    }

    return undefined;
}

inline static auto isAssignmentExpression(Node node, boolean excludeCompoundAssignment = false) -> boolean
{
    return isBinaryExpression(node) &&
           (excludeCompoundAssignment ? node.template as<BinaryExpression>()->operatorToken == SyntaxKind::EqualsToken
                                      : isAssignmentOperator(node.template as<BinaryExpression>()->operatorToken)) &&
           isLeftHandSideExpression(node.template as<BinaryExpression>()->left);
}

inline static auto isLogicalOrCoalescingAssignmentOperator(SyntaxKind token) -> boolean
{
    return token == SyntaxKind::BarBarEqualsToken || token == SyntaxKind::AmpersandAmpersandEqualsToken ||
           token == SyntaxKind::QuestionQuestionEqualsToken;
}

inline static auto getTargetOfBindingOrAssignmentElement(Node bindingElement) -> Node
{
    if (isDeclarationBindingElement(bindingElement))
    {
        // `a` in `let { a } = ...`
        // `a` in `let { a = 1 } = ...`
        // `b` in `let { a: b } = ...`
        // `b` in `let { a: b = 1 } = ...`
        // `a` in `let { ...a } = ...`
        // `{b}` in `let { a: {b} } = ...`
        // `{b}` in `let { a: {b} = 1 } = ...`
        // `[b]` in `let { a: [b] } = ...`
        // `[b]` in `let { a: [b] = 1 } = ...`
        // `a` in `let [a] = ...`
        // `a` in `let [a = 1] = ...`
        // `a` in `let [...a] = ...`
        // `{a}` in `let [{a}] = ...`
        // `{a}` in `let [{a} = 1] = ...`
        // `[a]` in `let [[a]] = ...`
        // `[a]` in `let [[a] = 1] = ...`
        return bindingElement.as<NamedDeclaration>()->name;
    }

    if (isObjectLiteralElementLike(bindingElement))
    {
        switch ((SyntaxKind)bindingElement)
        {
        case SyntaxKind::PropertyAssignment:
            // `b` in `({ a: b } = ...)`
            // `b` in `({ a: b = 1 } = ...)`
            // `{b}` in `({ a: {b} } = ...)`
            // `{b}` in `({ a: {b} = 1 } = ...)`
            // `[b]` in `({ a: [b] } = ...)`
            // `[b]` in `({ a: [b] = 1 } = ...)`
            // `b.c` in `({ a: b.c } = ...)`
            // `b.c` in `({ a: b.c = 1 } = ...)`
            // `b[0]` in `({ a: b[0] } = ...)`
            // `b[0]` in `({ a: b[0] = 1 } = ...)`
            return getTargetOfBindingOrAssignmentElement(bindingElement.as<PropertyAssignment>()->initializer);

        case SyntaxKind::ShorthandPropertyAssignment:
            // `a` in `({ a } = ...)`
            // `a` in `({ a = 1 } = ...)`
            return bindingElement.as<ShorthandPropertyAssignment>()->name;

        case SyntaxKind::SpreadAssignment:
            // `a` in `({ ...a } = ...)`
            return getTargetOfBindingOrAssignmentElement(bindingElement.as<SpreadAssignment>()->expression);
        }

        // no target
        return undefined;
    }

    if (isAssignmentExpression(bindingElement, /*excludeCompoundAssignment*/ true))
    {
        // `a` in `[a = 1] = ...`
        // `{a}` in `[{a} = 1] = ...`
        // `[a]` in `[[a] = 1] = ...`
        // `a.b` in `[a.b = 1] = ...`
        // `a[0]` in `[a[0] = 1] = ...`
        return getTargetOfBindingOrAssignmentElement(bindingElement.as<BinaryExpression>()->left);
    }

    if (isSpreadElement(bindingElement))
    {
        // `a` in `[...a] = ...`
        return getTargetOfBindingOrAssignmentElement(bindingElement.as<SpreadElement>()->expression);
    }

    // `a` in `[a] = ...`
    // `{a}` in `[{a}] = ...`
    // `[a]` in `[[a]] = ...`
    // `a.b` in `[a.b] = ...`
    // `a[0]` in `[a[0]] = ...`
    return bindingElement;
}

inline static auto getOperator(Expression expression) -> SyntaxKind
{
    if (expression == SyntaxKind::BinaryExpression)
    {
        return expression.as<BinaryExpression>()->operatorToken;
    }
    else if (expression == SyntaxKind::PrefixUnaryExpression)
    {
        return expression.as<PrefixUnaryExpression>()->_operator;
    }
    else if (expression == SyntaxKind::PostfixUnaryExpression)
    {
        return expression.as<PostfixUnaryExpression>()->_operator;
    }
    else
    {
        return expression;
    }

    return SyntaxKind::Unknown;
}

inline static auto getOperatorPrecedence(SyntaxKind nodeKind, SyntaxKind operatorKind, boolean hasArguments = false)
{
    switch (nodeKind)
    {
    case SyntaxKind::CommaListExpression:
        return OperatorPrecedence::Comma;

    case SyntaxKind::SpreadElement:
        return OperatorPrecedence::Spread;

    case SyntaxKind::YieldExpression:
        return OperatorPrecedence::Yield;

    case SyntaxKind::ConditionalExpression:
        return OperatorPrecedence::Conditional;

    case SyntaxKind::BinaryExpression:
        switch (operatorKind)
        {
        case SyntaxKind::CommaToken:
            return OperatorPrecedence::Comma;

        case SyntaxKind::EqualsToken:
        case SyntaxKind::PlusEqualsToken:
        case SyntaxKind::MinusEqualsToken:
        case SyntaxKind::AsteriskAsteriskEqualsToken:
        case SyntaxKind::AsteriskEqualsToken:
        case SyntaxKind::SlashEqualsToken:
        case SyntaxKind::PercentEqualsToken:
        case SyntaxKind::LessThanLessThanEqualsToken:
        case SyntaxKind::GreaterThanGreaterThanEqualsToken:
        case SyntaxKind::GreaterThanGreaterThanGreaterThanEqualsToken:
        case SyntaxKind::AmpersandEqualsToken:
        case SyntaxKind::CaretEqualsToken:
        case SyntaxKind::BarEqualsToken:
        case SyntaxKind::BarBarEqualsToken:
        case SyntaxKind::AmpersandAmpersandEqualsToken:
        case SyntaxKind::QuestionQuestionEqualsToken:
            return OperatorPrecedence::Assignment;

        default:
            return getBinaryOperatorPrecedence(operatorKind);
        }

    // TODO: Should prefix `++` and `--` be moved to the `Update` precedence?
    case SyntaxKind::TypeAssertionExpression:
    case SyntaxKind::NonNullExpression:
    case SyntaxKind::PrefixUnaryExpression:
    case SyntaxKind::TypeOfExpression:
    case SyntaxKind::VoidExpression:
    case SyntaxKind::DeleteExpression:
    case SyntaxKind::AwaitExpression:
        return OperatorPrecedence::Unary;

    case SyntaxKind::PostfixUnaryExpression:
        return OperatorPrecedence::Update;

    case SyntaxKind::CallExpression:
        return OperatorPrecedence::LeftHandSide;

    case SyntaxKind::NewExpression:
        return hasArguments ? OperatorPrecedence::Member : OperatorPrecedence::LeftHandSide;

    case SyntaxKind::TaggedTemplateExpression:
    case SyntaxKind::PropertyAccessExpression:
    case SyntaxKind::ElementAccessExpression:
    case SyntaxKind::MetaProperty:
        return OperatorPrecedence::Member;

    case SyntaxKind::AsExpression:
        return OperatorPrecedence::Relational;

    case SyntaxKind::ThisKeyword:
    case SyntaxKind::SuperKeyword:
    case SyntaxKind::Identifier:
    case SyntaxKind::NullKeyword:
    case SyntaxKind::TrueKeyword:
    case SyntaxKind::FalseKeyword:
    case SyntaxKind::NumericLiteral:
    case SyntaxKind::BigIntLiteral:
    case SyntaxKind::StringLiteral:
    case SyntaxKind::ArrayLiteralExpression:
    case SyntaxKind::ObjectLiteralExpression:
    case SyntaxKind::FunctionExpression:
    case SyntaxKind::ArrowFunction:
    case SyntaxKind::ClassExpression:
    case SyntaxKind::RegularExpressionLiteral:
    case SyntaxKind::NoSubstitutionTemplateLiteral:
    case SyntaxKind::TemplateExpression:
    case SyntaxKind::ParenthesizedExpression:
    case SyntaxKind::OmittedExpression:
    case SyntaxKind::JsxElement:
    case SyntaxKind::JsxSelfClosingElement:
    case SyntaxKind::JsxFragment:
        return OperatorPrecedence::Primary;

    default:
        return OperatorPrecedence::Invalid;
    }

    return OperatorPrecedence::Invalid;
}

inline static auto getExpressionPrecedence(Expression expression)
{
    auto _operator = getOperator(expression);
    auto hasArguments = expression == SyntaxKind::NewExpression && !!expression.as<NewExpression>()->arguments;
    return getOperatorPrecedence(expression, _operator, hasArguments);
}

inline static auto getLeftmostExpression(Expression node, boolean stopAtCallExpressions) -> Node
{
    while (true)
    {
        switch ((SyntaxKind)node)
        {
        case SyntaxKind::PostfixUnaryExpression:
            node = node.template as<PostfixUnaryExpression>()->operand;
            continue;

        case SyntaxKind::BinaryExpression:
            node = node.template as<BinaryExpression>()->left;
            continue;

        case SyntaxKind::ConditionalExpression:
            node = node.template as<ConditionalExpression>()->condition;
            continue;

        case SyntaxKind::TaggedTemplateExpression:
            node = node.template as<TaggedTemplateExpression>()->tag;
            continue;

        case SyntaxKind::CallExpression:
            if (stopAtCallExpressions)
            {
                return node;
            }
            node = node.template as<CallExpression>()->expression;
            continue;
            // falls through
        case SyntaxKind::AsExpression:
            node = node.template as<AsExpression>()->expression;
            continue;
        case SyntaxKind::ElementAccessExpression:
            node = node.template as<ElementAccessExpression>()->expression;
            continue;
        case SyntaxKind::PropertyAccessExpression:
            node = node.template as<PropertyAccessExpression>()->expression;
            continue;
        case SyntaxKind::NonNullExpression:
            node = node.template as<NonNullExpression>()->expression;
            continue;
        case SyntaxKind::PartiallyEmittedExpression:
            node = node.template as<PartiallyEmittedExpression>()->expression;
            continue;
        }

        return node;
    }
}

inline static auto isUnaryExpressionKind(SyntaxKind kind) -> boolean
{
    switch (kind)
    {
    case SyntaxKind::PrefixUnaryExpression:
    case SyntaxKind::PostfixUnaryExpression:
    case SyntaxKind::DeleteExpression:
    case SyntaxKind::TypeOfExpression:
    case SyntaxKind::VoidExpression:
    case SyntaxKind::AwaitExpression:
    case SyntaxKind::TypeAssertionExpression:
        return true;
    default:
        return isLeftHandSideExpressionKind(kind);
    }
}

inline static auto isUnaryExpression(Node node) -> boolean
{
    return isUnaryExpressionKind(skipPartiallyEmittedExpressions(node));
}

inline static auto isOptionalChain(Node node) -> boolean {
    auto kind = node->_kind;
    return !!(node->flags & NodeFlags::OptionalChain) &&
        (kind == SyntaxKind::PropertyAccessExpression
            || kind == SyntaxKind::ElementAccessExpression
            || kind == SyntaxKind::CallExpression
            || kind == SyntaxKind::NonNullExpression);
}

inline static auto getOperatorAssociativity(SyntaxKind kind, SyntaxKind _operator, boolean hasArguments = false) -> Associativity
{
    switch (kind)
    {
    case SyntaxKind::NewExpression:
        return hasArguments ? Associativity::Left : Associativity::Right;

    case SyntaxKind::PrefixUnaryExpression:
    case SyntaxKind::TypeOfExpression:
    case SyntaxKind::VoidExpression:
    case SyntaxKind::DeleteExpression:
    case SyntaxKind::AwaitExpression:
    case SyntaxKind::ConditionalExpression:
    case SyntaxKind::YieldExpression:
        return Associativity::Right;

    case SyntaxKind::BinaryExpression:
        switch (_operator)
        {
        case SyntaxKind::AsteriskAsteriskToken:
        case SyntaxKind::EqualsToken:
        case SyntaxKind::PlusEqualsToken:
        case SyntaxKind::MinusEqualsToken:
        case SyntaxKind::AsteriskAsteriskEqualsToken:
        case SyntaxKind::AsteriskEqualsToken:
        case SyntaxKind::SlashEqualsToken:
        case SyntaxKind::PercentEqualsToken:
        case SyntaxKind::LessThanLessThanEqualsToken:
        case SyntaxKind::GreaterThanGreaterThanEqualsToken:
        case SyntaxKind::GreaterThanGreaterThanGreaterThanEqualsToken:
        case SyntaxKind::AmpersandEqualsToken:
        case SyntaxKind::CaretEqualsToken:
        case SyntaxKind::BarEqualsToken:
        case SyntaxKind::BarBarEqualsToken:
        case SyntaxKind::AmpersandAmpersandEqualsToken:
        case SyntaxKind::QuestionQuestionEqualsToken:
            return Associativity::Right;
        }
    }
    return Associativity::Left;
}

inline static auto canHaveModifiers(SyntaxKind kind) -> boolean {
    return kind == SyntaxKind::TypeParameter
        || kind == SyntaxKind::Parameter
        || kind == SyntaxKind::PropertySignature
        || kind == SyntaxKind::PropertyDeclaration
        || kind == SyntaxKind::MethodSignature
        || kind == SyntaxKind::MethodDeclaration
        || kind == SyntaxKind::Constructor
        || kind == SyntaxKind::GetAccessor
        || kind == SyntaxKind::SetAccessor
        || kind == SyntaxKind::IndexSignature
        || kind == SyntaxKind::ConstructorType
        || kind == SyntaxKind::FunctionExpression
        || kind == SyntaxKind::ArrowFunction
        || kind == SyntaxKind::ClassExpression
        || kind == SyntaxKind::VariableStatement
        || kind == SyntaxKind::FunctionDeclaration
        || kind == SyntaxKind::ClassDeclaration
        || kind == SyntaxKind::InterfaceDeclaration
        || kind == SyntaxKind::TypeAliasDeclaration
        || kind == SyntaxKind::EnumDeclaration
        || kind == SyntaxKind::ModuleDeclaration
        || kind == SyntaxKind::ImportEqualsDeclaration
        || kind == SyntaxKind::ImportDeclaration
        || kind == SyntaxKind::ExportAssignment
        || kind == SyntaxKind::ExportDeclaration;
}

inline static auto getExpressionAssociativity(Expression expression)
{
    auto _operator = getOperator(expression);
    auto hasArguments = expression == SyntaxKind::NewExpression && !!expression.as<NewExpression>()->arguments;
    return getOperatorAssociativity(expression, _operator, hasArguments);
}

inline static auto isFunctionOrConstructorTypeNode(Node node) -> boolean
{
    switch ((SyntaxKind)node)
    {
    case SyntaxKind::FunctionType:
    case SyntaxKind::ConstructorType:
        return true;
    }

    return false;
}

// TODO: for emitNode
static auto isGeneratedIdentifier(Node node) -> boolean {
    //return isIdentifier(node) && node->emitNode->autoGenerate != undefined;
    return false;
}

//static auto getEmitFlags(Node node) -> EmitFlags {
//    auto emitNode = node->emitNode;
//    return emitNode && emitNode->flags || 0;
//}

static auto isLocalName(Identifier node) -> boolean {
    //return (getEmitFlags(node) & EmitFlags::LocalName) != EmitFlags::None;
    return false;
}

static auto getJSDocTypeAliasName(JSDocNamespaceBody fullName) -> Identifier
{
    // TODO: finish it: !rightNode.as<ModuleDeclaration>()->body
    if (fullName)
    {
        auto rightNode = fullName;
        while (true)
        {
            if (isIdentifier(rightNode) || !rightNode.as<ModuleDeclaration>()->body)
            {
                return isIdentifier(rightNode) ? rightNode : rightNode.as<ModuleDeclaration>()->name;
            }
            rightNode = rightNode.as<ModuleDeclaration>()->body;
        }
    }

    return undefined;
}

inline static auto regex_exec(string &text, regex regEx) -> boolean
{
    auto words_begin = sregex_iterator(text.begin(), text.end(), regEx);
    auto words_end = sregex_iterator();
    return words_begin != words_end;
}

inline static auto hasModifier(Node node, SyntaxKind key) -> boolean
{
    return some(node->modifiers, [=](auto m) { return m == key; });
}

/**
 * Remove extra underscore from escaped identifier text content.
 *
 * @param identifier The escaped identifier text.
 * @returns The unescaped identifier text.
 */
inline static auto unescapeLeadingUnderscores(string identifier) -> string {
    auto id = identifier;
    return id.length() >= 3 && id[0] == (char_t) CharacterCodes::_ && id[1] == (char_t) CharacterCodes::_ && id[2] == (char_t) CharacterCodes::_ ? id.substr(1) : id;
}

inline static auto idText(Node identifierOrPrivateName) -> string {
    if (identifierOrPrivateName == SyntaxKind::Identifier)
        return unescapeLeadingUnderscores(identifierOrPrivateName.as<Identifier>() ->escapedText);
    if (identifierOrPrivateName == SyntaxKind::PrivateIdentifier)
        return unescapeLeadingUnderscores(identifierOrPrivateName.as<PrivateIdentifier>()->escapedText);

    return S("");
}

} // namespace ts

#endif // UTILITIES_H