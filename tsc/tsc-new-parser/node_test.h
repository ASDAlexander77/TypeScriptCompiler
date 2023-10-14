#ifndef NODETEST_H
#define NODETEST_H

#include "enums.h"
#include "scanner.h"

// Literals

namespace ts
{
    inline auto isNumericLiteral(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NumericLiteral;
    }

    inline auto isBigIntLiteral(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::BigIntLiteral;
    }

    inline auto isStringLiteral(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::StringLiteral;
    }

    inline auto isJsxText(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxText;
    }

    inline auto isRegularExpressionLiteral(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::RegularExpressionLiteral;
    }

    inline auto isNoSubstitutionTemplateLiteral(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NoSubstitutionTemplateLiteral;
    }

    // Pseudo-literals

    inline auto isTemplateHead(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TemplateHead;
    }

    inline auto isTemplateMiddle(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TemplateMiddle;
    }

    inline auto isTemplateTail(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TemplateTail;
    }

    // Punctuation

    inline auto isDotDotDotToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::DotDotDotToken;
    }

    /*@internal*/
    inline auto isCommaToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::CommaToken;
    }

    inline auto isPlusToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PlusToken;
    }

    inline auto isMinusToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::MinusToken;
    }

    inline auto isAsteriskToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::AsteriskToken;
    }

    /*@internal*/
    inline auto isExclamationToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ExclamationToken;
    }

    /*@internal*/
    inline auto isQuestionToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::QuestionToken;
    }

    /*@internal*/
    inline auto isColonToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ColonToken;
    }

    /*@internal*/
    inline auto isQuestionDotToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::QuestionDotToken;
    }

    /*@internal*/
    inline auto isEqualsGreaterThanToken(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::EqualsGreaterThanToken;
    }

    // Identifiers

    inline auto isIdentifier(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::Identifier;
    }

    inline auto isPrivateIdentifier(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PrivateIdentifier;
    }

    // Reserved Words

    /* @internal */
    inline auto isExportModifier(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ExportKeyword;
    }

    /* @internal */
    inline auto isAsyncModifier(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::AsyncKeyword;
    }

    /* @internal */
    inline auto isAssertsKeyword(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::AssertsKeyword;
    }

    /* @internal */
    inline auto isAwaitKeyword(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::AwaitKeyword;
    }

    /* @internal */
    inline auto isReadonlyKeyword(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ReadonlyKeyword;
    }

    /* @internal */
    inline auto isStaticModifier(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::StaticKeyword;
    }

    /*@internal*/
    inline auto isSuperKeyword(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SuperKeyword;
    }

    /*@internal*/
    inline auto isImportKeyword(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ImportKeyword;
    }

    // Names

    inline auto isQualifiedName(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::QualifiedName;
    }

    inline auto isComputedPropertyName(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ComputedPropertyName;
    }

    // Signature elements

    inline auto isTypeParameterDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypeParameter;
    }

    // TODO(rbuckton): Rename to 'isParameterDeclaration'
    inline auto isParameter(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::Parameter;
    }

    inline auto isDecorator(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::Decorator;
    }

    // TypeMember

    inline auto isPropertySignature(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PropertySignature;
    }

    inline auto isPropertyDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PropertyDeclaration;
    }

    inline auto isMethodSignature(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::MethodSignature;
    }

    inline auto isMethodDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::MethodDeclaration;
    }

    inline auto isConstructorDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::Constructor;
    }

    inline auto isGetAccessorDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::GetAccessor;
    }

    inline auto isSetAccessorDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SetAccessor;
    }

    inline auto isCallSignatureDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::CallSignature;
    }

    inline auto isConstructSignatureDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ConstructSignature;
    }

    inline auto isIndexSignatureDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::IndexSignature;
    }

    // Type

    inline auto isTypePredicateNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypePredicate;
    }

    inline auto isTypeReferenceNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypeReference;
    }

    inline auto isFunctionTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::FunctionType;
    }

    inline auto isConstructorTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ConstructorType;
    }

    inline auto isTypeQueryNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypeQuery;
    }

    inline auto isTypeLiteralNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypeLiteral;
    }

    inline auto isArrayTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ArrayType;
    }

    inline auto isTupleTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TupleType;
    }

    inline auto isNamedTupleMember(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NamedTupleMember;
    }

    inline auto isOptionalTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::OptionalType;
    }

    inline auto isRestTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::RestType;
    }

    inline auto isUnionTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::UnionType;
    }

    inline auto isIntersectionTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::IntersectionType;
    }

    inline auto isConditionalTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ConditionalType;
    }

    inline auto isInferTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::InferType;
    }

    inline auto isParenthesizedTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ParenthesizedType;
    }

    inline auto isThisTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ThisType;
    }

    inline auto isTypeOperatorNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypeOperator;
    }

    inline auto isIndexedAccessTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::IndexedAccessType;
    }

    inline auto isMappedTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::MappedType;
    }

    inline auto isLiteralTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::LiteralType;
    }

    inline auto isImportTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ImportType;
    }

    inline auto isTemplateLiteralTypeSpan(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TemplateLiteralTypeSpan;
    }

    inline auto isTemplateLiteralTypeNode(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TemplateLiteralType;
    }

    // Binding patterns

    inline auto isObjectBindingPattern(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ObjectBindingPattern;
    }

    inline auto isArrayBindingPattern(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ArrayBindingPattern;
    }

    inline auto isBindingElement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::BindingElement;
    }

    // Expression

    inline auto isArrayLiteralExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ArrayLiteralExpression;
    }

    inline auto isObjectLiteralExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ObjectLiteralExpression;
    }

    inline auto isPropertyAccessExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PropertyAccessExpression;
    }

    inline auto isElementAccessExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ElementAccessExpression;
    }

    inline auto isCallExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::CallExpression;
    }

    inline auto isNewExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NewExpression;
    }

    inline auto isTaggedTemplateExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TaggedTemplateExpression;
    }

    inline auto isTypeAssertionExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypeAssertionExpression;
    }

    inline auto isParenthesizedExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ParenthesizedExpression;
    }

    inline auto isFunctionExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::FunctionExpression;
    }

    inline auto isArrowFunction(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ArrowFunction;
    }

    inline auto isDeleteExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::DeleteExpression;
    }

    inline auto isTypeOfExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypeOfExpression;
    }

    inline auto isVoidExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::VoidExpression;
    }

    inline auto isAwaitExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::AwaitExpression;
    }

    inline auto isPrefixUnaryExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PrefixUnaryExpression;
    }

    inline auto isPostfixUnaryExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PostfixUnaryExpression;
    }

    inline auto isBinaryExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::BinaryExpression;
    }

    inline auto isConditionalExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ConditionalExpression;
    }

    inline auto isTemplateExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TemplateExpression;
    }

    inline auto isYieldExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::YieldExpression;
    }

    inline auto isSpreadElement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SpreadElement;
    }

    inline auto isClassExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ClassExpression;
    }

    inline auto isOmittedExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::OmittedExpression;
    }

    inline auto isExpressionWithTypeArguments(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ExpressionWithTypeArguments;
    }

    inline auto isAsExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::AsExpression;
    }

    inline auto isNonNullExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NonNullExpression;
    }

    inline auto isMetaProperty(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::MetaProperty;
    }

    inline auto isSyntheticExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SyntheticExpression;
    }

    inline auto isPartiallyEmittedExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PartiallyEmittedExpression;
    }

    inline auto isCommaListExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::CommaListExpression;
    }

    // Misc

    inline auto isTemplateSpan(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TemplateSpan;
    }

    inline auto isSemicolonClassElement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SemicolonClassElement;
    }

    // Elements

    inline auto isBlock(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::Block;
    }

    inline auto isVariableStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::VariableStatement;
    }

    inline auto isEmptyStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::EmptyStatement;
    }

    inline auto isExpressionStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ExpressionStatement;
    }

    inline auto isIfStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::IfStatement;
    }

    inline auto isDoStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::DoStatement;
    }

    inline auto isWhileStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::WhileStatement;
    }

    inline auto isForStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ForStatement;
    }

    inline auto isForInStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ForInStatement;
    }

    inline auto isForOfStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ForOfStatement;
    }

    inline auto isContinueStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ContinueStatement;
    }

    inline auto isBreakStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::BreakStatement;
    }

    inline auto isReturnStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ReturnStatement;
    }

    inline auto isWithStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::WithStatement;
    }

    inline auto isSwitchStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SwitchStatement;
    }

    inline auto isLabeledStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::LabeledStatement;
    }

    inline auto isThrowStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ThrowStatement;
    }

    inline auto isTryStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TryStatement;
    }

    inline auto isDebuggerStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::DebuggerStatement;
    }

    inline auto isVariableDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::VariableDeclaration;
    }

    inline auto isVariableDeclarationList(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::VariableDeclarationList;
    }

    inline auto isFunctionDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::FunctionDeclaration;
    }

    inline auto isClassDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ClassDeclaration;
    }

    inline auto isInterfaceDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::InterfaceDeclaration;
    }

    inline auto isTypeAliasDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::TypeAliasDeclaration;
    }

    inline auto isEnumDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::EnumDeclaration;
    }

    inline auto isModuleDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ModuleDeclaration;
    }

    inline auto isModuleBlock(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ModuleBlock;
    }

    inline auto isCaseBlock(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::CaseBlock;
    }

    inline auto isNamespaceExportDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NamespaceExportDeclaration;
    }

    inline auto isImportEqualsDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ImportEqualsDeclaration;
    }

    inline auto isImportDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ImportDeclaration;
    }

    inline auto isImportClause(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ImportClause;
    }

    inline auto isNamespaceImport(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NamespaceImport;
    }

    inline auto isNamespaceExport(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NamespaceExport;
    }

    inline auto isNamedImports(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NamedImports;
    }

    inline auto isImportSpecifier(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ImportSpecifier;
    }

    inline auto isExportAssignment(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ExportAssignment;
    }

    inline auto isExportDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ExportDeclaration;
    }

    inline auto isNamedExports(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NamedExports;
    }

    inline auto isExportSpecifier(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ExportSpecifier;
    }

    inline auto isMissingDeclaration(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::MissingDeclaration;
    }

    inline auto isNotEmittedStatement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::NotEmittedStatement;
    }

    /* @internal */
    inline auto isSyntheticReference(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SyntheticReferenceExpression;
    }

    // Module References

    inline auto isExternalModuleReference(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ExternalModuleReference;
    }

    // JSX

    inline auto isJsxElement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxElement;
    }

    inline auto isJsxSelfClosingElement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxSelfClosingElement;
    }

    inline auto isJsxOpeningElement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxOpeningElement;
    }

    inline auto isJsxClosingElement(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxClosingElement;
    }

    inline auto isJsxFragment(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxFragment;
    }

    inline auto isJsxOpeningFragment(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxOpeningFragment;
    }

    inline auto isJsxClosingFragment(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxClosingFragment;
    }

    inline auto isJsxAttribute(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxAttribute;
    }

    inline auto isJsxAttributes(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxAttributes;
    }

    inline auto isJsxSpreadAttribute(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxSpreadAttribute;
    }

    inline auto isJsxExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JsxExpression;
    }

    // Clauses

    inline auto isCaseClause(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::CaseClause;
    }

    inline auto isDefaultClause(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::DefaultClause;
    }

    inline auto isHeritageClause(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::HeritageClause;
    }

    inline auto isCatchClause(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::CatchClause;
    }

    // Property assignments

    inline auto isPropertyAssignment(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::PropertyAssignment;
    }

    inline auto isShorthandPropertyAssignment(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::ShorthandPropertyAssignment;
    }

    inline auto isSpreadAssignment(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SpreadAssignment;
    }

    // Enum

    inline auto isEnumMember(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::EnumMember;
    }

    // Unparsed

    // TODO(rbuckton): isUnparsedPrologue

    inline auto isUnparsedPrepend(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::UnparsedPrepend;
    }

    // TODO(rbuckton): isUnparsedText
    // TODO(rbuckton): isUnparsedInternalText
    // TODO(rbuckton): isUnparsedSyntheticReference

    // Top-level nodes
    inline auto isSourceFile(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SourceFile;
    }

    inline auto isBundle(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::Bundle;
    }

    inline auto isUnparsedSource(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::UnparsedSource;
    }

    // TODO(rbuckton): isInputFiles

    // JSDoc Elements

    inline auto isJSDocTypeExpression(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocTypeExpression;
    }

    inline auto isJSDocNameReference(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocNameReference;
    }

    inline auto isJSDocAllType(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocAllType;
    }

    inline auto isJSDocUnknownType(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocUnknownType;
    }

    inline auto isJSDocNullableType(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocNullableType;
    }

    inline auto isJSDocNonNullableType(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocNonNullableType;
    }

    inline auto isJSDocOptionalType(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocOptionalType;
    }

    inline auto isJSDocFunctionType(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocFunctionType;
    }

    inline auto isJSDocVariadicType(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocVariadicType;
    }

    inline auto isJSDocNamepathType(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocNamepathType;
    }

    inline auto isJSDoc(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocComment;
    }

    inline auto isJSDocTypeLiteral(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocTypeLiteral;
    }

    inline auto isJSDocSignature(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocSignature;
    }

    // JSDoc Tags

    inline auto isJSDocAugmentsTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocAugmentsTag;
    }

    inline auto isJSDocAuthorTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocAuthorTag;
    }

    inline auto isJSDocClassTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocClassTag;
    }

    inline auto isJSDocCallbackTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocCallbackTag;
    }

    inline auto isJSDocPublicTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocPublicTag;
    }

    inline auto isJSDocPrivateTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocPrivateTag;
    }

    inline auto isJSDocProtectedTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocProtectedTag;
    }

    inline auto isJSDocReadonlyTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocReadonlyTag;
    }

    inline auto isJSDocDeprecatedTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocDeprecatedTag;
    }

    inline auto isJSDocSeeTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocSeeTag;
    }

    inline auto isJSDocEnumTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocEnumTag;
    }

    inline auto isJSDocParameterTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocParameterTag;
    }

    inline auto isJSDocReturnTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocReturnTag;
    }

    inline auto isJSDocThisTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocThisTag;
    }

    inline auto isJSDocTypeTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocTypeTag;
    }

    inline auto isJSDocTemplateTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocTemplateTag;
    }

    inline auto isJSDocTypedefTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocTypedefTag;
    }

    inline auto isJSDocUnknownTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocTag;
    }

    inline auto isJSDocPropertyTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocPropertyTag;
    }

    inline auto isJSDocImplementsTag(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::JSDocImplementsTag;
    }

    // Synthesized list

    /* @internal */
    inline auto isSyntaxList(SyntaxKind kind) -> boolean
    {
        return kind == SyntaxKind::SyntaxList;
    }
}

#endif // NODETEST_H