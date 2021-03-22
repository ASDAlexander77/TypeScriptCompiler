#ifndef NODEFACTORY_H
#define NODEFACTORY_H

#include "enums.h"
#include "scanner.h"

typedef std::function<Node(SyntaxKind)> NodeCreate;

enum class NodeFactoryFlags : number {
    None = 0,
    // Disables the parenthesizer rules for the factory.
    NoParenthesizerRules = 1 << 0,
    // Disables the node converters for the factory.
    NoNodeConverters = 1 << 1,
    // Ensures new `PropertyAccessExpression` nodes are created with the `NoIndentation` emit flag set.
    NoIndentationOnFreshPropertyAccess = 1 << 2,
    // Do not set an `original` pointer when updating a node.
    NoOriginalNode = 1 << 3,
};

static NodeFactoryFlags operator |(NodeFactoryFlags lhs, NodeFactoryFlags rhs)
{
    return (NodeFactoryFlags) ((number) lhs | (number) rhs);
}

struct BaseNodeFactory
{
    BaseNodeFactory() = default;

    NodeCreate createBaseSourceFileNode;
    NodeCreate createBaseIdentifierNode;
    NodeCreate createBasePrivateIdentifierNode;
    NodeCreate createBaseTokenNode;
    NodeCreate createBaseNode;
};

class NodeFactory
{
    NodeFactoryFlags nodeFactoryFlags;
    BaseNodeFactory baseNodeFactory;
public:
    NodeFactory(NodeFactoryFlags nodeFactoryFlags, BaseNodeFactory baseNodeFactory) : nodeFactoryFlags(nodeFactoryFlags), baseNodeFactory(baseNodeFactory) {}

    ArrayLiteralExpression createArrayLiteralExpression(Node, boolean multiLine = false);

    ExpressionStatement createExpressionStatement(Expression expression);

    SourceFile createSourceFile(Node statements, EndOfFileToken endOfFileToken, NodeFlags flags);

    SourceFile updateSourceFile(SourceFile node, Node statements, boolean isDeclarationFile = false, std::vector<FileReference> referencedFiles = std::vector<FileReference>(), std::vector<FileReference> typeReferences = std::vector<FileReference>(), boolean hasNoDefaultLib = false, std::vector<FileReference> libReferences = std::vector<FileReference>());

    Node createNodeArray(Node, boolean hasTrailingComma = false);

    Node createToken(SyntaxKind kind);

    SuperExpression createSuper();

    ThisExpression createThis();

    NullLiteral createNull();

    TrueLiteral createTrue();

    FalseLiteral createFalse();

    Identifier createIdentifier(string text, Node typeArguments = undefined, SyntaxKind originalKeywordKind = SyntaxKind::Unknown);

    TemplateLiteralLikeNode createTemplateLiteralLikeNode(SyntaxKind kind = SyntaxKind::Unknown, string text = string(), string rawText = string(), TokenFlags templateFlags = TokenFlags::None);

    NumericLiteral createNumericLiteral(string value = string(), TokenFlags numericLiteralFlags = TokenFlags::NumericLiteralFlags);

    NumericLiteral createNumericLiteral(number value, TokenFlags numericLiteralFlags = TokenFlags::NumericLiteralFlags);

    BigIntLiteral createBigIntLiteral(string value);

    BigIntLiteral createBigIntLiteral(PseudoBigInt value);

    StringLiteral createStringLiteral(string text, boolean isSingleQuote = false, boolean hasExtendedUnicodeEscape = false); // eslint-disable-line @typescript-eslint/unified-signatures

    StringLiteral createStringLiteralFromNode(PropertyNameLiteral sourceNode, boolean isSingleQuote = false);

    RegularExpressionLiteral createRegularExpressionLiteral(string text);

    MissingDeclaration createMissingDeclaration();

    ComputedPropertyName createComputedPropertyName(Expression expression);

    PrivateIdentifier createPrivateIdentifier(string name);

    QualifiedName createQualifiedName(Node left, string right);

    QualifiedName createQualifiedName(Node left, Identifier right);

    TemplateExpression createTemplateExpression(TemplateHead head, Node /*NodeArray<TemplateSpan>*/ templateSpans);

    TemplateLiteralTypeSpan createTemplateLiteralType(Node type, Node literal);

    TemplateLiteralTypeSpan createTemplateLiteralTypeSpan(Node type, Node literal);

    TemplateSpan createTemplateSpan(Expression expression, Node literal);

    LiteralToken createLiteralLikeNode(SyntaxKind kind, string text);

    TypeReferenceNode createTypeReferenceNode(string typeName, Node typeArguments);

    TypeReferenceNode createTypeReferenceNode(/*EntityName*/Node typeName, Node typeArguments);

    TypePredicateNode createTypePredicateNode(SyntaxKind assertsModifier, string parameterName, TypeNode type);

    TypePredicateNode createTypePredicateNode(SyntaxKind assertsModifier, Node parameterName, TypeNode type);

    ThisTypeNode createThisTypeNode();

    //
    // JSDoc
    //

    auto createJSDocAllType() -> JSDocAllType;
    auto createJSDocUnknownType() -> JSDocUnknownType;
    auto createJSDocNonNullableType(TypeNode type) -> JSDocNonNullableType;
    auto createJSDocNullableType(TypeNode type) -> JSDocNullableType;
    auto createJSDocOptionalType(TypeNode type) -> JSDocOptionalType;
    auto createJSDocFunctionType(Node/*ParameterDeclaration*/ parameters, TypeNode type) -> JSDocFunctionType;
    auto createJSDocVariadicType(TypeNode type) -> JSDocVariadicType;
    auto createJSDocNamepathType(TypeNode type) -> JSDocNamepathType;
    auto createJSDocTypeExpression(TypeNode type) -> JSDocTypeExpression;
    auto createJSDocNameReference(EntityName name) -> JSDocNameReference;
    auto createJSDocTypeLiteral(JSDocPropertyLikeTag jsDocPropertyTags = JSDocPropertyLikeTag(), boolean isArrayType = false) -> JSDocTypeLiteral;
    auto createJSDocSignature(JSDocTemplateTag typeParameters[], /*JSDocParameterTag[]*/ Node parameters, JSDocReturnTag type = JSDocReturnTag()) -> JSDocSignature;
    auto createJSDocTemplateTag(Identifier tagName, JSDocTypeExpression constraint, /*TypeParameterDeclaration[]*/Node typeParameters, string comment = string()) -> JSDocTemplateTag;
    auto createJSDocTypedefTag(Identifier tagName, Node typeExpression, Node fullName, string comment = string()) -> JSDocTypedefTag;
    auto createJSDocParameterTag(Identifier tagName, EntityName name, boolean isBracketed, JSDocTypeExpression typeExpression = JSDocTypeExpression(), boolean isNameFirst = false, string comment = string()) -> JSDocParameterTag;
    auto createJSDocPropertyTag(Identifier tagName, EntityName name, boolean isBracketed, JSDocTypeExpression typeExpression = JSDocTypeExpression(), boolean isNameFirst = false, string comment = string()) -> JSDocPropertyTag;
    auto createJSDocTypeTag(Identifier tagName, JSDocTypeExpression typeExpression, string comment = string()) -> JSDocTypeTag;
    auto createJSDocSeeTag(Identifier tagName, JSDocNameReference nameExpression, string comment = string()) -> JSDocSeeTag;
    auto createJSDocReturnTag(Identifier tagName, JSDocTypeExpression typeExpression = JSDocTypeExpression(), string comment = string()) -> JSDocReturnTag;
    auto createJSDocThisTag(Identifier tagName, JSDocTypeExpression typeExpression, string comment = string()) -> JSDocThisTag;
    auto createJSDocEnumTag(Identifier tagName, JSDocTypeExpression typeExpression, string comment = string()) -> JSDocEnumTag;
    auto createJSDocCallbackTag(Identifier tagName, JSDocSignature typeExpression, Node fullName = Node(), string comment = string()) -> JSDocCallbackTag;
    auto createJSDocAugmentsTag(Identifier tagName, JSDocAugmentsTag className, string comment = string()) -> JSDocAugmentsTag;
    auto createJSDocImplementsTag(Identifier tagName, JSDocImplementsTag className, string comment = string()) -> JSDocImplementsTag;
    auto createJSDocAuthorTag(Identifier tagName, string comment = string()) -> JSDocAuthorTag;
    auto createJSDocClassTag(Identifier tagName, string comment = string()) -> JSDocClassTag;
    auto createJSDocPublicTag(Identifier tagName, string comment = string()) -> JSDocPublicTag;
    auto createJSDocPrivateTag(Identifier tagName, string comment = string()) -> JSDocPrivateTag;
    auto createJSDocProtectedTag(Identifier tagName, string comment = string()) -> JSDocProtectedTag;
    auto createJSDocReadonlyTag(Identifier tagName, string comment = string()) -> JSDocReadonlyTag;
    auto createJSDocUnknownTag(Identifier tagName, string comment = string()) -> JSDocUnknownTag;
    auto createJSDocDeprecatedTag(Identifier tagName, string comment = string()) -> JSDocDeprecatedTag;
    auto createJSDocComment(string comment = string(), /*JSDocTag[]*/ Node tags = Node()) -> JSDoc;
};

#endif // NODEFACTORY_H