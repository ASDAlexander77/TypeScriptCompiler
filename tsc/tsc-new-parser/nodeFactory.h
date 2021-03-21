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

    Node createArrayLiteralExpression(Node);

    Node createExpressionStatement(Node);

    SourceFile updateSourceFile(SourceFile, Node);

    Node createNodeArray(Node, boolean hasTrailingComma = false);

    SourceFile createSourceFile(Node statements, Node endOfFileToken, NodeFlags flags);

    Node createToken(SyntaxKind kind);

    Node createIdentifier(string text, Node typeArguments = undefined, SyntaxKind originalKeywordKind = SyntaxKind::Unknown);

    Node createTemplateLiteralLikeNode(SyntaxKind kind = SyntaxKind::Unknown, string = string(), string = string(), TokenFlags templateFlags = TokenFlags::None);

    Node createNumericLiteral(string value = string(), TokenFlags numericLiteralFlags = TokenFlags::NumericLiteralFlags);

    Node createStringLiteral(string text, boolean isSingleQuote = false, boolean hasExtendedUnicodeEscape = false);

    Node createMissingDeclaration();

    Node createComputedPropertyName(Node expression);

    Node createPrivateIdentifier(string name);
};

#endif // NODEFACTORY_H