#ifndef NODEFACTORY_H
#define NODEFACTORY_H

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

    Node updateSourceFile(SourceFile, Node);

    Node createNodeArray(Node);

    SourceFile createSourceFile(Node statements, Node endOfFileToken, NodeFlags flags);
};

#endif // NODEFACTORY_H