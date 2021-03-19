#ifndef NODEFACTORY_H
#define NODEFACTORY_H

#include <functional>

typedef std::function<Node(ScriptKind)> NodeCreate;

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
public:
};

#endif // NODEFACTORY_H