#ifndef INCREMENTAL_PARSER_H
#define INCREMENTAL_PARSER_H

#include "config.h"

namespace IncrementalParser
{
    struct IncrementalElement : TextRange
    {
        Node parent;
        boolean intersectsChange;
        number length;
        NodeArray<PTR(Node)> _children;
    };

    struct IncrementalNode : Node, IncrementalElement
    {
        boolean hasBeenIncrementallyParsed;

        inline auto operator->()
        {
            return Node::operator->();
        }

        inline operator bool()
        {
            return Node::operator bool();
        }
    };

    struct IncrementalNodeArray : NodeArray<IncrementalNode>, IncrementalElement
    {
        number length;
    };

    // Allows finding nodes in the source file at a certain position in an efficient manner.
    // The implementation takes advantage of the calling pattern it knows the parser will
    // make in order to optimize finding nodes as quickly as possible.
    struct SyntaxCursor
    {
        SyntaxCursor(){};
        SyntaxCursor(std::function<IncrementalNode(number)> currentNode) : currentNode{currentNode} {};
        SyntaxCursor(undefined_t){};

        inline operator bool()
        {
            return !!currentNode;
        }

        std::function<IncrementalNode(number)> currentNode;
    };

    auto createSyntaxCursor(SourceFile sourceFile) -> SyntaxCursor;
}

#endif // INCREMENTAL_PARSER_H