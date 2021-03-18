#ifndef PARSER_H
#define PARSER_H

#include "scanner.h"

enum class SignatureFlags : number {
    None = 0,
    Yield = 1 << 0,
    Await = 1 << 1,
    Type  = 1 << 2,
    IgnoreMissingOpenBrace = 1 << 4,
    JSDoc = 1 << 5,
};

enum class SpeculationKind : number {
    TryParse,
    Lookahead,
    Reparse
};

class Node
{
    SyntaxKind kind;

    template <typename T> 
    auto as() -> T
    {
        return T();
    }

    operator bool()
    {
        return this->kind != SyntaxKind::Unknown;
    }

    Node operator||(Node rhs)
    {
        if (*this)
        {
            return *this;
        }

        return rhs;
    }
};

typedef std::vector<Node> NodeArray;

template <typename T>
using NodeFuncT = std::function<Node(T)>;

template <typename T>
using NodeArrayFuncT = std::function<NodeArray(T)>;

typedef std::function<Node(SyntaxKind, number, number)> NodeCreateFunc;

#endif // PARSER_H