#ifndef VISITOR_AST_H
#define VISITOR_AST_H

#include "AST.h"

template< typename T >
class FilterVisitorAST: public VisitorAST
{
    SyntaxKind kind;
    std::function<void (T*)> functor;
public:
    FilterVisitorAST(SyntaxKind kind, std::function<void (T*)> functor) 
        : kind(kind), functor(functor)
    {
    }

    virtual void visit(NodeAST *node) override
    {
        if (node && kind == node->getKind())
        {
            functor(dynamic_cast<T*>(node));
        }
    }
};

#endif // VISITOR_AST_H