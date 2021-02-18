#ifndef VISITOR_AST_H
#define VISITOR_AST_H

#include "AST.h"

class VisitorASTBase : public VisitorAST
{
public:
    virtual void visit(NodeAST *node) override
    {
        action(node);
    }

protected:
    virtual void action(NodeAST *node) = 0;
};

template< typename T >
class FilterVisitorAST: public VisitorASTBase
{
    SyntaxKind kind;
    std::function<void (T*)> functor;
public:
    FilterVisitorAST(SyntaxKind kind, std::function<void (T*)> functor) 
        : kind(kind), functor(functor)
    {
    }

protected:
    virtual void action(NodeAST *node) override
    {
        if (node && kind == node->getKind())
        {
            functor(dynamic_cast<T*>(node));
        }
    }
};

#endif // VISITOR_AST_H