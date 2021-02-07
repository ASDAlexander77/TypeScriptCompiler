#ifndef VISITOR_AST_H
#define VISITOR_AST_H

#include "antlr4-runtime.h"

class VisitorAST : public antlr4::tree::ParseTreeVisitor
{
public:
    virtual antlrcpp::Any visit(antlr4::tree::ParseTree *tree) override
    {
#if _DEBUG
    std::cout << "visit >>> " << tree->getText() << std::endl;
#endif        

        action(tree);

        visitChildren(tree);

        return antlrcpp::Any();
    }

protected:
    virtual void action(antlr4::tree::ParseTree *tree) = 0;

    virtual antlrcpp::Any visitChildren(antlr4::tree::ParseTree *node) override
    {
        for (const auto& child : node->children)
        {
            visit(child);
        }

        return antlrcpp::Any();
    }

    virtual antlrcpp::Any visitTerminal(antlr4::tree::TerminalNode *node) override
    {
        return antlrcpp::Any();
    }

    virtual antlrcpp::Any visitErrorNode(antlr4::tree::ErrorNode *node) override
    {
        return antlrcpp::Any();
    }
};

template< typename T >
class FilterVisitorAST: public VisitorAST
{
public:
    FilterVisitorAST(std::function<void (T*)> functor) 
        : functor(functor)
    {
    }

protected:
    virtual void action(antlr4::tree::ParseTree *tree) override
    {
        auto *functionDeclaration = dynamic_cast<T*>(tree);
        if (functionDeclaration != nullptr)
        {
            functor(functionDeclaration);
        }
    }

    std::function<void (T*)> functor;
};

#endif // VISITOR_AST_H