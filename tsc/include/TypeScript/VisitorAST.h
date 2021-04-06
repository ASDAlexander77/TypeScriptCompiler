#ifndef VISITOR_AST_H
#define VISITOR_AST_H

#include "parser.h"
#include "utilities.h"

namespace ts
{
    class VisitorAST {
    public:
        virtual ~VisitorAST() {};
        virtual void visitNode(Node tree) = 0;

        void visit(Node node)
        {
            ts::FuncT<> visitNode;
            ts::ArrayFuncT<> visitArray;

            visitNode = [&](ts::Node child) -> ts::Node 
            {
                visitNode(child);
                ts::forEachChild(child, visitNode, visitArray);    
                return undefined;
            };

            visitArray = [&](ts::NodeArray<ts::Node> array) -> ts::Node {
                for (auto node : array)
                {
                    visitNode(node);
                }

                return undefined;
            };

            auto result = ts::forEachChild(node, visitNode, visitArray);
        }
    }; 

    template< typename T >
    class FilterVisitorAST: public VisitorAST
    {
        SyntaxKind kind;
        std::function<void (T)> functor;
    public:
        FilterVisitorAST(SyntaxKind kind, std::function<void (T)> functor) 
            : kind(kind), functor(functor)
        {
        }

        virtual void visitNode(Node node) override
        {
            if (kind == node)
            {
                functor(node.as<T>());
            }
        }
    };

}

#endif // VISITOR_AST_H