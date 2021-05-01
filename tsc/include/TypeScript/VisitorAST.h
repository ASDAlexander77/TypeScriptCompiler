#ifndef VISITOR_AST_H
#define VISITOR_AST_H

#include "parser.h"
#include "utilities.h"

namespace ts
{
    class VisitorAST {
    protected:
        virtual ~VisitorAST() {};
        virtual void visitTree(Node tree) = 0;
        virtual bool isFiltered(Node tree) 
        {
            return false;
        };

    public:
        void visit(Node node)
        {
            ts::FuncT<> visitNode;
            ts::ArrayFuncT<> visitArray;

            visitNode = [&](ts::Node child) -> ts::Node 
            {
                if (isFiltered(child))
                {
                    return undefined;    
                }

                visitTree(child);
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
    protected:
        SyntaxKind kind;
        std::function<void (T)> functor;
    public:
        FilterVisitorAST(SyntaxKind kind, std::function<void (T)> functor) 
            : kind(kind), functor(functor)
        {
        }

    protected:
        virtual void visitTree(Node node) override
        {
            if (kind == node)
            {
                functor(node.as<T>());
            }
        }    
    };

    template< typename T >
    class FilterVisitorSkipFuncsAST: public FilterVisitorAST<T>
    {
    public:
        FilterVisitorSkipFuncsAST(SyntaxKind kind, std::function<void (T)> functor) 
            : FilterVisitorAST(kind, functor)
        {
        }

    protected:
        virtual bool isFiltered(Node node) override
        {
            SyntaxKind currentkind = node;
            switch (currentkind)
            {
                case SyntaxKind::MethodDeclaration:
                case SyntaxKind::Constructor:
                case SyntaxKind::GetAccessor:
                case SyntaxKind::SetAccessor:
                case SyntaxKind::FunctionExpression:
                case SyntaxKind::FunctionDeclaration:
                case SyntaxKind::ArrowFunction:               
                //
                case SyntaxKind::ClassDeclaration:
                case SyntaxKind::ClassExpression:                 
                case SyntaxKind::InterfaceDeclaration:
                    return true;
                default:
                    return false;
            }
        }        
    };
}

#endif // VISITOR_AST_H