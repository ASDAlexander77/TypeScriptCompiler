/* @internal */
#ifndef PARENTHESIZERRULES_H
#define PARENTHESIZERRULES_H

#include "enums.h"
#include "scanner.h"
#include "utilities.h"

namespace ts 
{
    class NodeFactory;

    struct ParenthesizerRules
    {
        NodeFactory *factory;

        ParenthesizerRules(NodeFactory *factory) : factory(factory) {}

        auto parenthesizeExpressionOfComputedPropertyName(Expression expression) -> Expression;

        auto parenthesizeExpressionsOfCommaDelimitedList(NodeArray<Expression> elements) -> NodeArray<Expression>;

        auto parenthesizeExpressionForDisallowedComma(Expression expression) -> Expression;

        auto parenthesizeLeftSideOfAccess(Expression expression) -> LeftHandSideExpression;

        auto parenthesizeTypeArguments(NodeArray<TypeNode> typeArguments) -> NodeArray<TypeNode>;

        auto parenthesizeElementTypeOfArrayType(TypeNode ember) -> TypeNode;

        auto parenthesizeConstituentTypesOfUnionOrIntersectionType(NodeArray<TypeNode> members) -> NodeArray<TypeNode>;

        auto parenthesizeMemberOfConditionalType(TypeNode member) -> TypeNode;

        auto parenthesizeMemberOfElementType(TypeNode member) -> TypeNode;

        auto parenthesizeExpressionOfNew(Expression expression) -> LeftHandSideExpression;
    };
}

#endif