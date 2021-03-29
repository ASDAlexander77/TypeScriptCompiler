namespace ts 
{
    auto ParenthesizerRules::parenthesizeExpressionOfComputedPropertyName(Expression expression) -> Expression
    {
        return isCommaSequence(expression) ? createParenthesizedExpression(expression) : expression;
    }

    auto ParenthesizerRules::parenthesizeExpressionsOfCommaDelimitedList(NodeArray<Expression> elements) -> NodeArray<Expression> {
        auto result = sameMap(elements, parenthesizeExpressionForDisallowedComma);
        return setTextRange(createNodeArray(result, elements->hasTrailingComma), elements);
    }

    auto ParenthesizerRules::parenthesizeExpressionForDisallowedComma(Expression expression) -> Expression {
        auto emittedExpression = skipPartiallyEmittedExpressions(expression);
        auto expressionPrecedence = getExpressionPrecedence(emittedExpression);
        auto commaPrecedence = getOperatorPrecedence(SyntaxKind.BinaryExpression, SyntaxKind.CommaToken);
        // TODO(rbuckton): Verifiy whether `setTextRange` is needed.
        return expressionPrecedence > commaPrecedence ? expression : setTextRange(factory->createParenthesizedExpression(expression), expression);
    }

    auto parenthesizeLeftSideOfAccess(Expression expression) -> LeftHandSideExpression {
        // isLeftHandSideExpression is almost the correct criterion for when it is not necessary
        // to parenthesize the expression before a dot. The known exception is:
        //
        //    NewExpression:
        //       new C.x        -> not the same as (new C).x
        //
        auto emittedExpression = skipPartiallyEmittedExpressions(expression);
        if (isLeftHandSideExpression(emittedExpression)
            && (emittedExpression->kind != SyntaxKind.NewExpression || emittedExpression.as<NewExpression>().arguments)) {
            // TODO(rbuckton): Verify whether this assertion holds.
            return expression.as<LeftHandSideExpression>();
        }

        // TODO(rbuckton): Verifiy whether `setTextRange` is needed.
        return setTextRange(factory->createParenthesizedExpression(expression), expression);
    }

    auto parenthesizeTypeArguments(NodeArray<TypeNode> typeArguments) -> NodeArray<TypeNode> {
        if (some(typeArguments)) {
            return factory->createNodeArray(sameMap(typeArguments, parenthesizeOrdinalTypeArgument));
        }
    }

    auto parenthesizeElementTypeOfArrayType(TypeNode ember) -> TypeNode {
        switch (member->kind) {
            case SyntaxKind::TypeQuery:
            case SyntaxKind::TypeOperator:
            case SyntaxKind::InferType:
                return factory->createParenthesizedType(member);
        }
        return parenthesizeMemberOfElementType(member);
    }

    auto parenthesizeConstituentTypesOfUnionOrIntersectionType(NodeArray<TypeNode> members) -> NodeArray<TypeNode> {
        return factory->createNodeArray(sameMap(members, parenthesizeMemberOfElementType));
    }

    auto parenthesizeMemberOfConditionalType(TypeNode member) -> TypeNode {
        return member->kind == SyntaxKind::ConditionalType ? factory->createParenthesizedType(member) : member;
    }

    auto parenthesizeMemberOfElementType(TypeNode member) -> TypeNode {
        switch (member->kind) {
            case SyntaxKind::UnionType:
            case SyntaxKind::IntersectionType:
            case SyntaxKind::FunctionType:
            case SyntaxKind::ConstructorType:
                return factory->createParenthesizedType(member);
        }
        return parenthesizeMemberOfConditionalType(member);
    }

    auto parenthesizeExpressionOfNew(Expression expression) -> LeftHandSideExpression {
        auto leftmostExpr = getLeftmostExpression(expression, /*stopAtCallExpressions*/ true);
        switch (leftmostExpr->kind) {
            case SyntaxKind::CallExpression:
                return factory.createParenthesizedExpression(expression);

            case SyntaxKind::NewExpression:
                return !leftmostExpr.as<NewExpression>().arguments
                    ? factory.createParenthesizedExpression(expression)
                    : expression.as<LeftHandSideExpression>(); // TODO(rbuckton): Verify this assertion holds
        }

        return parenthesizeLeftSideOfAccess(expression);
    }
}
