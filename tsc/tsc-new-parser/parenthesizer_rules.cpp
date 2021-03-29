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
}
