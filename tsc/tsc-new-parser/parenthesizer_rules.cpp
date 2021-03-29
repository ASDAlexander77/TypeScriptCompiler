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
}
