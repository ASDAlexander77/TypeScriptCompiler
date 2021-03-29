#include "parenthesizer_rules.h"

namespace ts 
{
    auto ParenthesizerRules::parenthesizeExpressionOfComputedPropertyName(Expression expression) -> Expression
    {
        return isCommaSequence(expression) ? factory->createParenthesizedExpression(expression) : expression;
    }

    auto ParenthesizerRules::parenthesizeExpressionsOfCommaDelimitedList(NodeArray<Expression> elements) -> NodeArray<Expression> {
        auto result = sameMap(elements, parenthesizeExpressionForDisallowedComma);
        return setTextRange(createNodeArray(result, elements->hasTrailingComma), elements);
    }

    auto ParenthesizerRules::parenthesizeExpressionForDisallowedComma(Expression expression) -> Expression {
        auto emittedExpression = skipPartiallyEmittedExpressions(expression);
        auto expressionPrecedence = getExpressionPrecedence(emittedExpression);
        auto commaPrecedence = getOperatorPrecedence(SyntaxKind::BinaryExpression, SyntaxKind::CommaToken);
        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return expressionPrecedence > commaPrecedence ? expression : setTextRange(factory->createParenthesizedExpression(expression), expression);
    }

    auto ParenthesizerRules::parenthesizeLeftSideOfAccess(Expression expression) -> LeftHandSideExpression {
        // isLeftHandSideExpression is almost the correct criterion for when it is not necessary
        // to parenthesize the expression before a dot. The known exception is:
        //
        //    NewExpression:
        //       new C.x        -> not the same as (new C).x
        //
        auto emittedExpression = skipPartiallyEmittedExpressions(expression);
        if (isLeftHandSideExpression(emittedExpression)
            && (emittedExpression->kind != SyntaxKind::NewExpression || emittedExpression.as<NewExpression>()->arguments)) {
            // TODO(rbuckton) -> Verify whether this assertion holds.
            return expression.as<LeftHandSideExpression>();
        }

        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return setTextRange(factory->createParenthesizedExpression(expression), expression);
    }

    auto ParenthesizerRules::parenthesizeTypeArguments(NodeArray<TypeNode> typeArguments) -> NodeArray<TypeNode> {
        if (some(typeArguments)) {
            return factory->createNodeArray(sameMap(typeArguments, parenthesizeOrdinalTypeArgument));
        }
    }

    auto ParenthesizerRules::parenthesizeElementTypeOfArrayType(TypeNode ember) -> TypeNode {
        switch (member->kind) {
            case SyntaxKind::TypeQuery:
            case SyntaxKind::TypeOperator:
            case SyntaxKind::InferType:
                return factory->createParenthesizedType(member);
        }
        return parenthesizeMemberOfElementType(member);
    }

    auto ParenthesizerRules::parenthesizeConstituentTypesOfUnionOrIntersectionType(NodeArray<TypeNode> members) -> NodeArray<TypeNode> {
        return factory->createNodeArray(sameMap(members, parenthesizeMemberOfElementType));
    }

    auto ParenthesizerRules::parenthesizeMemberOfConditionalType(TypeNode member) -> TypeNode {
        return member->kind == SyntaxKind::ConditionalType ? factory->createParenthesizedType(member) : member;
    }

    auto ParenthesizerRules::parenthesizeMemberOfElementType(TypeNode member) -> TypeNode {
        switch (member->kind) {
            case SyntaxKind::UnionType:
            case SyntaxKind::IntersectionType:
            case SyntaxKind::FunctionType:
            case SyntaxKind::ConstructorType:
                return factory->createParenthesizedType(member);
        }
        return parenthesizeMemberOfConditionalType(member);
    }

    auto ParenthesizerRules::parenthesizeExpressionOfNew(Expression expression) -> LeftHandSideExpression {
        auto leftmostExpr = getLeftmostExpression(expression, /*stopAtCallExpressions*/ true);
        switch (leftmostExpr->kind) {
            case SyntaxKind::CallExpression:
                return factory->createParenthesizedExpression(expression);

            case SyntaxKind::NewExpression:
                return !leftmostExpr.as<NewExpression>().arguments
                    ? factory->createParenthesizedExpression(expression)
                    : expression.as<LeftHandSideExpression>(); // TODO(rbuckton) -> Verify this assertion holds
        }

        return parenthesizeLeftSideOfAccess(expression);
    }

    auto ParenthesizerRules::parenthesizeOperandOfPrefixUnary(Expression operand) -> UnaryExpression {
        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return isUnaryExpression(operand) ? operand : setTextRange(factory->createParenthesizedExpression(operand), operand);
    }

    auto ParenthesizerRules::parenthesizeConciseBodyOfArrowFunction(ConciseBody body) -> ConciseBody {
        if (!isBlock(body) && (isCommaSequence(body) || getLeftmostExpression(body, /*stopAtCallExpressions*/ false)->kind == SyntaxKind::ObjectLiteralExpression)) {
            // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
            return setTextRange(factory->createParenthesizedExpression(body), body);
        }

        return body;
    }

    auto ParenthesizerRules::parenthesizeOperandOfPostfixUnary(Expression operand) -> LeftHandSideExpression {
        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return isLeftHandSideExpression(operand) ? operand : setTextRange(factory->createParenthesizedExpression(operand), operand);
    }

    auto ParenthesizerRules::parenthesizeLeftSideOfBinary(SyntaxKind binaryOperator, Expression leftSide) -> Expression {
        return parenthesizeBinaryOperand(binaryOperator, leftSide, /*isLeftSideOfBinary*/ true);
    }

    auto ParenthesizerRules::parenthesizeRightSideOfBinary(SyntaxKind binaryOperator, Expression leftSide, Expression rightSide) -> Expression {
        return parenthesizeBinaryOperand(binaryOperator, rightSide, /*isLeftSideOfBinary*/ false, leftSide);
    }

    auto ParenthesizerRules::parenthesizeConditionOfConditionalExpression(Expression condition) -> Expression {
        auto conditionalPrecedence = getOperatorPrecedence(SyntaxKind::ConditionalExpression, SyntaxKind::QuestionToken);
        auto emittedCondition = skipPartiallyEmittedExpressions(condition);
        auto conditionPrecedence = getExpressionPrecedence(emittedCondition);
        if (compareValues(conditionPrecedence, conditionalPrecedence) != Comparison::GreaterThan) {
            return factory->createParenthesizedExpression(condition);
        }
        return condition;
    }

    auto ParenthesizerRules::parenthesizeBranchOfConditionalExpression(Expression branch) -> Expression {
        // per ES grammar both 'whenTrue' and 'whenFalse' parts of conditional expression are assignment expressions
        // so in case when comma expression is introduced as a part of previous transformations
        // if should be wrapped in parens since comma operator has the lowest precedence
        auto emittedExpression = skipPartiallyEmittedExpressions(branch);
        return isCommaSequence(emittedExpression)
            ? factory->createParenthesizedExpression(branch)
            : branch;
    }

    /**
     *  [Per the spec](https://tc39.github.io/ecma262/#prod-ExportDeclaration), `export default` accepts _AssigmentExpression_ but
        *  has a lookahead restriction for `function`, `async function`, and `class`.
        *
        * Basically, that means we need to parenthesize in the following cases:
        *
        * - BinaryExpression of CommaToken
        * - CommaList (synthetic list of multiple comma expressions)
        * - FunctionExpression
        * - ClassExpression
        */
    auto ParenthesizerRules::parenthesizeExpressionOfExportDefault(Expression expression) -> Expression {
        auto check = skipPartiallyEmittedExpressions(expression);
        auto needsParens = isCommaSequence(check);
        if (!needsParens) {
            switch (getLeftmostExpression(check, /*stopAtCallExpression*/ false)->kind) {
                case SyntaxKind::ClassExpression:
                case SyntaxKind::FunctionExpression:
                    needsParens = true;
            }
        }
        return needsParens ? factory->createParenthesizedExpression(expression) : expression;
    }

    /**
     * Wraps an expression in parentheses if it is needed in order to use the expression
        * as the expression of a `NewExpression` node.
        */
    auto ParenthesizerRules::parenthesizeExpressionOfNew(Expression expression) -> LeftHandSideExpression {
        auto leftmostExpr = getLeftmostExpression(expression, /*stopAtCallExpressions*/ true);
        switch (leftmostExpr->kind) {
            case SyntaxKind::CallExpression:
                return factory->createParenthesizedExpression(expression);

            case SyntaxKind::NewExpression:
                return !(leftmostExpr.as<NewExpression>()).arguments
                    ? factory->createParenthesizedExpression(expression)
                    : expression as LeftHandSideExpression; // TODO(rbuckton) -> Verify this assertion holds
        }

        return parenthesizeLeftSideOfAccess(expression);
    }

    /**
     * Wraps an expression in parentheses if it is needed in order to use the expression for
        * property or element access.
        */
    auto ParenthesizerRules::parenthesizeLeftSideOfAccess(Expression expression) -> LeftHandSideExpression {
        // isLeftHandSideExpression is almost the correct criterion for when it is not necessary
        // to parenthesize the expression before a dot. The known exception is:
        //
        //    NewExpression:
        //       new C.x        -> not the same as (new C).x
        //
        auto emittedExpression = skipPartiallyEmittedExpressions(expression);
        if (isLeftHandSideExpression(emittedExpression)
            && (emittedExpression->kind != SyntaxKind::NewExpression || (<NewExpression>emittedExpression).arguments)) {
            // TODO(rbuckton) -> Verify whether this assertion holds.
            return expression as LeftHandSideExpression;
        }

        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return setTextRange(factory->createParenthesizedExpression(expression), expression);
    }

    auto ParenthesizerRules::parenthesizeOperandOfPostfixUnary(Expression operand) -> LeftHandSideExpression {
        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return isLeftHandSideExpression(operand) ? operand : setTextRange(factory->createParenthesizedExpression(operand), operand);
    }

    auto ParenthesizerRules::parenthesizeOperandOfPrefixUnary(Expression operand) -> UnaryExpression {
        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return isUnaryExpression(operand) ? operand : setTextRange(factory->createParenthesizedExpression(operand), operand);
    }

    auto ParenthesizerRules::parenthesizeExpressionsOfCommaDelimitedList(NodeArray<Expression> elements) -> NodeArray<Expression> {
        auto result = sameMap(elements, parenthesizeExpressionForDisallowedComma);
        return setTextRange(factory->createNodeArray(result, elements.hasTrailingComma), elements);
    }

    auto ParenthesizerRules::parenthesizeExpressionForDisallowedComma(Expression expression) -> Expression {
        auto emittedExpression = skipPartiallyEmittedExpressions(expression);
        auto expressionPrecedence = getExpressionPrecedence(emittedExpression);
        auto commaPrecedence = getOperatorPrecedence(SyntaxKind::BinaryExpression, SyntaxKind::CommaToken);
        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return expressionPrecedence > commaPrecedence ? expression : setTextRange(factory->createParenthesizedExpression(expression), expression);
    }

    auto ParenthesizerRules::parenthesizeExpressionOfExpressionStatement(Expression expression) -> Expression {
        auto emittedExpression = skipPartiallyEmittedExpressions(expression);
        if (isCallExpression(emittedExpression)) {
            auto callee = emittedExpression.expression;
            auto kind = skipPartiallyEmittedExpressions(callee)->kind;
            if (kind == SyntaxKind::FunctionExpression || kind == SyntaxKind::ArrowFunction) {
                // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
                auto updated = factory->updateCallExpression(
                    emittedExpression,
                    setTextRange(factory->createParenthesizedExpression(callee), callee),
                    emittedExpression.typeArguments,
                    emittedExpression.arguments
                );
                return factory->restoreOuterExpressions(expression, updated, OuterExpressionKinds.PartiallyEmittedExpressions);
            }
        }

        auto leftmostExpressionKind = getLeftmostExpression(emittedExpression, /*stopAtCallExpressions*/ false)->kind;
        if (leftmostExpressionKind == SyntaxKind::ObjectLiteralExpression || leftmostExpressionKind == SyntaxKind::FunctionExpression) {
            // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
            return setTextRange(factory->createParenthesizedExpression(expression), expression);
        }

        return expression;
    }

    auto ParenthesizerRules::parenthesizeOrdinalTypeArgument(TypeNode node, number i) {
        return i == 0 && isFunctionOrConstructorTypeNode(node) && node->typeParameters ? factory->createParenthesizedType(node) : node;
    }

    auto ParenthesizerRules::parenthesizeTypeArguments(NodeArray<TypeNode> typeArguments) -> NodeArray<TypeNode> {
        if (some(typeArguments)) {
            return factory->createNodeArray(sameMap(typeArguments, parenthesizeOrdinalTypeArgument));
        }
    }

}
