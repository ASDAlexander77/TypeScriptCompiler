#include "parenthesizer_rules.h"
#include "node_factory.h"

namespace ts
{
auto ParenthesizerRules::parenthesizeExpressionOfComputedPropertyName(Expression expression) -> Expression
{
    if (factory->NoParenthesizerRules())
    {
        return expression;
    }

    return isCommaSequence(expression) ? factory->createParenthesizedExpression(expression).as<Expression>() : expression;
}

auto ParenthesizerRules::parenthesizeExpressionsOfCommaDelimitedList(NodeArray<Expression> elements) -> NodeArray<Expression>
{
    if (factory->NoParenthesizerRules())
    {
        return elements;
    }

    auto result = sameMap(elements, std::bind(&ParenthesizerRules::parenthesizeExpressionForDisallowedComma, this, std::placeholders::_1));
    return setTextRange(factory->createNodeArray(result, elements->hasTrailingComma), static_cast<data::TextRange>(elements));
}

auto ParenthesizerRules::parenthesizeExpressionForDisallowedComma(Expression expression) -> Expression
{
    if (factory->NoParenthesizerRules())
    {
        return expression;
    }

    auto emittedExpression = skipPartiallyEmittedExpressions(expression);
    auto expressionPrecedence = getExpressionPrecedence(emittedExpression);
    auto commaPrecedence = getOperatorPrecedence(SyntaxKind::BinaryExpression, SyntaxKind::CommaToken);
    // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
    return expressionPrecedence > commaPrecedence
               ? expression
               : setTextRange(factory->createParenthesizedExpression(expression).as<Expression>(), expression);
}

auto ParenthesizerRules::parenthesizeLeftSideOfAccess(Expression expression, boolean optionalChain) -> LeftHandSideExpression
{
    if (factory->NoParenthesizerRules())
    {
        return expression;
    }

    // isLeftHandSideExpression is almost the correct criterion for when it is not necessary
    // to parenthesize the expression before a dot. The known exception is:
    //
    //    NewExpression:
    //       new C.x        -> not the same as (new C).x
    //
    auto emittedExpression = skipPartiallyEmittedExpressions(expression);
    if (isLeftHandSideExpression(emittedExpression) 
        && (emittedExpression != SyntaxKind::NewExpression || emittedExpression.as<NewExpression>()->arguments)
        && (optionalChain || !isOptionalChain(emittedExpression)))
    {
        // TODO(rbuckton) -> Verify whether this assertion holds.
        return expression.as<LeftHandSideExpression>();
    }

    // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
    return setTextRange(factory->createParenthesizedExpression(expression), expression);
}

auto ParenthesizerRules::parenthesizeTypeArguments(NodeArray<TypeNode> typeArguments) -> NodeArray<TypeNode>
{
    if (factory->NoParenthesizerRules())
    {
        return typeArguments;
    }

    if (some(typeArguments))
    {
        return factory->createNodeArray(sameMapWithNumber(typeArguments, std::bind(&ParenthesizerRules::parenthesizeOrdinalTypeArgument,
                                                                                   this, std::placeholders::_1, std::placeholders::_2)));
    }

    return undefined;
}

auto ParenthesizerRules::parenthesizeElementTypeOfArrayType(TypeNode member) -> TypeNode
{
    if (factory->NoParenthesizerRules())
    {
        return member;
    }

    switch ((SyntaxKind)member)
    {
    case SyntaxKind::TypeQuery:
    case SyntaxKind::TypeOperator:
    case SyntaxKind::InferType:
        return factory->createParenthesizedType(member);
    }
    return parenthesizeMemberOfElementType(member);
}

auto ParenthesizerRules::parenthesizeConstituentTypesOfUnionOrIntersectionType(NodeArray<TypeNode> members) -> NodeArray<TypeNode>
{
    if (factory->NoParenthesizerRules())
    {
        return members;
    }

    return factory->createNodeArray(
        sameMap(members, std::bind(&ParenthesizerRules::parenthesizeMemberOfElementType, this, std::placeholders::_1)));
}

auto ParenthesizerRules::parenthesizeMemberOfConditionalType(TypeNode member) -> TypeNode
{
    if (factory->NoParenthesizerRules())
    {
        return member;
    }

    return member == SyntaxKind::ConditionalType ? factory->createParenthesizedType(member).as<TypeNode>() : member;
}

auto ParenthesizerRules::parenthesizeMemberOfElementType(TypeNode member) -> TypeNode
{
    if (factory->NoParenthesizerRules())
    {
        return member;
    }

    switch ((SyntaxKind)member)
    {
    case SyntaxKind::UnionType:
    case SyntaxKind::IntersectionType:
    case SyntaxKind::FunctionType:
    case SyntaxKind::ConstructorType:
        return factory->createParenthesizedType(member);
    }
    return parenthesizeMemberOfConditionalType(member);
}

auto ParenthesizerRules::parenthesizeExpressionOfNew(Expression expression) -> LeftHandSideExpression
{
    if (factory->NoParenthesizerRules())
    {
        return expression;
    }

    auto leftmostExpr = getLeftmostExpression(expression, /*stopAtCallExpressions*/ true);
    switch ((SyntaxKind)leftmostExpr)
    {
    case SyntaxKind::CallExpression:
        return factory->createParenthesizedExpression(expression);

    case SyntaxKind::NewExpression:
        return !leftmostExpr.as<NewExpression>()->arguments
                   ? factory->createParenthesizedExpression(expression).as<LeftHandSideExpression>()
                   : expression.as<LeftHandSideExpression>(); // TODO(rbuckton) -> Verify this assertion holds
    }

    return parenthesizeLeftSideOfAccess(expression);
}

auto ParenthesizerRules::parenthesizeOperandOfPrefixUnary(Expression operand) -> UnaryExpression
{
    if (factory->NoParenthesizerRules())
    {
        return operand;
    }

    // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
    return isUnaryExpression(operand) ? operand : setTextRange(factory->createParenthesizedExpression(operand).as<Expression>(), operand);
}

auto ParenthesizerRules::parenthesizeConciseBodyOfArrowFunction(ConciseBody body) -> ConciseBody
{
    if (factory->NoParenthesizerRules())
    {
        return body;
    }

    if (!isBlock(body) &&
        (isCommaSequence(body) || getLeftmostExpression(body, /*stopAtCallExpressions*/ false) == SyntaxKind::ObjectLiteralExpression))
    {
        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return setTextRange(factory->createParenthesizedExpression(body), body);
    }

    return body;
}

auto ParenthesizerRules::parenthesizeOperandOfPostfixUnary(Expression operand) -> LeftHandSideExpression
{
    if (factory->NoParenthesizerRules())
    {
        return operand;
    }

    // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
    return isLeftHandSideExpression(operand) ? operand
                                             : setTextRange(factory->createParenthesizedExpression(operand).as<Expression>(), operand);
}

inline static auto operatorHasAssociativeProperty(SyntaxKind binaryOperator)
{
    // The following operators are associative in JavaScript:
    //  (a*b)*c     -> a*(b*c)  -> a*b*c
    //  (a|b)|c     -> a|(b|c)  -> a|b|c
    //  (a&b)&c     -> a&(b&c)  -> a&b&c
    //  (a^b)^c     -> a^(b^c)  -> a^b^c
    //
    // While addition is associative in mathematics, JavaScript's `+` is not
    // guaranteed to be associative as it is overloaded with string concatenation.
    return binaryOperator == SyntaxKind::AsteriskToken || binaryOperator == SyntaxKind::BarToken ||
           binaryOperator == SyntaxKind::AmpersandToken || binaryOperator == SyntaxKind::CaretToken;
}

auto getLiteralKindOfBinaryPlusOperand(Expression node1) -> SyntaxKind
{
    auto node = skipPartiallyEmittedExpressions(node1);

    if (isLiteralKind(node))
    {
        return node;
    }

    if (node == SyntaxKind::BinaryExpression && (node.as<BinaryExpression>())->operatorToken == SyntaxKind::PlusToken)
    {
        if (node.as<BinaryExpression>()->cachedLiteralKind != SyntaxKind::Unknown)
        {
            return node.as<BinaryExpression>()->cachedLiteralKind;
        }

        auto leftKind = getLiteralKindOfBinaryPlusOperand(node.as<BinaryExpression>()->left);
        auto literalKind = isLiteralKind(leftKind) && leftKind == getLiteralKindOfBinaryPlusOperand(node.as<BinaryExpression>()->right)
                               ? leftKind
                               : SyntaxKind::Unknown;

        node.as<BinaryExpression>()->cachedLiteralKind = literalKind;
        return literalKind;
    }

    return SyntaxKind::Unknown;
}

auto ParenthesizerRules::binaryOperandNeedsParentheses(SyntaxKind binaryOperator, Expression operand, boolean isLeftSideOfBinary,
                                                       Expression leftOperand) -> boolean
{
    if (factory->NoParenthesizerRules())
    {
        return false;
    }

    // If the operand has lower precedence, then it needs to be parenthesized to preserve the
    // intent of the expression. For example, if the operand is `a + b` and the operator is
    // `*`, then we need to parenthesize the operand to preserve the intended order of
    // operations: `(a + b) * x`.
    //
    // If the operand has higher precedence, then it does not need to be parenthesized. For
    // example, if the operand is `a * b` and the operator is `+`, then we do not need to
    // parenthesize to preserve the intended order of operations: `a * b + x`.
    //
    // If the operand has the same precedence, then we need to check the associativity of
    // the operator based on whether this is the left or right operand of the expression.
    //
    // For example, if `a / d` is on the right of operator `*`, we need to parenthesize
    // to preserve the intended order of operations: `x * (a / d)`
    //
    // If `a ** d` is on the left of operator `**`, we need to parenthesize to preserve
    // the intended order of operations: `(a ** b) ** c`
    auto binaryOperatorPrecedence = getOperatorPrecedence(SyntaxKind::BinaryExpression, binaryOperator);
    auto binaryOperatorAssociativity = getOperatorAssociativity(SyntaxKind::BinaryExpression, binaryOperator);
    auto emittedOperand = skipPartiallyEmittedExpressions(operand);
    if (!isLeftSideOfBinary && operand == SyntaxKind::ArrowFunction && binaryOperatorPrecedence > OperatorPrecedence::Assignment)
    {
        // We need to parenthesize arrow functions on the right side to avoid it being
        // parsed as parenthesized expression: `a && (() => {})`
        return true;
    }
    auto operandPrecedence = getExpressionPrecedence(emittedOperand);
    switch (compareValues(operandPrecedence, binaryOperatorPrecedence))
    {
    case Comparison::LessThan:
        // If the operand is the right side of a right-associative binary operation
        // and is a yield expression, then we do not need parentheses.
        if (!isLeftSideOfBinary && binaryOperatorAssociativity == Associativity::Right && operand == SyntaxKind::YieldExpression)
        {
            return false;
        }

        return true;

    case Comparison::GreaterThan:
        return false;

    case Comparison::EqualTo:
        if (isLeftSideOfBinary)
        {
            // No need to parenthesize the left operand when the binary operator is
            // left associative:
            //  (a*b)/x    -> a*b/x
            //  (a**b)/x   -> a**b/x
            //
            // Parentheses are needed for the left operand when the binary operator is
            // right associative:
            //  (a/b)**x   -> (a/b)**x
            //  (a**b)**x  -> (a**b)**x
            return binaryOperatorAssociativity == Associativity::Right;
        }
        else
        {
            if (isBinaryExpression(emittedOperand) && emittedOperand.as<BinaryExpression>()->operatorToken == binaryOperator)
            {
                // No need to parenthesize the right operand when the binary operator and
                // operand are the same and one of the following:
                //  x*(a*b)     => x*a*b
                //  x|(a|b)     => x|a|b
                //  x&(a&b)     => x&a&b
                //  x^(a^b)     => x^a^b
                if (operatorHasAssociativeProperty(binaryOperator))
                {
                    return false;
                }

                // No need to parenthesize the right operand when the binary operator
                // is plus (+) if both the left and right operands consist solely of either
                // literals of the same kind or binary plus (+) expressions for literals of
                // the same kind (recursively).
                //  "a"+(1+2)       => "a"+(1+2)
                //  "a"+("b"+"c")   => "a"+"b"+"c"
                if (binaryOperator == SyntaxKind::PlusToken)
                {
                    auto leftKind = leftOperand ? getLiteralKindOfBinaryPlusOperand(leftOperand) : SyntaxKind::Unknown;
                    if (isLiteralKind(leftKind) && leftKind == getLiteralKindOfBinaryPlusOperand(emittedOperand))
                    {
                        return false;
                    }
                }
            }

            // No need to parenthesize the right operand when the operand is right
            // associative:
            //  x/(a**b)    -> x/a**b
            //  x**(a**b)   -> x**a**b
            //
            // Parentheses are needed for the right operand when the operand is left
            // associative:
            //  x/(a*b)     -> x/(a*b)
            //  x**(a/b)    -> x**(a/b)
            auto operandAssociativity = getExpressionAssociativity(emittedOperand);
            return operandAssociativity == Associativity::Left;
        }
    }

    return false;
}

auto ParenthesizerRules::parenthesizeBinaryOperand(SyntaxKind binaryOperator, Expression operand, boolean isLeftSideOfBinary,
                                                   Expression leftOperand) -> Expression
{
    auto skipped = skipPartiallyEmittedExpressions(operand);

    // If the resulting expression is already parenthesized, we do not need to do any further processing.
    if (skipped == SyntaxKind::ParenthesizedExpression)
    {
        return operand;
    }

    return binaryOperandNeedsParentheses(binaryOperator, operand, isLeftSideOfBinary, leftOperand)
               ? factory->createParenthesizedExpression(operand).as<Expression>()
               : operand;
}

auto ParenthesizerRules::parenthesizeLeftSideOfBinary(SyntaxKind binaryOperator, Expression leftSide) -> Expression
{
    return parenthesizeBinaryOperand(binaryOperator, leftSide, /*isLeftSideOfBinary*/ true);
}

auto ParenthesizerRules::parenthesizeRightSideOfBinary(SyntaxKind binaryOperator, Expression leftSide, Expression rightSide) -> Expression
{
    return parenthesizeBinaryOperand(binaryOperator, rightSide, /*isLeftSideOfBinary*/ false, leftSide);
}

auto ParenthesizerRules::parenthesizeConditionOfConditionalExpression(Expression condition) -> Expression
{
    if (factory->NoParenthesizerRules())
    {
        return condition;
    }

    auto conditionalPrecedence = getOperatorPrecedence(SyntaxKind::ConditionalExpression, SyntaxKind::QuestionToken);
    auto emittedCondition = skipPartiallyEmittedExpressions(condition);
    auto conditionPrecedence = getExpressionPrecedence(emittedCondition);
    if (compareValues(conditionPrecedence, conditionalPrecedence) != Comparison::GreaterThan)
    {
        return factory->createParenthesizedExpression(condition);
    }
    return condition;
}

auto ParenthesizerRules::parenthesizeBranchOfConditionalExpression(Expression branch) -> Expression
{
    if (factory->NoParenthesizerRules())
    {
        return branch;
    }

    // per ES grammar both 'whenTrue' and 'whenFalse' parts of conditional expression are assignment expressions
    // so in case when comma expression is introduced as a part of previous transformations
    // if should be wrapped in parens since comma operator has the lowest precedence
    auto emittedExpression = skipPartiallyEmittedExpressions(branch);
    return isCommaSequence(emittedExpression) ? factory->createParenthesizedExpression(branch).as<Expression>() : branch;
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
auto ParenthesizerRules::parenthesizeExpressionOfExportDefault(Expression expression) -> Expression
{
    if (factory->NoParenthesizerRules())
    {
        return expression;
    }

    auto check = skipPartiallyEmittedExpressions(expression);
    auto needsParens = isCommaSequence(check);
    if (!needsParens)
    {
        switch ((SyntaxKind)getLeftmostExpression(check, /*stopAtCallExpression*/ false))
        {
        case SyntaxKind::ClassExpression:
        case SyntaxKind::FunctionExpression:
            needsParens = true;
        }
    }
    return needsParens ? factory->createParenthesizedExpression(expression).as<Expression>() : expression;
}

auto ParenthesizerRules::parenthesizeExpressionOfExpressionStatement(Expression expression) -> Expression
{
    if (factory->NoParenthesizerRules())
    {
        return expression;
    }

    auto emittedExpression = skipPartiallyEmittedExpressions(expression);
    if (isCallExpression(emittedExpression))
    {
        auto callee = emittedExpression.as<CallExpression>()->expression;
        auto kind = skipPartiallyEmittedExpressions(callee);
        if (kind == SyntaxKind::FunctionExpression || kind == SyntaxKind::ArrowFunction)
        {
            // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
            auto updated = factory->updateCallExpression(
                emittedExpression.as<CallExpression>(), setTextRange(factory->createParenthesizedExpression(callee), callee),
                emittedExpression.as<CallExpression>()->typeArguments, emittedExpression.as<CallExpression>()->arguments);
            // TODO: finish it
            // return factory->restoreOuterExpressions(expression, updated, OuterExpressionKinds::PartiallyEmittedExpressions);
            return expression;
        }
    }

    auto leftmostExpressionKind = getLeftmostExpression(emittedExpression, /*stopAtCallExpressions*/ false);
    if (leftmostExpressionKind == SyntaxKind::ObjectLiteralExpression || leftmostExpressionKind == SyntaxKind::FunctionExpression)
    {
        // TODO(rbuckton) -> Verifiy whether `setTextRange` is needed.
        return setTextRange(factory->createParenthesizedExpression(expression), expression);
    }

    return expression;
}

auto ParenthesizerRules::parenthesizeOrdinalTypeArgument(TypeNode node, number i) -> TypeNode
{
    if (factory->NoParenthesizerRules())
    {
        return node;
    }

    return i == 0 && isFunctionOrConstructorTypeNode(node) && node.as<FunctionOrConstructorTypeNodeBase>()->typeParameters
               ? factory->createParenthesizedType(node).as<TypeNode>()
               : node;
}

auto ParenthesizerRules::parenthesizeCheckTypeOfConditionalType(TypeNode checkType) -> TypeNode {
    switch ((SyntaxKind)checkType) {
        case SyntaxKind::FunctionType:
        case SyntaxKind::ConstructorType:
        case SyntaxKind::ConditionalType:
            return factory->createParenthesizedType(checkType);
    }
    return checkType;
}

auto ParenthesizerRules::parenthesizeConstituentTypeOfUnionType(TypeNode type) -> TypeNode {
    switch ((SyntaxKind)type) {
        case SyntaxKind::UnionType: // Not strictly necessary, but a union containing a union should have been flattened
        case SyntaxKind::IntersectionType: // Not strictly necessary, but makes generated output more readable and avoids breaks in DT tests
            return factory->createParenthesizedType(type);
    }
    return parenthesizeCheckTypeOfConditionalType(type);
}

auto ParenthesizerRules::parenthesizeConstituentTypeOfIntersectionType(TypeNode type) -> TypeNode {
    switch ((SyntaxKind)type) {
        case SyntaxKind::UnionType:
        case SyntaxKind::IntersectionType: // Not strictly necessary, but an intersection containing an intersection should have been flattened
            return factory->createParenthesizedType(type);
    }
    return parenthesizeConstituentTypeOfUnionType(type);
}

auto ParenthesizerRules::parenthesizeOperandOfTypeOperator(TypeNode type) -> TypeNode {
    switch ((SyntaxKind)type) {
        case SyntaxKind::IntersectionType:
            return factory->createParenthesizedType(type);
    }
    return parenthesizeConstituentTypeOfIntersectionType(type);
}

auto ParenthesizerRules::parenthesizeNonArrayTypeOfPostfixType(TypeNode type) -> TypeNode {
    switch ((SyntaxKind)type) {
        case SyntaxKind::InferType:
        case SyntaxKind::TypeOperator:
        case SyntaxKind::TypeQuery: // Not strictly necessary, but makes generated output more readable and avoids breaks in DT tests
            return factory->createParenthesizedType(type);
    }
    return parenthesizeOperandOfTypeOperator(type);
}


} // namespace ts
