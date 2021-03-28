#include "nodeFactory.h"

auto NodeFactory::createNumericLiteral(string value, TokenFlags numericLiteralFlags) -> NumericLiteral
{
    auto node = createBaseLiteral<NumericLiteral>(SyntaxKind::NumericLiteral, value);
    node->numericLiteralFlags = numericLiteralFlags;
    if (numericLiteralFlags & TokenFlags::BinaryOrOctalSpecifier) node->transformFlags |= TransformFlags::ContainsES2015;
    return node;
}