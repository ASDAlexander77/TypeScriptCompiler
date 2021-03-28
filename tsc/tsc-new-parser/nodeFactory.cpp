#include "nodeFactory.h"

auto NodeFactory::createNumericLiteral(string value, TokenFlags numericLiteralFlags) -> NumericLiteral
{
    auto node = createBaseLiteral<NumericLiteral>(SyntaxKind::NumericLiteral, value);
    node->numericLiteralFlags = numericLiteralFlags;
    if (!!(numericLiteralFlags & TokenFlags::BinaryOrOctalSpecifier)) node->transformFlags |= TransformFlags::ContainsES2015;
    return node;
}

auto NodeFactory::createBaseStringLiteral(string text, boolean isSingleQuote) -> StringLiteral 
{
    auto node = createBaseLiteral<StringLiteral>(SyntaxKind::StringLiteral, text);
    node->singleQuote = isSingleQuote;
    return node;
}

/* @internal*/ auto NodeFactory::createStringLiteral(string text, boolean isSingleQuote, boolean hasExtendedUnicodeEscape) -> StringLiteral // eslint-disable-line @typescript-eslint/unified-signatures
{
    auto node = createBaseStringLiteral(text, isSingleQuote);
    node->hasExtendedUnicodeEscape = hasExtendedUnicodeEscape;
    if (hasExtendedUnicodeEscape) node->transformFlags |= TransformFlags::ContainsES2015;
    return node;
}

auto NodeFactory::createBaseIdentifier(string text, SyntaxKind originalKeywordKind) 
{
    if (originalKeywordKind == SyntaxKind::Unknown && !text.empty()) {
        originalKeywordKind = scanner->stringToToken(text);
    }
    if (originalKeywordKind == SyntaxKind::Identifier) {
        originalKeywordKind = SyntaxKind::Unknown;
    }
    auto node = createBaseNode<Identifier>(SyntaxKind::Identifier);
    node->originalKeywordKind = originalKeywordKind;
    node->escapedText = escapeLeadingUnderscores(text);
    return node;
}

/* @internal */ auto NodeFactory::createIdentifier(string text, NodeArray</*TypeNode | TypeParameterDeclaration*/Node> typeArguments, SyntaxKind originalKeywordKind) -> Identifier // eslint-disable-line @typescript-eslint/unified-signatures
{
    auto node = createBaseIdentifier(text, originalKeywordKind);
    if (!!typeArguments) {
        // NOTE: we do not use `setChildren` here because typeArguments in an identifier do not contribute to transformations
        copy(node->typeArguments, createNodeArray(typeArguments));
    }
    if (node->originalKeywordKind == SyntaxKind::AwaitKeyword) {
        node->transformFlags |= TransformFlags::ContainsPossibleTopLevelAwait;
    }
    return node;
}

auto NodeFactory::createPrivateIdentifier(string text) -> PrivateIdentifier
{
    if (!startsWith(text, S("#"))) Debug::fail<void>(S("First character of private identifier must be #: ") + text);
    auto node = createBaseNode<PrivateIdentifier>(SyntaxKind::PrivateIdentifier);
    node->escapedText = escapeLeadingUnderscores(text);
    node->transformFlags |= TransformFlags::ContainsClassFields;
    return node;
}

auto NodeFactory::createToken(SyntaxKind token) -> Node {
    Debug::_assert(token >= SyntaxKind::FirstToken && token <= SyntaxKind::LastToken, S("Invalid token"));
    Debug::_assert(token <= SyntaxKind::FirstTemplateToken || token >= SyntaxKind::LastTemplateToken, S("Invalid token. Use 'createTemplateLiteralLikeNode' to create template literals."));
    Debug::_assert(token <= SyntaxKind::FirstLiteralToken || token >= SyntaxKind::LastLiteralToken, S("Invalid token. Use 'createLiteralLikeNode' to create literals."));
    Debug::_assert(token != SyntaxKind::Identifier, S("Invalid token. Use 'createIdentifier' to create identifiers"));
    //auto node = createBaseTokenNode<Token<TKind>>(token);
    auto node = createBaseNode<Node>(token);
    auto transformFlags = TransformFlags::None;
    switch (token) {
        case SyntaxKind::AsyncKeyword:
            // 'async' modifier is ES2017 (async functions) or ES2018 (async generators)
            transformFlags =
                TransformFlags::ContainsES2017 |
                TransformFlags::ContainsES2018;
            break;

        case SyntaxKind::PublicKeyword:
        case SyntaxKind::PrivateKeyword:
        case SyntaxKind::ProtectedKeyword:
        case SyntaxKind::ReadonlyKeyword:
        case SyntaxKind::AbstractKeyword:
        case SyntaxKind::DeclareKeyword:
        case SyntaxKind::ConstKeyword:
        case SyntaxKind::AnyKeyword:
        case SyntaxKind::NumberKeyword:
        case SyntaxKind::BigIntKeyword:
        case SyntaxKind::NeverKeyword:
        case SyntaxKind::ObjectKeyword:
        case SyntaxKind::StringKeyword:
        case SyntaxKind::BooleanKeyword:
        case SyntaxKind::SymbolKeyword:
        case SyntaxKind::VoidKeyword:
        case SyntaxKind::UnknownKeyword:
        case SyntaxKind::UndefinedKeyword: // `undefined` is an Identifier in the expression case.
            transformFlags = TransformFlags::ContainsTypeScript;
            break;
        case SyntaxKind::StaticKeyword:
        case SyntaxKind::SuperKeyword:
            transformFlags = TransformFlags::ContainsES2015;
            break;
        case SyntaxKind::ThisKeyword:
            // 'this' indicates a lexical 'this'
            transformFlags = TransformFlags::ContainsLexicalThis;
            break;
    }
    if (!!transformFlags) {
        node->transformFlags |= transformFlags;
    }
    return node;
}

auto NodeFactory::createQualifiedName(EntityName left, Identifier right) -> Identifier {
    auto node = createBaseNode<QualifiedName>(SyntaxKind::QualifiedName);
    node->left = left;
    node->right = asName(right);
    node->transformFlags |=
        propagateChildFlags(node->left) |
        propagateIdentifierNameFlags(node->right);
    return node;
}
