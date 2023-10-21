#include "node_factory.h"

namespace ts
{
auto NodeFactory::NoParenthesizerRules() -> boolean
{
    return !!(flags & NodeFactoryFlags::NoParenthesizerRules);
}

auto NodeFactory::propagateIdentifierNameFlags(Identifier node) -> TransformFlags
{
    // An IdentifierName is allowed to be `await`
    return propagateChildFlags(node) & ~TransformFlags::ContainsPossibleTopLevelAwait;
}

auto NodeFactory::propagateNameFlags(Node node) -> TransformFlags {
    return !!node && isIdentifier(node) ? propagateIdentifierNameFlags(node) : propagateChildFlags(node);
}

auto NodeFactory::propagatePropertyNameFlagsOfChild(PropertyName node, TransformFlags transformFlags) -> TransformFlags
{
    return transformFlags | (node->transformFlags & TransformFlags::PropertyNamePropagatingFlags);
}

auto NodeFactory::propagateChildFlags(Node child) -> TransformFlags
{
    if (!child)
        return TransformFlags::None;
    auto childFlags = child->transformFlags & ~getTransformFlagsSubtreeExclusions(child);
    return isNamedDeclaration(child) && isPropertyName(child.as<NamedDeclaration>()->name)
               ? propagatePropertyNameFlagsOfChild(child.as<NamedDeclaration>()->name, childFlags)
               : childFlags;
}

auto propagateChildrenFlags(NodeArray<Node> children) -> TransformFlags {
    return children ? children.transformFlags : TransformFlags::None;
}

auto NodeFactory::propagateAssignmentPatternFlags(AssignmentPattern node) -> TransformFlags
{
    if (!!(node->transformFlags & TransformFlags::ContainsObjectRestOrSpread))
        return TransformFlags::ContainsObjectRestOrSpread;
    if (!!(node->transformFlags & TransformFlags::ContainsES2018))
    {
        // check for nested spread assignments, otherwise '{ x: { a, ...b } = foo } = c'
        // will not be correctly interpreted by the ES2018 transformer
        for (auto element : getElementsOfBindingOrAssignmentPattern(node))
        {
            auto target = getTargetOfBindingOrAssignmentElement(element);
            if (!!(target && isAssignmentPattern(target)))
            {
                if (!!(target->transformFlags & TransformFlags::ContainsObjectRestOrSpread))
                {
                    return TransformFlags::ContainsObjectRestOrSpread;
                }
                if (!!(target->transformFlags & TransformFlags::ContainsES2018))
                {
                    auto flags = propagateAssignmentPatternFlags(target);
                    if (!!flags)
                        return flags;
                }
            }
        }
    }
    return TransformFlags::None;
}

auto NodeFactory::getTransformFlagsSubtreeExclusions(SyntaxKind kind) -> TransformFlags
{
    if (kind >= SyntaxKind::FirstTypeNode && kind <= SyntaxKind::LastTypeNode)
    {
        return TransformFlags::TypeExcludes;
    }

    switch (kind)
    {
    case SyntaxKind::CallExpression:
    case SyntaxKind::NewExpression:
    case SyntaxKind::ArrayLiteralExpression:
        return TransformFlags::ArrayLiteralOrCallOrNewExcludes;
    case SyntaxKind::ModuleDeclaration:
        return TransformFlags::ModuleExcludes;
    case SyntaxKind::Parameter:
        return TransformFlags::ParameterExcludes;
    case SyntaxKind::ArrowFunction:
        return TransformFlags::ArrowFunctionExcludes;
    case SyntaxKind::FunctionExpression:
    case SyntaxKind::FunctionDeclaration:
        return TransformFlags::FunctionExcludes;
    case SyntaxKind::VariableDeclarationList:
        return TransformFlags::VariableDeclarationListExcludes;
    case SyntaxKind::ClassDeclaration:
    case SyntaxKind::ClassExpression:
        return TransformFlags::ClassExcludes;
    case SyntaxKind::Constructor:
        return TransformFlags::ConstructorExcludes;
    case SyntaxKind::PropertyDeclaration:
        return TransformFlags::PropertyExcludes;
    case SyntaxKind::MethodDeclaration:
    case SyntaxKind::GetAccessor:
    case SyntaxKind::SetAccessor:
        return TransformFlags::MethodOrAccessorExcludes;
    case SyntaxKind::AnyKeyword:
    case SyntaxKind::NumberKeyword:
    case SyntaxKind::BigIntKeyword:
    case SyntaxKind::NeverKeyword:
    case SyntaxKind::StringKeyword:
    case SyntaxKind::ObjectKeyword:
    case SyntaxKind::BooleanKeyword:
    case SyntaxKind::SymbolKeyword:
    case SyntaxKind::VoidKeyword:
    case SyntaxKind::TypeParameter:
    case SyntaxKind::PropertySignature:
    case SyntaxKind::MethodSignature:
    case SyntaxKind::CallSignature:
    case SyntaxKind::ConstructSignature:
    case SyntaxKind::IndexSignature:
    case SyntaxKind::InterfaceDeclaration:
    case SyntaxKind::TypeAliasDeclaration:
        return TransformFlags::TypeExcludes;
    case SyntaxKind::ObjectLiteralExpression:
        return TransformFlags::ObjectLiteralExcludes;
    case SyntaxKind::CatchClause:
        return TransformFlags::CatchClauseExcludes;
    case SyntaxKind::ObjectBindingPattern:
    case SyntaxKind::ArrayBindingPattern:
        return TransformFlags::BindingPatternExcludes;
    case SyntaxKind::TypeAssertionExpression:
    case SyntaxKind::AsExpression:
    case SyntaxKind::PartiallyEmittedExpression:
    case SyntaxKind::ParenthesizedExpression:
    case SyntaxKind::SuperKeyword:
        return TransformFlags::OuterExpressionExcludes;
    case SyntaxKind::PropertyAccessExpression:
    case SyntaxKind::ElementAccessExpression:
        return TransformFlags::PropertyAccessExcludes;
    default:
        return TransformFlags::NodeExcludes;
    }
}

auto NodeFactory::createNumericLiteral(string value, TokenFlags numericLiteralFlags) -> NumericLiteral
{
    auto node = createBaseLiteral<NumericLiteral>(SyntaxKind::NumericLiteral, value);
    node->numericLiteralFlags = numericLiteralFlags;
    if (!!(numericLiteralFlags & TokenFlags::BinaryOrOctalSpecifier))
        node->transformFlags |= TransformFlags::ContainsES2015;
    return node;
}

auto NodeFactory::createBaseStringLiteral(string text, boolean isSingleQuote) -> StringLiteral
{
    auto node = createBaseLiteral<StringLiteral>(SyntaxKind::StringLiteral, text);
    node->singleQuote = isSingleQuote;
    return node;
}

/* @internal*/ auto NodeFactory::createStringLiteral(string text, boolean isSingleQuote, boolean hasExtendedUnicodeEscape)
    -> StringLiteral // eslint-disable-line @typescript-eslint/unified-signatures
{
    auto node = createBaseStringLiteral(text, isSingleQuote);
    node->hasExtendedUnicodeEscape = hasExtendedUnicodeEscape;
    if (hasExtendedUnicodeEscape)
        node->transformFlags |= TransformFlags::ContainsES2015;
    return node;
}

auto NodeFactory::createBigIntLiteral(string value) -> BigIntLiteral
{
    auto node = createBaseLiteral<BigIntLiteral>(SyntaxKind::BigIntLiteral, value);
    node->transformFlags |= TransformFlags::ContainsESNext;
    return node;
}

auto NodeFactory::createRegularExpressionLiteral(string text) -> RegularExpressionLiteral
{
    auto node = createBaseLiteral<RegularExpressionLiteral>(SyntaxKind::RegularExpressionLiteral, text);
    return node;
}

auto NodeFactory::createLiteralLikeNode(SyntaxKind kind, string text) -> LiteralLikeNode
{
    switch (kind)
    {
    case SyntaxKind::NumericLiteral:
        return createNumericLiteral(text);
    case SyntaxKind::BigIntLiteral:
        return createBigIntLiteral(text);
    case SyntaxKind::StringLiteral:
        return createStringLiteral(text);
    case SyntaxKind::JsxText:
        return createJsxText(text);
    case SyntaxKind::JsxTextAllWhiteSpaces:
        return createJsxText(text, /*containsOnlyTriviaWhiteSpaces*/ true);
    case SyntaxKind::RegularExpressionLiteral:
        return createRegularExpressionLiteral(text);
    case SyntaxKind::NoSubstitutionTemplateLiteral:
        return createTemplateLiteralLikeNode(kind, text);
    }

    return undefined;
}

auto NodeFactory::createBaseIdentifier(string text)
{
    auto node = createBaseNode<Identifier>(SyntaxKind::Identifier);
    node->escapedText = text;
    return node;
}

/* @internal */ auto NodeFactory::createIdentifier(string text, SyntaxKind originalKeywordKind, boolean hasExtendedUnicodeEscape)
    -> Identifier // eslint-disable-line @typescript-eslint/unified-signatures
{
    if (originalKeywordKind == SyntaxKind::Unknown && text.length() > 0) {
        originalKeywordKind = scanner->stringToToken(text);
    }
    if (originalKeywordKind == SyntaxKind::Identifier) {
        originalKeywordKind = SyntaxKind::Unknown;
    }

    auto node = createBaseIdentifier(escapeLeadingUnderscores(text));
    if (hasExtendedUnicodeEscape) node->flags |= NodeFlags::IdentifierHasExtendedUnicodeEscape;

    // we NOTE do not include transform flags typeArguments  in an identifier as they do not contribute to transformations
    if (node->escapedText == S("await")) {
        node->transformFlags |= TransformFlags::ContainsPossibleTopLevelAwait;
    }
    if ((node->flags & NodeFlags::IdentifierHasExtendedUnicodeEscape) > NodeFlags::None) {
        node->transformFlags |= TransformFlags::ContainsES2015;
    }

    return node;
}

auto NodeFactory::createPrivateIdentifier(string text) -> PrivateIdentifier
{
    if (!startsWith(text, S("#")))
        Debug::fail<void>(S("First character of private identifier must be #: ") + text);
    auto node = createBaseNode<PrivateIdentifier>(SyntaxKind::PrivateIdentifier);
    node->escapedText = escapeLeadingUnderscores(text);
    node->transformFlags |= TransformFlags::ContainsClassFields;
    return node;
}

auto NodeFactory::createQualifiedName(EntityName left, Identifier right) -> QualifiedName
{
    auto node = createBaseNode<QualifiedName>(SyntaxKind::QualifiedName);
    node->left = left;
    node->right = asName(right);
    node->transformFlags |= propagateChildFlags(node->left) | propagateIdentifierNameFlags(node->right);
    return node;
}

auto NodeFactory::createComputedPropertyName(Expression expression) -> ComputedPropertyName
{
    auto node = createBaseNode<ComputedPropertyName>(SyntaxKind::ComputedPropertyName);
    node->expression = parenthesizerRules.parenthesizeExpressionOfComputedPropertyName(expression);
    node->transformFlags |=
        propagateChildFlags(node->expression) | TransformFlags::ContainsES2015 | TransformFlags::ContainsComputedPropertyName;
    return node;
}

auto NodeFactory::createTypeParameterDeclaration(NodeArray<ModifierLike> modifiers, Identifier name, TypeNode constraint, TypeNode defaultType) -> TypeParameterDeclaration
{
    auto node = createBaseDeclaration<TypeParameterDeclaration>(SyntaxKind::TypeParameter);
    node->modifiers = asNodeArray(modifiers);
    node->name = asName(name);
    node->constraint = constraint;
    node->_default = defaultType;
    node->transformFlags = TransformFlags::ContainsTypeScript;

    node->expression = undefined; // initialized by parser to report grammar errors
    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

auto NodeFactory::createParameterDeclaration(NodeArray<ModifierLike> modifiers, DotDotDotToken dotDotDotToken,
                                             BindingName name, QuestionToken questionToken, TypeNode type, Expression initializer)
    -> ParameterDeclaration
{
    auto node = createBaseDeclaration<ParameterDeclaration>(SyntaxKind::Parameter);
    node->modifiers = asNodeArray(modifiers);
    node->dotDotDotToken = dotDotDotToken;
    node->name = asName(name);
    node->questionToken = questionToken;
    node->type = type;
    node->initializer = asInitializer(initializer);
    if (isThisIdentifier(node->name))
    {
        node->transformFlags = TransformFlags::ContainsTypeScript;
    }
    else
    {
        node->transformFlags |= propagateChildrenFlags(node->modifiers) |
            propagateChildFlags(node->dotDotDotToken) | 
            propagateNameFlags(node->name) |
            propagateChildFlags(node->questionToken) |
            propagateChildFlags(node->initializer);
        if (questionToken || node->type)
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        if (!!initializer || !!dotDotDotToken)
            node->transformFlags |= TransformFlags::ContainsES2015;
        if (!!(modifiersToFlags(node->modifiers) & ModifierFlags::ParameterPropertyModifier))
            node->transformFlags |= TransformFlags::ContainsTypeScriptClassSyntax;
    }

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

// @api
auto NodeFactory::createDecorator(Expression expression) -> Decorator
{
    auto node = createBaseNode<Decorator>(SyntaxKind::Decorator);
    node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
    node->transformFlags |=
        propagateChildFlags(node->expression) | TransformFlags::ContainsTypeScript | TransformFlags::ContainsTypeScriptClassSyntax;
    return node;
}

//
// Type Elements
//

// @api
auto NodeFactory::createPropertySignature(ModifiersArray modifiers, PropertyName name, QuestionToken questionToken, TypeNode type)
    -> PropertySignature
{
    auto node = createBaseDeclaration<PropertySignature>(SyntaxKind::PropertySignature);
    node->modifiers = asNodeArray(modifiers);
    node->name = asName(name);    
    node->type = type;
    node->questionToken = questionToken;
    node->transformFlags = TransformFlags::ContainsTypeScript;

    node->initializer = undefined; // initialized by parser to report grammar errors
    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)    
    return node;
}

// @api

// @api
auto NodeFactory::createPropertyDeclaration(NodeArray<ModifierLike> modifiers, PropertyName name,
                                            Node questionOrExclamationToken, TypeNode type, Expression initializer) -> PropertyDeclaration
{
    auto node = createBaseDeclaration<PropertyDeclaration>(SyntaxKind::PropertyDeclaration);
    node->modifiers = asNodeArray(modifiers);
    node->name = asName(name);    
    node->questionToken =
        questionOrExclamationToken && isQuestionToken(questionOrExclamationToken) ? questionOrExclamationToken : undefined;
    node->exclamationToken =
        questionOrExclamationToken && isExclamationToken(questionOrExclamationToken) ? questionOrExclamationToken : undefined;
    node->type = type;
    node->initializer = asInitializer(initializer);        

    auto isAmbient = !!(node->flags & NodeFlags::Ambient) || !!(modifiersToFlags(node->modifiers) & ModifierFlags::Ambient);

    node->transformFlags |= propagateChildrenFlags(node->modifiers) |
        propagateNameFlags(node->name) |
        propagateChildFlags(node->initializer) |
        ((isAmbient || !!node->questionToken || !!node->exclamationToken || !!node->type) ? TransformFlags::ContainsClassFields : TransformFlags::None);
    if (isComputedPropertyName(node->name) || (!!(modifiersToFlags(node->modifiers) & ModifierFlags::Static) && node->initializer))
    {
        node->transformFlags |= TransformFlags::ContainsTypeScriptClassSyntax;
    }

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

// @api

// @api
auto NodeFactory::createMethodSignature(ModifiersArray modifiers, PropertyName name, QuestionToken questionToken,
                                        NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                        TypeNode type) -> MethodSignature
{
    auto node = createBaseDeclaration<MethodSignature>(SyntaxKind::MethodSignature);
    node->modifiers = asNodeArray(modifiers);
    node->name = asName(name);
    node->questionToken = questionToken;
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = asNodeArray(parameters);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;

    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    //node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    node->typeArguments = undefined; // used in quick info
    return node;
}

// @api

// @api
auto NodeFactory::createMethodDeclaration(NodeArray<ModifierLike> modifiers, AsteriskToken asteriskToken,
                                          PropertyName name, QuestionToken questionToken,
                                          NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                          TypeNode type, Block body) -> MethodDeclaration
{
    auto node = createBaseDeclaration<MethodDeclaration>(SyntaxKind::MethodDeclaration);
    node->modifiers = asNodeArray(modifiers);
    node->asteriskToken = asteriskToken;
    node->name = asName(name);
    node->questionToken = questionToken;
    node->exclamationToken = undefined; // initialized by parser for grammar errors
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = createNodeArray(parameters);
    node->type = type;
    node->body = body;

    if (!node->body) {
        node->transformFlags = TransformFlags::ContainsTypeScript;
    }
    else {
        auto isAsync = !!(modifiersToFlags(node->modifiers) & ModifierFlags::Async);
        auto isGenerator = !!node->asteriskToken;
        auto isAsyncGenerator = isAsync && isGenerator;

        node->transformFlags = propagateChildrenFlags(node->modifiers) |
            propagateChildFlags(node->asteriskToken) |
            propagateNameFlags(node->name) |
            propagateChildFlags(node->questionToken) |
            propagateChildrenFlags(node->typeParameters) |
            propagateChildrenFlags(node->parameters) |
            propagateChildFlags(node->type) |
            (propagateChildFlags(node->body) & ~TransformFlags::ContainsPossibleTopLevelAwait) |
            (isAsyncGenerator ? TransformFlags::ContainsES2018 :
                isAsync ? TransformFlags::ContainsES2017 :
                isGenerator ? TransformFlags::ContainsGenerator :
                TransformFlags::None) |
            (node->questionToken || node->typeParameters || node->type ? TransformFlags::ContainsTypeScript : TransformFlags::None) |
            TransformFlags::ContainsES2015;
    }

    node->typeArguments = undefined; // used in quick info
    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    //node->endFlowNode = undefined;
    //node->returnFlowNode = undefined;

    return node;
}

// @api

auto NodeFactory::createClassStaticBlockDeclaration(Block body) -> ClassStaticBlockDeclaration {
    auto node = createBaseDeclaration<ClassStaticBlockDeclaration>(SyntaxKind::ClassStaticBlockDeclaration);
    node->body = body;
    node->transformFlags = propagateChildFlags(body) | TransformFlags::ContainsClassFields;

    node->modifiers = undefined; // initialized by parser for grammar errors
    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    //node->endFlowNode = undefined;
    //node->returnFlowNode = undefined;
    return node;
}



// @api
auto NodeFactory::createConstructorDeclaration(NodeArray<ModifierLike> modifiers,
                                               NodeArray<ParameterDeclaration> parameters, Block body) -> ConstructorDeclaration
{
    auto node = createBaseDeclaration<ConstructorDeclaration>(SyntaxKind::Constructor);
    node->modifiers = asNodeArray(modifiers);
    node->parameters = createNodeArray(parameters);
    node->body = body;

    node->transformFlags = propagateChildrenFlags(node->modifiers) |
        propagateChildrenFlags(node->parameters) |
        (propagateChildFlags(node->body) & ~TransformFlags::ContainsPossibleTopLevelAwait) |
        TransformFlags::ContainsES2015;

    node->typeParameters = undefined; // initialized by parser for grammar errors
    node->type = undefined; // initialized by parser for grammar errors
    node->typeArguments = undefined; // used in quick info
    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    //node->endFlowNode = undefined;
    //node->returnFlowNode = undefined;
    return node;
}

// @api

// @api
auto NodeFactory::createGetAccessorDeclaration(NodeArray<ModifierLike> modifiers, PropertyName name,
                                               NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body)
    -> GetAccessorDeclaration
{
    auto node = createBaseDeclaration<GetAccessorDeclaration>(SyntaxKind::GetAccessor);
    node->modifiers = asNodeArray(modifiers);
    node->name = asName(name);
    node->parameters = createNodeArray(parameters);
    node->type = type;
    node->body = body;

    if (!node->body) {
        node->transformFlags = TransformFlags::ContainsTypeScript;
    }
    else {
        node->transformFlags = propagateChildrenFlags(node->modifiers) |
            propagateNameFlags(node->name) |
            propagateChildrenFlags(node->parameters) |
            propagateChildFlags(node->type) |
            (propagateChildFlags(node->body) & ~TransformFlags::ContainsPossibleTopLevelAwait) |
            (node->type ? TransformFlags::ContainsTypeScript : TransformFlags::None);
    }

    node->typeArguments = undefined; // used in quick info
    node->typeParameters = undefined; // initialized by parser for grammar errors
    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    //node->endFlowNode = undefined;
    //node->returnFlowNode = undefined;
    return node;
}

// @api

// @api
auto NodeFactory::createSetAccessorDeclaration(NodeArray<ModifierLike> modifiers, PropertyName name,
                                               NodeArray<ParameterDeclaration> parameters, Block body) -> SetAccessorDeclaration
{
    auto node = createBaseDeclaration<SetAccessorDeclaration>(SyntaxKind::SetAccessor);
    node->modifiers = asNodeArray(modifiers);
    node->name = asName(name);
    node->parameters = createNodeArray(parameters);
    node->body = body;

    if (!node->body) {
        node->transformFlags = TransformFlags::ContainsTypeScript;
    }
    else {
        node->transformFlags = propagateChildrenFlags(node->modifiers) |
            propagateNameFlags(node->name) |
            propagateChildrenFlags(node->parameters) |
            (propagateChildFlags(node->body) & ~TransformFlags::ContainsPossibleTopLevelAwait) |
            (node->type ? TransformFlags::ContainsTypeScript : TransformFlags::None);
    }

    node->typeArguments = undefined; // used in quick info
    node->typeParameters = undefined; // initialized by parser for grammar errors
    node->type = undefined; // initialized by parser for grammar errors
    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    //node->endFlowNode = undefined;
    //node->returnFlowNode = undefined;
    return node;
}

// @api

// @api
auto NodeFactory::createCallSignature(NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                      TypeNode type) -> CallSignatureDeclaration
{
    auto node = createBaseDeclaration<CallSignatureDeclaration>(SyntaxKind::CallSignature);
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = asNodeArray(parameters);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    node->typeArguments = undefined; // used in quick info
    return node;
}

// @api

// @api
auto NodeFactory::createConstructSignature(NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                           TypeNode type) -> ConstructSignatureDeclaration
{
    auto node = createBaseDeclaration<ConstructSignatureDeclaration>(SyntaxKind::ConstructSignature);
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = asNodeArray(parameters);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    node->typeArguments = undefined; // used in quick info
    return node;
}

// @api

// @api
auto NodeFactory::createIndexSignature(NodeArray<ModifierLike> modifiers, NodeArray<ParameterDeclaration> parameters,
                                       TypeNode type) -> IndexSignatureDeclaration
{
    auto node = createBaseDeclaration<IndexSignatureDeclaration>(SyntaxKind::IndexSignature);
    node->modifiers = asNodeArray(modifiers);
    node->parameters = asNodeArray(parameters);
    node->type = type; // TODO(rbuckton): We mark this as required in IndexSignatureDeclaration, but it looks like the parser allows it to be elided.
    node->transformFlags = TransformFlags::ContainsTypeScript;

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    node->typeArguments = undefined; // used in quick info
    return node;
}

// @api

// @api
auto NodeFactory::createTemplateLiteralTypeSpan(TypeNode type, Node literal) -> TemplateLiteralTypeSpan
{
    auto node = createBaseNode<TemplateLiteralTypeSpan>(SyntaxKind::TemplateLiteralTypeSpan);
    node->type = type;
    node->literal = literal;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

//
// Types
//

// @api
auto NodeFactory::createTypePredicateNode(AssertsKeyword assertsModifier, Node parameterName, TypeNode type) -> TypePredicateNode
{
    auto node = createBaseNode<TypePredicateNode>(SyntaxKind::TypePredicate);
    node->assertsModifier = assertsModifier;
    node->parameterName = asName(parameterName);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createTypeReferenceNode(EntityName typeName, NodeArray<TypeNode> typeArguments) -> TypeReferenceNode
{
    auto node = createBaseNode<TypeReferenceNode>(SyntaxKind::TypeReference);
    node->typeName = asName(typeName);
    node->typeArguments = typeArguments ? parenthesizerRules.parenthesizeTypeArguments(createNodeArray(typeArguments)) : undefined;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createFunctionTypeNode(NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                         TypeNode type) -> FunctionTypeNode
{
    auto node = createBaseDeclaration<FunctionTypeNode>(SyntaxKind::FunctionType);
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = asNodeArray(parameters);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;

    node->modifiers = undefined; // initialized by parser for grammar errors
    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    //node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    node->typeArguments = undefined; // used in quick info
    return node;
}

// @api
auto NodeFactory::createConstructorTypeNode(ModifiersArray modifiers, NodeArray<TypeParameterDeclaration> typeParameters,
                                            NodeArray<ParameterDeclaration> parameters, TypeNode type) -> ConstructorTypeNode
{
    auto node = createBaseDeclaration<ConstructorTypeNode>(SyntaxKind::ConstructorType);
    node->modifiers = asNodeArray(modifiers);
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = asNodeArray(parameters);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    node->typeArguments = undefined; // used in quick info
    return node;
}

// @api
auto NodeFactory::createTypeQueryNode(EntityName exprName, NodeArray<TypeNode> typeArguments) -> TypeQueryNode
{
    auto node = createBaseNode<TypeQueryNode>(SyntaxKind::TypeQuery);
    node->exprName = exprName;
    node->typeArguments = !!typeArguments ? parenthesizerRules.parenthesizeTypeArguments(typeArguments) : typeArguments;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createTypeLiteralNode(NodeArray<TypeElement> members) -> TypeLiteralNode
{
    auto node = createBaseNode<TypeLiteralNode>(SyntaxKind::TypeLiteral);
    node->members = createNodeArray(members);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createArrayTypeNode(TypeNode elementType) -> ArrayTypeNode
{
    auto node = createBaseNode<ArrayTypeNode>(SyntaxKind::ArrayType);
    node->elementType = parenthesizerRules.parenthesizeElementTypeOfArrayType(elementType);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createTupleTypeNode(NodeArray</*TypeNode | NamedTupleMember*/ Node> elements) -> TupleTypeNode
{
    auto node = createBaseNode<TupleTypeNode>(SyntaxKind::TupleType);
    node->elements = createNodeArray(elements);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createNamedTupleMember(DotDotDotToken dotDotDotToken, Identifier name, QuestionToken questionToken, TypeNode type)
    -> NamedTupleMember
{
    auto node = createBaseNode<NamedTupleMember>(SyntaxKind::NamedTupleMember);
    node->dotDotDotToken = dotDotDotToken;
    node->name = name;
    node->questionToken = questionToken;
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createOptionalTypeNode(TypeNode type) -> OptionalTypeNode
{
    auto node = createBaseNode<OptionalTypeNode>(SyntaxKind::OptionalType);
    node->type = parenthesizerRules.parenthesizeElementTypeOfArrayType(type);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createRestTypeNode(TypeNode type) -> RestTypeNode
{
    auto node = createBaseNode<RestTypeNode>(SyntaxKind::RestType);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createUnionTypeNode(NodeArray<TypeNode> types) -> UnionTypeNode
{
    auto node = createBaseNode<UnionTypeNode>(SyntaxKind::UnionType);
    node->types = parenthesizerRules.parenthesizeConstituentTypesOfUnionOrIntersectionType(types);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createIntersectionTypeNode(NodeArray<TypeNode> types) -> IntersectionTypeNode
{
    auto node = createBaseNode<IntersectionTypeNode>(SyntaxKind::IntersectionType);
    node->types = parenthesizerRules.parenthesizeConstituentTypesOfUnionOrIntersectionType(types);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createConditionalTypeNode(TypeNode checkType, TypeNode extendsType, TypeNode trueType, TypeNode falseType)
    -> ConditionalTypeNode
{
    auto node = createBaseNode<ConditionalTypeNode>(SyntaxKind::ConditionalType);
    node->checkType = parenthesizerRules.parenthesizeMemberOfConditionalType(checkType);
    node->extendsType = parenthesizerRules.parenthesizeMemberOfConditionalType(extendsType);
    node->trueType = trueType;
    node->falseType = falseType;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createInferTypeNode(TypeParameterDeclaration typeParameter) -> InferTypeNode
{
    auto node = createBaseNode<InferTypeNode>(SyntaxKind::InferType);
    node->typeParameter = typeParameter;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createTemplateLiteralType(TemplateHead head, NodeArray<TemplateLiteralTypeSpan> templateSpans) -> TemplateLiteralTypeNode
{
    auto node = createBaseNode<TemplateLiteralTypeNode>(SyntaxKind::TemplateLiteralType);
    node->head = head;
    node->templateSpans = createNodeArray(templateSpans);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createImportTypeNode(TypeNode argument, ImportAttributes attributes, EntityName qualifier, NodeArray<TypeNode> typeArguments, boolean isTypeOf)
    -> ImportTypeNode
{
    auto node = createBaseNode<ImportTypeNode>(SyntaxKind::ImportType);
    node->argument = argument;
    node->attributes = attributes;
    node->qualifier = qualifier;
    node->typeArguments = typeArguments ? parenthesizerRules.parenthesizeTypeArguments(typeArguments) : undefined;
    node->isTypeOf = isTypeOf;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createParenthesizedType(TypeNode type) -> ParenthesizedTypeNode
{
    auto node = createBaseNode<ParenthesizedTypeNode>(SyntaxKind::ParenthesizedType);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createThisTypeNode() -> ThisTypeNode
{
    auto node = createBaseNode<ThisTypeNode>(SyntaxKind::ThisType);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api
auto NodeFactory::createTypeOperatorNode(SyntaxKind _operator, TypeNode type) -> TypeOperatorNode
{
    auto node = createBaseNode<TypeOperatorNode>(SyntaxKind::TypeOperator);
    node->_operator = _operator;
    node->type = parenthesizerRules.parenthesizeMemberOfElementType(type);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createIndexedAccessTypeNode(TypeNode objectType, TypeNode indexType) -> IndexedAccessTypeNode
{
    auto node = createBaseNode<IndexedAccessTypeNode>(SyntaxKind::IndexedAccessType);
    node->objectType = parenthesizerRules.parenthesizeMemberOfElementType(objectType);
    node->indexType = indexType;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createMappedTypeNode(Node readonlyToken, TypeParameterDeclaration typeParameter, TypeNode nameType, Node questionToken,
                                       TypeNode type, NodeArray<TypeElement> members) -> MappedTypeNode
{
    auto node = createBaseNode<MappedTypeNode>(SyntaxKind::MappedType);
    node->readonlyToken = readonlyToken;
    node->typeParameter = typeParameter;
    node->nameType = nameType;
    node->questionToken = questionToken;
    node->type = type;
    node->members = members ? members : createNodeArray(members);
    node->transformFlags = TransformFlags::ContainsTypeScript;

    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)    
    return node;
}

// @api

// @api
auto NodeFactory::createLiteralTypeNode(LiteralTypeNode literal) -> LiteralTypeNode
{
    auto node = createBaseNode<LiteralTypeNode>(SyntaxKind::LiteralType);
    node->literal = literal;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

//
// Binding Patterns
//

// @api
auto NodeFactory::createObjectBindingPattern(NodeArray<BindingElement> elements) -> ObjectBindingPattern
{
    auto node = createBaseNode<ObjectBindingPattern>(SyntaxKind::ObjectBindingPattern);
    node->elements = createNodeArray(elements);
    node->transformFlags |=
        propagateChildrenFlags(node->elements) | TransformFlags::ContainsES2015 | TransformFlags::ContainsBindingPattern;
    if (!!(node->transformFlags & TransformFlags::ContainsRestOrSpread))
    {
        node->transformFlags |= TransformFlags::ContainsES2018 | TransformFlags::ContainsObjectRestOrSpread;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createArrayBindingPattern(NodeArray<ArrayBindingElement> elements) -> ArrayBindingPattern
{
    auto node = createBaseNode<ArrayBindingPattern>(SyntaxKind::ArrayBindingPattern);
    node->elements = createNodeArray(elements);
    node->transformFlags |=
        propagateChildrenFlags(node->elements) | TransformFlags::ContainsES2015 | TransformFlags::ContainsBindingPattern;
    return node;
}

// @api

// @api
auto NodeFactory::createBindingElement(DotDotDotToken dotDotDotToken, PropertyName propertyName, BindingName name, Expression initializer)
    -> BindingElement
{
    auto node = createBaseDeclaration<BindingElement>(SyntaxKind::BindingElement);
    node->dotDotDotToken = dotDotDotToken;
    node->propertyName = asName(propertyName);
    node->name = asName(name);
    node->initializer = asInitializer(initializer);
    node->transformFlags |= propagateChildFlags(node->dotDotDotToken) |
        propagateNameFlags(node->propertyName) |
        propagateNameFlags(node->name) |
        propagateChildFlags(node->initializer) |
        (node->dotDotDotToken ? TransformFlags::ContainsRestOrSpread : TransformFlags::None) |
        TransformFlags::ContainsES2015;

    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    return node;
}

// @api

//
// Expression
//

// @api
auto NodeFactory::createArrayLiteralExpression(NodeArray<Expression> elements, boolean multiLine) -> ArrayLiteralExpression
{
    auto node = createBaseNode<ArrayLiteralExpression>(SyntaxKind::ArrayLiteralExpression);
    // Ensure we add a trailing comma for something like `[NumericLiteral(1), NumericLiteral(2), OmittedExpresion]` so that
    // we end up with `[1, 2, ,]` instead : `[1, 2, ]` otherwise the `OmittedExpression` will just end up being treated like
    // a trailing comma.
    auto lastElement = lastOrUndefined(elements);
    auto elementsArray = createNodeArray(elements, lastElement && isOmittedExpression(lastElement) ? true : undefined);
    node->elements = parenthesizerRules.parenthesizeExpressionsOfCommaDelimitedList(elementsArray);
    node->multiLine = multiLine;
    node->transformFlags |= propagateChildrenFlags(node->elements);
    return node;
}

// @api

// @api
auto NodeFactory::createObjectLiteralExpression(NodeArray<ObjectLiteralElementLike> properties, boolean multiLine)
    -> ObjectLiteralExpression
{
    auto node = createBaseDeclaration<ObjectLiteralExpression>(SyntaxKind::ObjectLiteralExpression);
    node->properties = createNodeArray(properties);
    node->multiLine = multiLine;
    node->transformFlags |= propagateChildrenFlags(node->properties);

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

// @api
auto NodeFactory::createBasePropertyAccessExpression(LeftHandSideExpression expression, QuestionDotToken questionDotToken, MemberName name) -> PropertyAccessExpression {
    auto node = createBaseDeclaration<PropertyAccessExpression>(SyntaxKind::PropertyAccessExpression);
    node->expression = expression;
    node->questionDotToken = questionDotToken;
    node->name = name;
    node->transformFlags = propagateChildFlags(node->expression) |
        propagateChildFlags(node->questionDotToken) |
        (isIdentifier(node->name) ?
            propagateIdentifierNameFlags(node->name) : propagateChildFlags(node->name) | TransformFlags::ContainsPrivateIdentifierInExpression);

    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    return node;
}

// @api
auto NodeFactory::createPropertyAccessExpression(Expression expression, MemberName name) -> PropertyAccessExpression
{
    auto node = createBasePropertyAccessExpression(
        parenthesizerRules.parenthesizeLeftSideOfAccess(expression, /*optionalChain*/ false),
        /*questionDotToken*/ undefined,
        asName(name)
    );
    if (isSuperKeyword(expression))
    {
        // super method calls require a lexical 'this'
        // super method calls require 'super' hoisting in ES2017 and ES2018 async functions and async generators
        node->transformFlags |= TransformFlags::ContainsES2017 | TransformFlags::ContainsES2018;
    }
    return node;
}

// @api
auto NodeFactory::createPropertyAccessChain(Expression expression, QuestionDotToken questionDotToken, MemberName name)
    -> PropertyAccessChain
{
    auto node = createBasePropertyAccessExpression(
        parenthesizerRules.parenthesizeLeftSideOfAccess(expression, /*optionalChain*/ true),
        questionDotToken,
        asName(name)
    ).as<PropertyAccessChain>();
    node->flags |= NodeFlags::OptionalChain;
    node->transformFlags |= TransformFlags::ContainsES2020;
    return node;
}

// @api
auto NodeFactory::createBaseElementAccessExpression(LeftHandSideExpression expression, QuestionDotToken questionDotToken, Expression argumentExpression) -> ElementAccessExpression {
    auto node = createBaseDeclaration<ElementAccessExpression>(SyntaxKind::ElementAccessExpression);
    node->expression = expression;
    node->questionDotToken = questionDotToken;
    node->argumentExpression = argumentExpression;
    node->transformFlags |= propagateChildFlags(node->expression) |
        propagateChildFlags(node->questionDotToken) |
        propagateChildFlags(node->argumentExpression);

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    return node;
}

// @api
auto NodeFactory::createElementAccessExpression(Expression expression, Expression index) -> ElementAccessExpression
{
    auto node = createBaseElementAccessExpression(
        parenthesizerRules.parenthesizeLeftSideOfAccess(expression, /*optionalChain*/ false),
        /*questionDotToken*/ undefined,
        asExpression(index)
    );
    if (isSuperKeyword(expression)) {
        // super method calls require a lexical 'this'
        // super method calls require 'super' hoisting in ES2017 and ES2018 async functions and async generators
        node->transformFlags |= TransformFlags::ContainsES2017 |
            TransformFlags::ContainsES2018;
    }
    return node;
}

// @api
auto NodeFactory::createElementAccessChain(Expression expression, QuestionDotToken questionDotToken, Expression index) -> ElementAccessChain
{
    auto node = createBaseElementAccessExpression(
        parenthesizerRules.parenthesizeLeftSideOfAccess(expression, /*optionalChain*/ true),
        questionDotToken,
        asExpression(index)
    ).as<ElementAccessChain>();
    node->flags |= NodeFlags::OptionalChain;
    node->transformFlags |= TransformFlags::ContainsES2020;
    return node;
}

// @api
auto NodeFactory::createBaseCallExpression(LeftHandSideExpression expression, QuestionDotToken questionDotToken, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray) -> CallExpression {
    auto node = createBaseDeclaration<CallExpression>(SyntaxKind::CallExpression);
    node->expression = expression;
    node->questionDotToken = questionDotToken;
    node->typeArguments = typeArguments;
    node->arguments = argumentsArray;
    node->transformFlags |= propagateChildFlags(node->expression) |
        propagateChildFlags(node->questionDotToken) |
        propagateChildrenFlags(node->typeArguments) |
        propagateChildrenFlags(node->arguments);
    if (node->typeArguments) {
        node->transformFlags |= TransformFlags::ContainsTypeScript;
    }
    if (isSuperProperty(node->expression)) {
        node->transformFlags |= TransformFlags::ContainsLexicalThis;
    }
    return node;
}

// @api
auto NodeFactory::createCallExpression(Expression expression, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray)
    -> CallExpression
{
    auto node = createBaseCallExpression(
        parenthesizerRules.parenthesizeLeftSideOfAccess(expression, /*optionalChain*/ false),
        /*questionDotToken*/ undefined,
        asNodeArray(typeArguments),
        parenthesizerRules.parenthesizeExpressionsOfCommaDelimitedList(createNodeArray(argumentsArray))
    );
    if (isImportKeyword(node->expression)) {
        node->transformFlags |= TransformFlags::ContainsDynamicImport;
    }
    return node;
}

// @api
auto NodeFactory::updateCallExpression(CallExpression node, Expression expression, NodeArray<TypeNode> typeArguments,
                                       NodeArray<Expression> argumentsArray) -> CallExpression
{
    return node->expression != expression || node->typeArguments != typeArguments || node->arguments != argumentsArray
               ? update(createCallExpression(expression, typeArguments, argumentsArray), node)
               : node;
}

// @api
auto NodeFactory::createCallChain(Expression expression, QuestionDotToken questionDotToken, NodeArray<TypeNode> typeArguments,
                                  NodeArray<Expression> argumentsArray) -> CallChain
{
    auto node = createBaseCallExpression(
        parenthesizerRules.parenthesizeLeftSideOfAccess(expression, /*optionalChain*/ true),
        questionDotToken,
        asNodeArray(typeArguments),
        parenthesizerRules.parenthesizeExpressionsOfCommaDelimitedList(createNodeArray(argumentsArray))
    ).as<CallChain>();
    node->flags |= NodeFlags::OptionalChain;
    node->transformFlags |= TransformFlags::ContainsES2020;
    return node;
}

// @api

// @api
auto NodeFactory::createNewExpression(Expression expression, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray)
    -> NewExpression
{
    auto node = createBaseDeclaration<NewExpression>(SyntaxKind::NewExpression);
    node->expression = parenthesizerRules.parenthesizeExpressionOfNew(expression);
    node->typeArguments = asNodeArray(typeArguments);
    node->arguments = argumentsArray ? parenthesizerRules.parenthesizeExpressionsOfCommaDelimitedList(argumentsArray) : undefined;
    node->transformFlags |= propagateChildFlags(node->expression) |
        propagateChildrenFlags(node->typeArguments) |
        propagateChildrenFlags(node->arguments) |
        TransformFlags::ContainsES2020;
    if (node->typeArguments) {
        node->transformFlags |= TransformFlags::ContainsTypeScript;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createTaggedTemplateExpression(Expression tag, NodeArray<TypeNode> typeArguments, TemplateLiteral _template)
    -> TaggedTemplateExpression
{
    auto node = createBaseNode<TaggedTemplateExpression>(SyntaxKind::TaggedTemplateExpression);
    node->tag = parenthesizerRules.parenthesizeLeftSideOfAccess(tag, /*optionalChain*/ false);
    node->typeArguments = asNodeArray(typeArguments);
    node->_template = _template;
    node->transformFlags |= propagateChildFlags(node->tag) |
        propagateChildrenFlags(node->typeArguments) |
        propagateChildFlags(node->_template) |
        TransformFlags::ContainsES2015;
    if (node->typeArguments) {
        node->transformFlags |= TransformFlags::ContainsTypeScript;
    }
    if (hasInvalidEscape(node->_template)) {
        node->transformFlags |= TransformFlags::ContainsES2018;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createTypeAssertion(TypeNode type, Expression expression) -> TypeAssertion
{
    auto node = createBaseNode<TypeAssertion>(SyntaxKind::TypeAssertionExpression);
    node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
    node->type = type;
    node->transformFlags |= propagateChildFlags(node->expression) |
        propagateChildFlags(node->type) |
        TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createParenthesizedExpression(Expression expression) -> ParenthesizedExpression
{
    auto node = createBaseNode<ParenthesizedExpression>(SyntaxKind::ParenthesizedExpression);
    node->expression = expression;
    node->transformFlags = propagateChildFlags(node->expression);

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

// @api

// @api
auto NodeFactory::createFunctionExpression(ModifiersArray modifiers, AsteriskToken asteriskToken, Identifier name,
                                           NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                           TypeNode type, Block body) -> FunctionExpression
{
    auto node = createBaseDeclaration<FunctionExpression>(SyntaxKind::FunctionExpression);
    node->modifiers = asNodeArray(modifiers);
    node->asteriskToken = asteriskToken;
    node->name = asName(name);
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = createNodeArray(parameters);
    node->type = type;
    node->body = body;

    auto isAsync = !!(modifiersToFlags(node->modifiers) & ModifierFlags::Async);
    auto isGenerator = !!node->asteriskToken;
    auto isAsyncGenerator = isAsync && isGenerator;

    node->transformFlags = propagateChildrenFlags(node->modifiers) |
        propagateChildFlags(node->asteriskToken) |
        propagateNameFlags(node->name) |
        propagateChildrenFlags(node->typeParameters) |
        propagateChildrenFlags(node->parameters) |
        propagateChildFlags(node->type) |
        (propagateChildFlags(node->body) & ~TransformFlags::ContainsPossibleTopLevelAwait) |
        (isAsyncGenerator ? TransformFlags::ContainsES2018 :
            isAsync ? TransformFlags::ContainsES2017 :
            isGenerator ? TransformFlags::ContainsGenerator :
            TransformFlags::None) |
        (node->typeParameters || node->type ? TransformFlags::ContainsTypeScript : TransformFlags::None) |
        TransformFlags::ContainsHoistedDeclarationOrCompletion;

    node->typeArguments = undefined; // used in quick info
    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    //node->endFlowNode = undefined;
    //node->returnFlowNode = undefined;
    return node;
}

// @api

// @api
auto NodeFactory::createArrowFunction(ModifiersArray modifiers, NodeArray<TypeParameterDeclaration> typeParameters,
                                      NodeArray<ParameterDeclaration> parameters, TypeNode type,
                                      EqualsGreaterThanToken equalsGreaterThanToken, ConciseBody body) -> ArrowFunction
{
    auto node = createBaseDeclaration<ArrowFunction>(SyntaxKind::ArrowFunction);
    node->modifiers = asNodeArray(modifiers);
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = createNodeArray(parameters);
    node->type = type;
    node->equalsGreaterThanToken = !!equalsGreaterThanToken 
        ? equalsGreaterThanToken 
        : createToken(SyntaxKind::EqualsGreaterThanToken).as<EqualsGreaterThanToken>();
    node->body = parenthesizerRules.parenthesizeConciseBodyOfArrowFunction(body);

    auto isAsync = !!(modifiersToFlags(node->modifiers) & ModifierFlags::Async);

    node->transformFlags = propagateChildrenFlags(node->modifiers) |
        propagateChildrenFlags(node->typeParameters) |
        propagateChildrenFlags(node->parameters) |
        propagateChildFlags(node->type) |
        propagateChildFlags(node->equalsGreaterThanToken) |
        (propagateChildFlags(node->body) & ~TransformFlags::ContainsPossibleTopLevelAwait) |
        (node->typeParameters || node->type ? TransformFlags::ContainsTypeScript : TransformFlags::None) |
        (isAsync ? TransformFlags::ContainsES2017 | TransformFlags::ContainsLexicalThis : TransformFlags::None) |
        TransformFlags::ContainsES2015;

    node->typeArguments = undefined; // used in quick info
    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    //node->endFlowNode = undefined;
    //node->returnFlowNode = undefined;
    return node;
}

// @api

// @api
auto NodeFactory::createDeleteExpression(Expression expression) -> DeleteExpression
{
    auto node = createBaseNode<DeleteExpression>(SyntaxKind::DeleteExpression);
    node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
    node->transformFlags |= propagateChildFlags(node->expression);
    return node;
}

// @api

// @api
auto NodeFactory::createTypeOfExpression(Expression expression) -> TypeOfExpression
{
    auto node = createBaseNode<TypeOfExpression>(SyntaxKind::TypeOfExpression);
    node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
    node->transformFlags |= propagateChildFlags(node->expression);
    return node;
}

// @api

// @api
auto NodeFactory::createVoidExpression(Expression expression) -> VoidExpression
{
    auto node = createBaseNode<VoidExpression>(SyntaxKind::VoidExpression);
    node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
    node->transformFlags |= propagateChildFlags(node->expression);
    return node;
}

// @api

// @api
auto NodeFactory::createAwaitExpression(Expression expression) -> AwaitExpression
{
    auto node = createBaseNode<AwaitExpression>(SyntaxKind::AwaitExpression);
    node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
    node->transformFlags |= propagateChildFlags(node->expression) |
        TransformFlags::ContainsES2017 |
        TransformFlags::ContainsES2018 |
        TransformFlags::ContainsAwait;
    return node;
}

// @api

// @api
auto NodeFactory::createPrefixUnaryExpression(PrefixUnaryOperator _operator, Expression operand) -> PrefixUnaryExpression
{
    auto node = createBaseNode<PrefixUnaryExpression>(SyntaxKind::PrefixUnaryExpression);
    node->_operator = _operator;
    node->operand = parenthesizerRules.parenthesizeOperandOfPrefixUnary(operand);
    node->transformFlags |= propagateChildFlags(node->operand);
    // Only set this flag for non-generated identifiers and non-S("local") names. See the
    // comment in `visitPreOrPostfixUnaryExpression` in module.ts
    if (
        (_operator == SyntaxKind::PlusPlusToken || _operator == SyntaxKind::MinusMinusToken) &&
        isIdentifier(node->operand) &&
        !isGeneratedIdentifier(node->operand) &&
        !isLocalName(node->operand)
    ) {
        node->transformFlags |= TransformFlags::ContainsUpdateExpressionForIdentifier;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createPostfixUnaryExpression(Expression operand, PostfixUnaryOperator _operator) -> PostfixUnaryExpression
{
    auto node = createBaseNode<PostfixUnaryExpression>(SyntaxKind::PostfixUnaryExpression);
    node->_operator = _operator;
    node->operand = parenthesizerRules.parenthesizeOperandOfPostfixUnary(operand);
    node->transformFlags |= propagateChildFlags(node->operand);
    // Only set this flag for non-generated identifiers and non-S("local") names. See the
    // comment in `visitPreOrPostfixUnaryExpression` in module.ts
    if (
        isIdentifier(node->operand) &&
        !isGeneratedIdentifier(node->operand) &&
        !isLocalName(node->operand)
    ) {
        node->transformFlags |= TransformFlags::ContainsUpdateExpressionForIdentifier;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createBinaryExpression(Expression left, Node _operator, Expression right) -> BinaryExpression
{
    auto node = createBaseDeclaration<BinaryExpression>(SyntaxKind::BinaryExpression);
    auto operatorToken = asToken(_operator);
    auto operatorKind = (SyntaxKind)operatorToken;
    node->left = parenthesizerRules.parenthesizeLeftSideOfBinary(operatorKind, left);
    node->operatorToken = operatorToken;
    node->right = parenthesizerRules.parenthesizeRightSideOfBinary(operatorKind, node->left, right);
    node->transformFlags |= propagateChildFlags(node->left) |
        propagateChildFlags(node->operatorToken) |
        propagateChildFlags(node->right);
    if (operatorKind == SyntaxKind::QuestionQuestionToken) {
        node->transformFlags |= TransformFlags::ContainsES2020;
    }
    else if (operatorKind == SyntaxKind::EqualsToken) {
        if (isObjectLiteralExpression(node->left)) {
            node->transformFlags |= TransformFlags::ContainsES2015 |
                TransformFlags::ContainsES2018 |
                TransformFlags::ContainsDestructuringAssignment |
                propagateAssignmentPatternFlags(node->left);
        }
        else if (isArrayLiteralExpression(node->left)) {
            node->transformFlags |= TransformFlags::ContainsES2015 |
                TransformFlags::ContainsDestructuringAssignment |
                propagateAssignmentPatternFlags(node->left);
        }
    }
    else if (operatorKind == SyntaxKind::AsteriskAsteriskToken || operatorKind == SyntaxKind::AsteriskAsteriskEqualsToken) {
        node->transformFlags |= TransformFlags::ContainsES2016;
    }
    else if (isLogicalOrCoalescingAssignmentOperator(operatorKind)) {
        node->transformFlags |= TransformFlags::ContainsES2021;
    }
    if (operatorKind == SyntaxKind::InKeyword && isPrivateIdentifier(node->left)) {
        node->transformFlags |= TransformFlags::ContainsPrivateIdentifierInExpression;
    }

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

auto propagateAssignmentPatternFlags(AssignmentPattern node) -> TransformFlags
{
    if (!!(node->transformFlags & TransformFlags::ContainsObjectRestOrSpread))
        return TransformFlags::ContainsObjectRestOrSpread;
    if (!!(node->transformFlags & TransformFlags::ContainsES2018))
    {
        // check for nested spread assignments, otherwise '{ x: { a, ...b } = foo } = c'
        // will not be correctly interpreted by the ES2018 transformer
        for (auto element : getElementsOfBindingOrAssignmentPattern(node))
        {
            auto target = getTargetOfBindingOrAssignmentElement(element);
            if (target && isAssignmentPattern(target))
            {
                if (!!(target->transformFlags & TransformFlags::ContainsObjectRestOrSpread))
                {
                    return TransformFlags::ContainsObjectRestOrSpread;
                }
                if (!!(target->transformFlags & TransformFlags::ContainsES2018))
                {
                    auto flags = propagateAssignmentPatternFlags(target);
                    if (!!flags)
                        return flags;
                }
            }
        }
    }
    return TransformFlags::None;
}

// @api

// @api
auto NodeFactory::createConditionalExpression(Expression condition, QuestionToken questionToken, Expression whenTrue, ColonToken colonToken,
                                              Expression whenFalse) -> ConditionalExpression
{
    auto node = createBaseNode<ConditionalExpression>(SyntaxKind::ConditionalExpression);
    node->condition = parenthesizerRules.parenthesizeConditionOfConditionalExpression(condition);
    node->questionToken = !!questionToken ? questionToken : createToken(SyntaxKind::QuestionToken).as<QuestionToken>();
    node->whenTrue = parenthesizerRules.parenthesizeBranchOfConditionalExpression(whenTrue);
    node->colonToken = !!colonToken ? colonToken : createToken(SyntaxKind::ColonToken).as<ColonToken>();
    node->whenFalse = parenthesizerRules.parenthesizeBranchOfConditionalExpression(whenFalse);
    node->transformFlags |= propagateChildFlags(node->condition) |
        propagateChildFlags(node->questionToken) |
        propagateChildFlags(node->whenTrue) |
        propagateChildFlags(node->colonToken) |
        propagateChildFlags(node->whenFalse);
    return node;
}

// @api

// @api
auto NodeFactory::createTemplateExpression(TemplateHead head, NodeArray<TemplateSpan> templateSpans) -> TemplateExpression
{
    auto node = createBaseNode<TemplateExpression>(SyntaxKind::TemplateExpression);
    node->head = head;
    node->templateSpans = createNodeArray(templateSpans);
    node->transformFlags |= propagateChildFlags(node->head) |
        propagateChildrenFlags(node->templateSpans) |
        TransformFlags::ContainsES2015;
    return node;
}

// @api
auto NodeFactory::getCookedText(SyntaxKind kind, string rawText) -> std::pair<string, boolean>
{
    switch (kind)
    {
    case SyntaxKind::NoSubstitutionTemplateLiteral:
        rawTextScanner.setText(S("`") + rawText + S("`"));
        break;
    case SyntaxKind::TemplateHead:
        // tslint:disable-next-line no-invalid-template-strings
        rawTextScanner.setText(S("`") + rawText + S("${"));
        break;
    case SyntaxKind::TemplateMiddle:
        // tslint:disable-next-line no-invalid-template-strings
        rawTextScanner.setText(S("}") + rawText + S("${"));
        break;
    case SyntaxKind::TemplateTail:
        rawTextScanner.setText(S("}") + rawText + S("`"));
        break;
    }

    auto token = rawTextScanner.scan();
    if (token == SyntaxKind::CloseBracketToken)
    {
        token = rawTextScanner.reScanTemplateToken(/*isTaggedTemplate*/ false);
    }

    if (rawTextScanner.isUnterminated())
    {
        rawTextScanner.setText(string());
        return std::make_pair(S(""), false);
    }

    string tokenValue;
    switch (token)
    {
    case SyntaxKind::NoSubstitutionTemplateLiteral:
    case SyntaxKind::TemplateHead:
    case SyntaxKind::TemplateMiddle:
    case SyntaxKind::TemplateTail:
        tokenValue = rawTextScanner.getTokenValue();
        break;
    }

    if (tokenValue.empty() || rawTextScanner.scan() != SyntaxKind::EndOfFileToken)
    {
        rawTextScanner.setText(string());
        return std::make_pair(S(""), false);
    }

    rawTextScanner.setText(string());
    return std::make_pair(tokenValue, true);
}

auto NodeFactory::createTemplateLiteralLikeNodeChecked(SyntaxKind kind, string text, string rawText, TokenFlags templateFlags)
    -> TemplateLiteralLikeNode
{
    Debug::_assert(!(templateFlags & ~TokenFlags::TemplateLiteralLikeFlags), S("Unsupported template flags."));
    // NOTE: without the assignment to `undefined`, we don't narrow the initial type of `cooked`.
    // eslint-disable-next-line no-undef-init
    string cooked;
    if (!rawText.empty() && rawText != text)
    {
        auto res = getCookedText(kind, rawText);
        if (!std::get<1>(res))
        {
            return Debug::fail<TemplateLiteralLikeNode>(S("Invalid raw text"));
        }

        cooked = std::get<0>(res);
    }
    if (text.empty())
    {
        if (cooked.empty())
        {
            return Debug::fail<TemplateLiteralLikeNode>(S("Arguments 'text' and 'rawText' may not both be undefined."));
        }
        text = cooked;
    }
    else if (!cooked.empty())
    {
        Debug::_assert(text == cooked, S("Expected argument 'text' to be the normalized (i.e. 'cooked') version of argument 'rawText'."));
    }
    return createTemplateLiteralLikeNode(kind, text, rawText, templateFlags);
}

// @api
auto NodeFactory::createTemplateLiteralLikeNode(SyntaxKind kind, string text, string rawText, TokenFlags templateFlags)
    -> TemplateLiteralLikeNode
{
    auto node = createBaseToken<TemplateLiteralLikeNode>(kind);
    node->text = text;
    node->rawText = rawText;
    node->templateFlags = templateFlags & TokenFlags::TemplateLiteralLikeFlags;
    node->transformFlags |= TransformFlags::ContainsES2015;
    if (!!node->templateFlags)
    {
        node->transformFlags |= TransformFlags::ContainsES2018;
    }
    return node;
}

// @api
auto NodeFactory::createTemplateHead(string text, string rawText, TokenFlags templateFlags) -> TemplateHead
{
    return createTemplateLiteralLikeNodeChecked(SyntaxKind::TemplateHead, text, rawText, templateFlags);
}

// @api
auto NodeFactory::createTemplateMiddle(string text, string rawText, TokenFlags templateFlags) -> TemplateMiddle
{
    return createTemplateLiteralLikeNodeChecked(SyntaxKind::TemplateMiddle, text, rawText, templateFlags);
}

// @api
auto NodeFactory::createTemplateTail(string text, string rawText, TokenFlags templateFlags) -> TemplateTail
{
    return createTemplateLiteralLikeNodeChecked(SyntaxKind::TemplateTail, text, rawText, templateFlags);
}

// @api
auto NodeFactory::createNoSubstitutionTemplateLiteral(string text, string rawText, TokenFlags templateFlags)
    -> NoSubstitutionTemplateLiteral
{
    return createTemplateLiteralLikeNodeChecked(SyntaxKind::NoSubstitutionTemplateLiteral, text, rawText, templateFlags);
}

// @api
auto NodeFactory::createYieldExpression(AsteriskToken asteriskToken, Expression expression) -> YieldExpression
{
    Debug::_assert(!asteriskToken || !!expression, S("A `YieldExpression` with an asteriskToken must have an expression."));
    auto node = createBaseNode<YieldExpression>(SyntaxKind::YieldExpression);
    node->expression = !!expression ? parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression) : expression;
    node->asteriskToken = asteriskToken;
    node->transformFlags |= propagateChildFlags(node->expression) |
        propagateChildFlags(node->asteriskToken) |
        TransformFlags::ContainsES2015 |
        TransformFlags::ContainsES2018 |
        TransformFlags::ContainsYield;
    return node;
}

// @api

// @api
auto NodeFactory::createSpreadElement(Expression expression) -> SpreadElement
{
    auto node = createBaseExpression<SpreadElement>(SyntaxKind::SpreadElement);
    node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
    node->transformFlags |= propagateChildFlags(node->expression) | TransformFlags::ContainsES2015 | TransformFlags::ContainsRestOrSpread;
    return node;
}

// @api

// @api
auto NodeFactory::createClassExpression(NodeArray<ModifierLike> modifiers, Identifier name,
                                        NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses,
                                        NodeArray<ClassElement> members) -> ClassExpression
{
    auto node = createBaseClassLikeDeclaration<ClassExpression>(SyntaxKind::ClassExpression, modifiers, name, typeParameters,
                                                                heritageClauses, members);
    node->transformFlags |= TransformFlags::ContainsES2015;
    return node;
}

// @api

// @api
auto NodeFactory::createOmittedExpression() -> OmittedExpression
{
    return createBaseExpression<OmittedExpression>(SyntaxKind::OmittedExpression);
}

// @api
auto NodeFactory::createExpressionWithTypeArguments(Expression expression, NodeArray<TypeNode> typeArguments) -> ExpressionWithTypeArguments
{
    auto node = createBaseNode<ExpressionWithTypeArguments>(SyntaxKind::ExpressionWithTypeArguments);
    node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
    node->typeArguments = typeArguments ? parenthesizerRules.parenthesizeTypeArguments(typeArguments) : undefined;
    node->transformFlags |=
        propagateChildFlags(node->expression) | propagateChildrenFlags(node->typeArguments) | TransformFlags::ContainsES2015;
    return node;
}

// @api

// @api
auto NodeFactory::createAsExpression(Expression expression, TypeNode type) -> AsExpression
{
    auto node = createBaseExpression<AsExpression>(SyntaxKind::AsExpression);
    node->expression = expression;
    node->type = type;
    node->transformFlags |= propagateChildFlags(node->expression) | propagateChildFlags(node->type) | TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createNonNullExpression(Expression expression) -> NonNullExpression
{
    auto node = createBaseExpression<NonNullExpression>(SyntaxKind::NonNullExpression);
    node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
    node->transformFlags |= propagateChildFlags(node->expression) | TransformFlags::ContainsTypeScript;
    return node;
}


auto NodeFactory::createSatisfiesExpression(Expression expression, TypeNode type) -> SatisfiesExpression {
    auto node = createBaseNode<SatisfiesExpression>(SyntaxKind::SatisfiesExpression);
    node->expression = expression;
    node->type = type;
    node->transformFlags |= propagateChildFlags(node->expression) |
        propagateChildFlags(node->type) |
        TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createNonNullChain(Expression expression) -> NonNullChain
{
    auto node = createBaseExpression<NonNullChain>(SyntaxKind::NonNullExpression);
    node->flags |= NodeFlags::OptionalChain;
    node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
    node->transformFlags |= propagateChildFlags(node->expression) | TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createMetaProperty(SyntaxKind keywordToken, Identifier name) -> MetaProperty
{
    auto node = createBaseExpression<MetaProperty>(SyntaxKind::MetaProperty);
    node->keywordToken = keywordToken;
    node->name = name;
    node->transformFlags |= propagateChildFlags(node->name);
    switch (keywordToken)
    {
    case SyntaxKind::NewKeyword:
        node->transformFlags |= TransformFlags::ContainsES2015;
        break;
    case SyntaxKind::ImportKeyword:
        node->transformFlags |= TransformFlags::ContainsESNext;
        break;
    default:
        Debug::_assertNever(node);
    }
    return node;
}

// @api

//
// Misc
//

// @api
auto NodeFactory::createTemplateSpan(Expression expression, Node literal) -> TemplateSpan
{
    auto node = createBaseNode<TemplateSpan>(SyntaxKind::TemplateSpan);
    node->expression = expression;
    node->literal = literal;
    node->transformFlags |= propagateChildFlags(node->expression) | propagateChildFlags(node->literal) | TransformFlags::ContainsES2015;
    return node;
}

// @api

// @api
auto NodeFactory::createSemicolonClassElement() -> SemicolonClassElement
{
    auto node = createBaseNode<SemicolonClassElement>(SyntaxKind::SemicolonClassElement);
    node->transformFlags |= TransformFlags::ContainsES2015;
    return node;
}

//
// Element
//

// @api
auto NodeFactory::createBlock(NodeArray<Statement> statements, boolean multiLine) -> Block
{
    auto node = createBaseNode<Block>(SyntaxKind::Block);
    node->statements = createNodeArray(statements);
    node->multiLine = multiLine;
    node->transformFlags |= propagateChildrenFlags(node->statements);
    return node;
}

// @api

// @api
auto NodeFactory::createVariableStatement(ModifiersArray modifiers, VariableDeclarationList declarationList) -> VariableStatement
{
    auto node = createBaseNode<VariableStatement>(SyntaxKind::VariableStatement);
    node->modifiers = asNodeArray(modifiers);
    //node->declarationList = isArray(declarationList) ? createVariableDeclarationList(declarationList) : declarationList;
    node->declarationList = declarationList;
    node->transformFlags |= propagateChildrenFlags(node->modifiers) |
        propagateChildFlags(node->declarationList);
    if ((modifiersToFlags(node->modifiers) & ModifierFlags::Ambient) > ModifierFlags::None) {
        node->transformFlags = TransformFlags::ContainsTypeScript;
    }

    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    //node->flowNode = undefined; // initialized by binder (FlowContainer)
    return node;
}

// @api

// @api
auto NodeFactory::createEmptyStatement() -> EmptyStatement
{
    return createBaseNode<EmptyStatement>(SyntaxKind::EmptyStatement);
}

// @api
auto NodeFactory::createExpressionStatement(Expression expression) -> ExpressionStatement
{
    auto node = createBaseNode<ExpressionStatement>(SyntaxKind::ExpressionStatement);
    node->expression = parenthesizerRules.parenthesizeExpressionOfExpressionStatement(expression);
    node->transformFlags |= propagateChildFlags(node->expression);
    return node;
}

// @api

// @api
auto NodeFactory::createIfStatement(Expression expression, Statement thenStatement, Statement elseStatement) -> IfStatement
{
    auto node = createBaseNode<IfStatement>(SyntaxKind::IfStatement);
    node->expression = expression;
    node->thenStatement = asEmbeddedStatement(thenStatement);
    node->elseStatement = asEmbeddedStatement(elseStatement);
    node->transformFlags |=
        propagateChildFlags(node->expression) | propagateChildFlags(node->thenStatement) | propagateChildFlags(node->elseStatement);
    return node;
}

// @api

// @api
auto NodeFactory::createDoStatement(Statement statement, Expression expression) -> DoStatement
{
    auto node = createBaseNode<DoStatement>(SyntaxKind::DoStatement);
    node->statement = asEmbeddedStatement(statement);
    node->expression = expression;
    node->transformFlags |= propagateChildFlags(node->statement) | propagateChildFlags(node->expression);
    return node;
}

// @api

// @api
auto NodeFactory::createWhileStatement(Expression expression, Statement statement) -> WhileStatement
{
    auto node = createBaseNode<WhileStatement>(SyntaxKind::WhileStatement);
    node->expression = expression;
    node->statement = asEmbeddedStatement(statement);
    node->transformFlags |= propagateChildFlags(node->expression) | propagateChildFlags(node->statement);
    return node;
}

// @api

// @api
auto NodeFactory::createForStatement(ForInitializer initializer, Expression condition, Expression incrementor, Statement statement)
    -> ForStatement
{
    auto node = createBaseNode<ForStatement>(SyntaxKind::ForStatement);
    node->initializer = initializer;
    node->condition = condition;
    node->incrementor = incrementor;
    node->statement = asEmbeddedStatement(statement);
    node->transformFlags |= propagateChildFlags(node->initializer) | propagateChildFlags(node->condition) |
                            propagateChildFlags(node->incrementor) | propagateChildFlags(node->statement);
    return node;
}

// @api

// @api
auto NodeFactory::createForInStatement(ForInitializer initializer, Expression expression, Statement statement) -> ForInStatement
{
    auto node = createBaseNode<ForInStatement>(SyntaxKind::ForInStatement);
    node->initializer = initializer;
    node->expression = expression;
    node->statement = asEmbeddedStatement(statement);
    node->transformFlags |=
        propagateChildFlags(node->initializer) | propagateChildFlags(node->expression) | propagateChildFlags(node->statement);
    return node;
}

// @api

// @api
auto NodeFactory::createForOfStatement(AwaitKeyword awaitModifier, ForInitializer initializer, Expression expression, Statement statement)
    -> ForOfStatement
{
    auto node = createBaseNode<ForOfStatement>(SyntaxKind::ForOfStatement);
    node->awaitModifier = awaitModifier;
    node->initializer = initializer;
    node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
    node->statement = asEmbeddedStatement(statement);
    node->transformFlags |= propagateChildFlags(node->awaitModifier) | propagateChildFlags(node->initializer) |
                            propagateChildFlags(node->expression) | propagateChildFlags(node->statement) | TransformFlags::ContainsES2015;
    if (awaitModifier)
        node->transformFlags |= TransformFlags::ContainsES2018;
    return node;
}

// @api

// @api
auto NodeFactory::createContinueStatement(Identifier label) -> ContinueStatement
{
    auto node = createBaseNode<ContinueStatement>(SyntaxKind::ContinueStatement);
    node->label = asName(label);
    node->transformFlags |= propagateChildFlags(node->label) | TransformFlags::ContainsHoistedDeclarationOrCompletion;
    return node;
}

// @api

// @api
auto NodeFactory::createBreakStatement(Identifier label) -> BreakStatement
{
    auto node = createBaseNode<BreakStatement>(SyntaxKind::BreakStatement);
    node->label = asName(label);
    node->transformFlags |= propagateChildFlags(node->label) | TransformFlags::ContainsHoistedDeclarationOrCompletion;
    return node;
}

// @api

// @api
auto NodeFactory::createReturnStatement(Expression expression) -> ReturnStatement
{
    auto node = createBaseNode<ReturnStatement>(SyntaxKind::ReturnStatement);
    node->expression = expression;
    // return in an ES2018 async generator must be awaited
    node->transformFlags |=
        propagateChildFlags(node->expression) | TransformFlags::ContainsES2018 | TransformFlags::ContainsHoistedDeclarationOrCompletion;
    return node;
}

// @api

// @api
auto NodeFactory::createWithStatement(Expression expression, Statement statement) -> WithStatement
{
    auto node = createBaseNode<WithStatement>(SyntaxKind::WithStatement);
    node->expression = expression;
    node->statement = asEmbeddedStatement(statement);
    node->transformFlags |= propagateChildFlags(node->expression) | propagateChildFlags(node->statement);
    return node;
}

// @api

// @api
auto NodeFactory::createSwitchStatement(Expression expression, CaseBlock caseBlock) -> SwitchStatement
{
    auto node = createBaseNode<SwitchStatement>(SyntaxKind::SwitchStatement);
    node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
    node->caseBlock = caseBlock;
    node->transformFlags |= propagateChildFlags(node->expression) | propagateChildFlags(node->caseBlock);
    return node;
}

// @api

// @api
auto NodeFactory::createLabeledStatement(Identifier label, Statement statement) -> LabeledStatement
{
    auto node = createBaseNode<LabeledStatement>(SyntaxKind::LabeledStatement);
    node->label = asName(label);
    node->statement = asEmbeddedStatement(statement);
    node->transformFlags |= propagateChildFlags(node->label) | propagateChildFlags(node->statement);
    return node;
}

// @api

// @api
auto NodeFactory::createThrowStatement(Expression expression) -> ThrowStatement
{
    auto node = createBaseNode<ThrowStatement>(SyntaxKind::ThrowStatement);
    node->expression = expression;
    node->transformFlags |= propagateChildFlags(node->expression);
    return node;
}

// @api

// @api
auto NodeFactory::createTryStatement(Block tryBlock, CatchClause catchClause, Block finallyBlock) -> TryStatement
{
    auto node = createBaseNode<TryStatement>(SyntaxKind::TryStatement);
    node->tryBlock = tryBlock;
    node->catchClause = catchClause;
    node->finallyBlock = finallyBlock;
    node->transformFlags |=
        propagateChildFlags(node->tryBlock) | propagateChildFlags(node->catchClause) | propagateChildFlags(node->finallyBlock);
    return node;
}

// @api

// @api
auto NodeFactory::createDebuggerStatement() -> DebuggerStatement
{
    return createBaseNode<DebuggerStatement>(SyntaxKind::DebuggerStatement);
}

// @api
auto NodeFactory::createVariableDeclaration(BindingName name, ExclamationToken exclamationToken, TypeNode type, Expression initializer)
    -> VariableDeclaration
{
    auto node = createBaseVariableLikeDeclaration<VariableDeclaration>(
        SyntaxKind::VariableDeclaration,
        
        /*modifiers*/ undefined, name, type,
        initializer ? parenthesizerRules.parenthesizeExpressionForDisallowedComma(initializer) : undefined);
    node->exclamationToken = exclamationToken;
    node->transformFlags |= propagateChildFlags(node->exclamationToken);
    if (exclamationToken)
    {
        node->transformFlags |= TransformFlags::ContainsTypeScript;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createVariableDeclarationList(NodeArray<VariableDeclaration> declarations, NodeFlags flags) -> VariableDeclarationList
{
    auto node = createBaseNode<VariableDeclarationList>(SyntaxKind::VariableDeclarationList);
    node->flags |= flags & NodeFlags::BlockScoped;
    node->declarations = createNodeArray(declarations);
    node->transformFlags |= propagateChildrenFlags(node->declarations) | TransformFlags::ContainsHoistedDeclarationOrCompletion;
    if (!!(flags & NodeFlags::BlockScoped))
    {
        node->transformFlags |= TransformFlags::ContainsES2015 | TransformFlags::ContainsBlockScopedBinding;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createFunctionDeclaration(NodeArray<ModifierLike> modifiers, AsteriskToken asteriskToken,
                                            Identifier name, NodeArray<TypeParameterDeclaration> typeParameters,
                                            NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body) -> FunctionDeclaration
{
    auto node = createBaseFunctionLikeDeclaration<FunctionDeclaration>(SyntaxKind::FunctionDeclaration, modifiers, name,
                                                                       typeParameters, parameters, type, body);
    node->asteriskToken = asteriskToken;
    if (!node->body || !!(modifiersToFlags(node->modifiers) & ModifierFlags::Ambient))
    {
        node->transformFlags = TransformFlags::ContainsTypeScript;
    }
    else
    {
        node->transformFlags |= propagateChildFlags(node->asteriskToken) | TransformFlags::ContainsHoistedDeclarationOrCompletion;
        if (!!(modifiersToFlags(node->modifiers) & ModifierFlags::Async))
        {
            if (node->asteriskToken)
            {
                node->transformFlags |= TransformFlags::ContainsES2018;
            }
            else
            {
                node->transformFlags |= TransformFlags::ContainsES2017;
            }
        }
        else if (node->asteriskToken)
        {
            node->transformFlags |= TransformFlags::ContainsGenerator;
        }
    }
    return node;
}

// @api

// @api
auto NodeFactory::createClassDeclaration(NodeArray<ModifierLike> modifiers, Identifier name,
                                         NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses,
                                         NodeArray<ClassElement> members) -> ClassDeclaration
{
    auto node = createBaseDeclaration<ClassDeclaration>(SyntaxKind::ClassDeclaration);
    node->modifiers = asNodeArray(modifiers);
    node->name = asName(name);
    node->typeParameters = asNodeArray(typeParameters);
    node->heritageClauses = asNodeArray(heritageClauses);
    node->members = createNodeArray(members);

    if ((modifiersToFlags(node->modifiers) & ModifierFlags::Ambient) > ModifierFlags::None) {
        node->transformFlags = TransformFlags::ContainsTypeScript;
    }
    else {
        node->transformFlags |= propagateChildrenFlags(node->modifiers) |
            propagateNameFlags(node->name) |
            propagateChildrenFlags(node->typeParameters) |
            propagateChildrenFlags(node->heritageClauses) |
            propagateChildrenFlags(node->members) |
            (node->typeParameters ? TransformFlags::ContainsTypeScript : TransformFlags::None) |
            TransformFlags::ContainsES2015;
        if ((node->transformFlags & TransformFlags::ContainsTypeScriptClassSyntax) > TransformFlags::None) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
    }

    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;    
}

// @api

// @api
auto NodeFactory::createInterfaceDeclaration(NodeArray<ModifierLike> modifiers, Identifier name,
                                             NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses,
                                             NodeArray<TypeElement> members) -> InterfaceDeclaration
{
    auto node = createBaseInterfaceOrClassLikeDeclaration<InterfaceDeclaration>(SyntaxKind::InterfaceDeclaration, modifiers,
                                                                                name, typeParameters, heritageClauses);
    node->members = createNodeArray(members);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createTypeAliasDeclaration(NodeArray<ModifierLike> modifiers, Identifier name,
                                             NodeArray<TypeParameterDeclaration> typeParameters, TypeNode type) -> TypeAliasDeclaration
{
    auto node = createBaseGenericNamedDeclaration<TypeAliasDeclaration>(SyntaxKind::TypeAliasDeclaration, modifiers, name,
                                                                        typeParameters);
    node->type = type;
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createEnumDeclaration(NodeArray<ModifierLike> modifiers, Identifier name,
                                        NodeArray<EnumMember> members) -> EnumDeclaration
{
    auto node = createBaseNamedDeclaration<EnumDeclaration>(SyntaxKind::EnumDeclaration, modifiers, name);
    node->members = createNodeArray(members);
    node->transformFlags |= propagateChildrenFlags(node->members) | TransformFlags::ContainsTypeScript;
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // Enum declarations cannot contain `await`
    return node;
}

// @api

// @api
auto NodeFactory::createModuleDeclaration(NodeArray<ModifierLike> modifiers, ModuleName name, ModuleBody body,
                                          NodeFlags flags) -> ModuleDeclaration
{
    // auto node = createBaseDeclaration<ModuleDeclaration>(SyntaxKind::ModuleDeclaration, modifiers);
    // node->flags |= flags & (NodeFlags::Namespace | NodeFlags::NestedNamespace | NodeFlags::GlobalAugmentation);
    // node->name = name;
    // node->body = body;
    // if (!!(modifiersToFlags(node->modifiers) & ModifierFlags::Ambient))
    // {
    //     node->transformFlags = TransformFlags::ContainsTypeScript;
    // }
    // else
    // {
    //     node->transformFlags |= propagateChildFlags(node->name) | propagateChildFlags(node->body) | TransformFlags::ContainsTypeScript;
    // }
    // node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // Module declarations cannot contain `await`.
    // return node;

    auto node = createBaseDeclaration<ModuleDeclaration>(SyntaxKind::ModuleDeclaration);
    node->modifiers = asNodeArray(modifiers);
    node->flags |= flags & (NodeFlags::Namespace | NodeFlags::NestedNamespace | NodeFlags::GlobalAugmentation);
    node->name = name;
    node->body = body;
    if ((modifiersToFlags(node->modifiers) & ModifierFlags::Ambient) > ModifierFlags::None) {
        node->transformFlags = TransformFlags::ContainsTypeScript;
    }
    else {
        node->transformFlags |= propagateChildrenFlags(node->modifiers) |
            propagateChildFlags(node->name) |
            propagateChildFlags(node->body) |
            TransformFlags::ContainsTypeScript;
    }
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // Module declarations cannot contain `await`.

    node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    node->locals.clear(); // initialized by binder (LocalsContainer)
    //node->nextContainer = undefined; // initialized by binder (LocalsContainer)
    return node;    
}

// @api

// @api
auto NodeFactory::createModuleBlock(NodeArray<Statement> statements) -> ModuleBlock
{
    auto node = createBaseNode<ModuleBlock>(SyntaxKind::ModuleBlock);
    node->statements = createNodeArray(statements);
    node->transformFlags |= propagateChildrenFlags(node->statements);
    return node;
}

// @api

// @api
auto NodeFactory::createCaseBlock(NodeArray<CaseOrDefaultClause> clauses) -> CaseBlock
{
    auto node = createBaseNode<CaseBlock>(SyntaxKind::CaseBlock);
    node->clauses = createNodeArray(clauses);
    node->transformFlags |= propagateChildrenFlags(node->clauses);
    return node;
}

// @api

// @api
auto NodeFactory::createNamespaceExportDeclaration(Identifier name) -> NamespaceExportDeclaration
{
    auto node = createBaseNamedDeclaration<NamespaceExportDeclaration>(SyntaxKind::NamespaceExportDeclaration,
                                                                       
                                                                       /*modifiers*/ undefined, name);
    node->transformFlags = TransformFlags::ContainsTypeScript;
    return node;
}

// @api

// @api
auto NodeFactory::createImportEqualsDeclaration(NodeArray<ModifierLike> modifiers, boolean isTypeOnly, Identifier name,
                                                ModuleReference moduleReference) -> ImportEqualsDeclaration
{
    auto node = createBaseNamedDeclaration<ImportEqualsDeclaration>(SyntaxKind::ImportEqualsDeclaration, modifiers, name);
    node->isTypeOnly = isTypeOnly;
    node->moduleReference = moduleReference;
    node->transformFlags |= propagateChildFlags(node->moduleReference);
    if (!isExternalModuleReference(node->moduleReference))
        node->transformFlags |= TransformFlags::ContainsTypeScript;
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // Import= declaration is always parsed in an Await context
    return node;
}

// @api

// @api
auto NodeFactory::createImportDeclaration(NodeArray<ModifierLike> modifiers, ImportClause importClause,
                                          Expression moduleSpecifier, ImportAttributes attributes) -> ImportDeclaration
{
    auto node = createBaseNode<ImportDeclaration>(SyntaxKind::ImportDeclaration);
    node->modifiers = asNodeArray(modifiers);
    node->importClause = importClause;
    node->moduleSpecifier = moduleSpecifier;
    node->attributes = attributes;
    node->transformFlags |= propagateChildFlags(node->importClause) | propagateChildFlags(node->moduleSpecifier);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context

    // node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

// @api

// @api
auto NodeFactory::createImportClause(boolean isTypeOnly, Identifier name, NamedImportBindings namedBindings) -> ImportClause
{
    auto node = createBaseNode<ImportClause>(SyntaxKind::ImportClause);
    node->isTypeOnly = isTypeOnly;
    node->name = name;
    node->namedBindings = namedBindings;
    node->transformFlags |= propagateChildFlags(node->name) | propagateChildFlags(node->namedBindings);
    if (isTypeOnly)
    {
        node->transformFlags |= TransformFlags::ContainsTypeScript;
    }
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
    return node;
}

// @api

auto  NodeFactory::createImportAttributes(NodeArray<ImportAttribute> elements, boolean multiLine, SyntaxKind token) -> ImportAttributes {
    auto node = createBaseNode<ImportAttributes>(SyntaxKind::ImportAttributes);
    node->token = token != SyntaxKind::Unknown ? token : SyntaxKind::WithKeyword;
    node->elements = createNodeArray(elements);
    node->multiLine = multiLine;
    node->transformFlags |= TransformFlags::ContainsESNext;
    return node;
}

// @api
auto NodeFactory::createImportAttribute(Node name, Expression value) -> ImportAttribute {
    auto node = createBaseNode<ImportAttribute>(SyntaxKind::ImportAttribute);
    node->name = name;
    node->value = value;
    node->transformFlags |= TransformFlags::ContainsESNext;
    return node;
}

// @api
auto NodeFactory::createNamespaceImport(Identifier name) -> NamespaceImport
{
    auto node = createBaseNode<NamespaceImport>(SyntaxKind::NamespaceImport);
    node->name = name;
    node->transformFlags |= propagateChildFlags(node->name);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
    return node;
}

// @api

// @api
auto NodeFactory::createNamespaceExport(Identifier name) -> NamespaceExport
{
    auto node = createBaseNode<NamespaceExport>(SyntaxKind::NamespaceExport);
    node->name = name;
    node->transformFlags |= propagateChildFlags(node->name) | TransformFlags::ContainsESNext;
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
    return node;
}

// @api

// @api
auto NodeFactory::createNamedImports(NodeArray<ImportSpecifier> elements) -> NamedImports
{
    auto node = createBaseNode<NamedImports>(SyntaxKind::NamedImports);
    node->elements = createNodeArray(elements);
    node->transformFlags |= propagateChildrenFlags(node->elements);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
    return node;
}

// @api

// @api
auto NodeFactory::createImportSpecifier(boolean isTypeOnly, Identifier propertyName, Identifier name) -> ImportSpecifier
{
    auto node = createBaseNode<ImportSpecifier>(SyntaxKind::ImportSpecifier);
    node->isTypeOnly = isTypeOnly;
    node->propertyName = propertyName;
    node->name = name;
    node->transformFlags |= propagateChildFlags(node->propertyName) | propagateChildFlags(node->name);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
    return node;
}

// @api

// @api
auto NodeFactory::createExportAssignment(NodeArray<ModifierLike> modifiers, boolean isExportEquals,
                                         Expression expression) -> ExportAssignment
{
    auto node = createBaseNode<ExportAssignment>(SyntaxKind::ExportAssignment);
    node->modifiers = asNodeArray(modifiers);
    node->isExportEquals = isExportEquals;
    node->expression = isExportEquals
                           ? parenthesizerRules.parenthesizeRightSideOfBinary(SyntaxKind::EqualsToken, /*leftSide*/ undefined, expression)
                           : parenthesizerRules.parenthesizeExpressionOfExportDefault(expression);
    node->transformFlags |= propagateChildrenFlags(node->modifiers) | propagateChildFlags(node->expression);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

// @api

// @api
auto NodeFactory::createExportDeclaration(NodeArray<ModifierLike> modifiers, boolean isTypeOnly,
                                          NamedExportBindings exportClause, Expression moduleSpecifier, ImportAttributes attributes) -> ExportDeclaration
{
    auto node = createBaseNode<ExportDeclaration>(SyntaxKind::ExportDeclaration);
    node->modifiers = asNodeArray(modifiers);
    node->isTypeOnly = isTypeOnly;
    node->exportClause = exportClause;
    node->moduleSpecifier = moduleSpecifier;
    node->attributes = attributes;
    node->transformFlags |= propagateChildrenFlags(node->modifiers) | 
        propagateChildFlags(node->exportClause) | 
        propagateChildFlags(node->moduleSpecifier);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context

    //node->jsDoc = undefined; // initialized by parser (JsDocContainer)
    return node;
}

// @api

// @api
auto NodeFactory::createNamedExports(NodeArray<ExportSpecifier> elements) -> NamedExports
{
    auto node = createBaseNode<NamedExports>(SyntaxKind::NamedExports);
    node->elements = createNodeArray(elements);
    node->transformFlags |= propagateChildrenFlags(node->elements);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
    return node;
}

// @api

// @api
auto NodeFactory::createExportSpecifier(boolean isTypeOnly, Identifier propertyName, Identifier name) -> ExportSpecifier
{
    auto node = createBaseNode<ExportSpecifier>(SyntaxKind::ExportSpecifier);
    node->isTypeOnly = isTypeOnly;
    node->propertyName = asName(propertyName);
    node->name = asName(name);
    node->transformFlags |= propagateChildFlags(node->propertyName) | propagateChildFlags(node->name);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
    return node;
}

// @api

// @api
auto NodeFactory::createMissingDeclaration() -> MissingDeclaration
{
    auto node = createBaseDeclaration<MissingDeclaration>(SyntaxKind::MissingDeclaration,
                                                          
                                                          /*modifiers*/ undefined);
    return node;
}

//
// Module references
//

// @api
auto NodeFactory::createExternalModuleReference(Expression expression) -> ExternalModuleReference
{
    auto node = createBaseNode<ExternalModuleReference>(SyntaxKind::ExternalModuleReference);
    node->expression = expression;
    node->transformFlags |= propagateChildFlags(node->expression);
    node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
    return node;
}

//
// JSDoc
//

auto NodeFactory::createJSDocAllType() -> JSDocAllType
{
    return createBaseNode<JSDocAllType>(SyntaxKind::JSDocAllType);
}

auto NodeFactory::createJSDocUnknownType() -> JSDocUnknownType
{
    return createBaseNode<JSDocUnknownType>(SyntaxKind::JSDocUnknownType);
}

auto NodeFactory::createJSDocNonNullableType(TypeNode type, boolean postfix) -> JSDocNonNullableType
{
    auto typeNode = postfix ? !!type ? parenthesizerRules.parenthesizeNonArrayTypeOfPostfixType(type) : type : type;
    auto node = createJSDocUnaryTypeWorker<JSDocNonNullableType>(SyntaxKind::JSDocNonNullableType, typeNode);
    node->postfix = postfix;
    return node;
}

auto NodeFactory::createJSDocNullableType(TypeNode type, boolean postfix) -> JSDocNullableType
{
    auto typeNode = postfix ? !!type ? parenthesizerRules.parenthesizeNonArrayTypeOfPostfixType(type) : type : type;
    auto node = createJSDocUnaryTypeWorker<JSDocNullableType>(
        SyntaxKind::JSDocNullableType,
        typeNode);
    node->postfix = postfix;
    return node;
}

auto NodeFactory::createJSDocOptionalType(TypeNode type) -> JSDocOptionalType
{
    auto node = createBaseNode<JSDocOptionalType>(SyntaxKind::JSDocOptionalType);
    node->type = type;
    return node;
}

auto NodeFactory::createJSDocVariadicType(TypeNode type) -> JSDocVariadicType
{
    auto node = createBaseNode<JSDocVariadicType>(SyntaxKind::JSDocVariadicType);
    node->type = type;
    return node;
}

auto NodeFactory::createJSDocNamepathType(TypeNode type) -> JSDocNamepathType
{
    auto node = createBaseNode<JSDocNamepathType>(SyntaxKind::JSDocNamepathType);
    node->type = type;
    return node;
}

// @api
auto NodeFactory::createJSDocFunctionType(NodeArray<ParameterDeclaration> parameters, TypeNode type) -> JSDocFunctionType
{
    auto node = createBaseSignatureDeclaration<JSDocFunctionType>(SyntaxKind::JSDocFunctionType,
                                                                  
                                                                  /*modifiers*/ undefined,
                                                                  /*name*/ undefined,
                                                                  /*typeParameters*/ undefined, parameters, type);
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocTypeLiteral(NodeArray<JSDocPropertyLikeTag> propertyTags, boolean isArrayType) -> JSDocTypeLiteral
{
    auto node = createBaseNode<JSDocTypeLiteral>(SyntaxKind::JSDocTypeLiteral);
    node->jsDocPropertyTags = asNodeArray(propertyTags);
    node->isArrayType = isArrayType;
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocTypeExpression(TypeNode type) -> JSDocTypeExpression
{
    auto node = createBaseNode<JSDocTypeExpression>(SyntaxKind::JSDocTypeExpression);
    node->type = type;
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocSignature(NodeArray<JSDocTemplateTag> typeParameters, NodeArray<JSDocParameterTag> parameters,
                                       JSDocReturnTag type) -> JSDocSignature
{
    auto node = createBaseNode<JSDocSignature>(SyntaxKind::JSDocSignature);
    node->typeParameters = asNodeArray(typeParameters);
    node->parameters = createNodeArray(parameters);
    node->type = type;
    return node;
}

// @api
auto NodeFactory::getDefaultTagNameForKind(SyntaxKind kind) -> string
{
    switch (kind)
    {
    case SyntaxKind::JSDocTypeTag:
        return S("type");
    case SyntaxKind::JSDocReturnTag:
        return S("returns");
    case SyntaxKind::JSDocThisTag:
        return S("this");
    case SyntaxKind::JSDocEnumTag:
        return S("enum");
    case SyntaxKind::JSDocAuthorTag:
        return S("author");
    case SyntaxKind::JSDocClassTag:
        return S("class");
    case SyntaxKind::JSDocPublicTag:
        return S("public");
    case SyntaxKind::JSDocPrivateTag:
        return S("private");
    case SyntaxKind::JSDocProtectedTag:
        return S("protected");
    case SyntaxKind::JSDocReadonlyTag:
        return S("readonly");
    case SyntaxKind::JSDocTemplateTag:
        return S("template");
    case SyntaxKind::JSDocTypedefTag:
        return S("typedef");
    case SyntaxKind::JSDocParameterTag:
        return S("param");
    case SyntaxKind::JSDocPropertyTag:
        return S("prop");
    case SyntaxKind::JSDocCallbackTag:
        return S("callback");
    case SyntaxKind::JSDocAugmentsTag:
        return S("augments");
    case SyntaxKind::JSDocImplementsTag:
        return S("implements");
    default:
        return Debug::fail<string>(S("Unsupported kind"));
    }
}

auto NodeFactory::getDefaultTagName(JSDocTag node) -> Identifier
{
    auto defaultTagName = getDefaultTagNameForKind(node);
    return node->tagName->escapedText == escapeLeadingUnderscores(defaultTagName) ? node->tagName : createIdentifier(defaultTagName);
}

// @api
auto NodeFactory::createJSDocTemplateTag(Identifier tagName, JSDocTypeExpression constraint,
                                         NodeArray<TypeParameterDeclaration> typeParameters, string comment) -> JSDocTemplateTag
{
    auto node =
        createBaseJSDocTag<JSDocTemplateTag>(SyntaxKind::JSDocTemplateTag, tagName ? tagName : createIdentifier(S("template")), comment);
    node->constraint = constraint;
    node->typeParameters = createNodeArray(typeParameters);
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocTypedefTag(Identifier tagName, Node typeExpression, Node fullName, string comment) -> JSDocTypedefTag
{
    auto node =
        createBaseJSDocTag<JSDocTypedefTag>(SyntaxKind::JSDocTypedefTag, tagName ? tagName : createIdentifier(S("typedef")), comment);
    node->typeExpression = typeExpression;
    node->fullName = fullName;
    node->name = getJSDocTypeAliasName(fullName);
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocParameterTag(Identifier tagName, EntityName name, boolean isBracketed, JSDocTypeExpression typeExpression,
                                          boolean isNameFirst, string comment) -> JSDocParameterTag
{
    auto node =
        createBaseJSDocTag<JSDocParameterTag>(SyntaxKind::JSDocParameterTag, tagName ? tagName : createIdentifier(S("param")), comment);
    node->typeExpression = typeExpression;
    node->name = name;
    node->isNameFirst = !!isNameFirst;
    node->isBracketed = isBracketed;
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocPropertyTag(Identifier tagName, EntityName name, boolean isBracketed, JSDocTypeExpression typeExpression,
                                         boolean isNameFirst, string comment) -> JSDocPropertyTag
{
    auto node =
        createBaseJSDocTag<JSDocPropertyTag>(SyntaxKind::JSDocPropertyTag, tagName ? tagName : createIdentifier(S("prop")), comment);
    node->typeExpression = typeExpression;
    node->name = name;
    node->isNameFirst = !!isNameFirst;
    node->isBracketed = isBracketed;
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocCallbackTag(Identifier tagName, JSDocSignature typeExpression, Node fullName, string comment)
    -> JSDocCallbackTag
{
    auto node =
        createBaseJSDocTag<JSDocCallbackTag>(SyntaxKind::JSDocCallbackTag, tagName ? tagName : createIdentifier(S("callback")), comment);
    node->typeExpression = typeExpression;
    node->fullName = fullName;
    node->name = getJSDocTypeAliasName(fullName);
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocAugmentsTag(Identifier tagName, JSDocAugmentsTag className, string comment) -> JSDocAugmentsTag
{
    auto node =
        createBaseJSDocTag<JSDocAugmentsTag>(SyntaxKind::JSDocAugmentsTag, tagName ? tagName : createIdentifier(S("augments")), comment);
    // TODO: review
    // node->_class = className;
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocImplementsTag(Identifier tagName, JSDocImplementsTag className, string comment) -> JSDocImplementsTag
{
    auto node = createBaseJSDocTag<JSDocImplementsTag>(SyntaxKind::JSDocImplementsTag,
                                                       tagName ? tagName : createIdentifier(S("implements")), comment);
    // TODO: review
    // node->class = className;
    return node;
}

// @api
auto NodeFactory::createJSDocSeeTag(Identifier tagName, JSDocNameReference name, string comment) -> JSDocSeeTag
{
    auto node = createBaseJSDocTag<JSDocSeeTag>(SyntaxKind::JSDocSeeTag, tagName ? tagName : createIdentifier(S("see")), comment);
    node->name = name;
    return node;
}

// @api

// @api
auto NodeFactory::createJSDocNameReference(EntityName name) -> JSDocNameReference
{
    auto node = createBaseNode<JSDocNameReference>(SyntaxKind::JSDocNameReference);
    node->name = name;
    return node;
}

// @api
auto NodeFactory::createJSDocMemberName(Node left, Identifier right) -> JSDocMemberName {
    auto node = createBaseNode<JSDocMemberName>(SyntaxKind::JSDocMemberName);
    node->left = left;
    node->right = right;
    node->transformFlags |= propagateChildFlags(node->left) |
        propagateChildFlags(node->right);
    return node;
}

// @api
auto NodeFactory::createJSDocUnknownTag(Identifier tagName, string comment) -> JSDocUnknownTag
{
    auto node = createBaseJSDocTag<JSDocUnknownTag>(SyntaxKind::JSDocTag, tagName, comment);
    return node;
}

// @api
auto NodeFactory::createJSDocComment(string comment, NodeArray<JSDocTag> tags) -> JSDoc
{
    auto node = createBaseNode<JSDoc>(SyntaxKind::JSDocComment);
    node->comment = comment;
    node->tags = asNodeArray(tags);
    return node;
}

//
// JSX
//

// @api
auto NodeFactory::createJsxElement(JsxOpeningElement openingElement, NodeArray<JsxChild> children, JsxClosingElement closingElement)
    -> JsxElement
{
    auto node = createBaseNode<JsxElement>(SyntaxKind::JsxElement);
    node->openingElement = openingElement;
    node->children = createNodeArray(children);
    node->closingElement = closingElement;
    node->transformFlags |= propagateChildFlags(node->openingElement) | propagateChildrenFlags(node->children) |
                            propagateChildFlags(node->closingElement) | TransformFlags::ContainsJsx;
    return node;
}

// @api

// @api
auto NodeFactory::createJsxSelfClosingElement(JsxTagNameExpression tagName, NodeArray<TypeNode> typeArguments, JsxAttributes attributes)
    -> JsxSelfClosingElement
{
    auto node = createBaseNode<JsxSelfClosingElement>(SyntaxKind::JsxSelfClosingElement);
    node->tagName = tagName;
    node->typeArguments = asNodeArray(typeArguments);
    node->attributes = attributes;
    node->transformFlags |= propagateChildFlags(node->tagName) | propagateChildrenFlags(node->typeArguments) |
                            propagateChildFlags(node->attributes) | TransformFlags::ContainsJsx;
    if (node->typeArguments)
    {
        node->transformFlags |= TransformFlags::ContainsTypeScript;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createJsxOpeningElement(JsxTagNameExpression tagName, NodeArray<TypeNode> typeArguments, JsxAttributes attributes)
    -> JsxOpeningElement
{
    auto node = createBaseNode<JsxOpeningElement>(SyntaxKind::JsxOpeningElement);
    node->tagName = tagName;
    node->typeArguments = asNodeArray(typeArguments);
    node->attributes = attributes;
    node->transformFlags |= propagateChildFlags(node->tagName) | propagateChildrenFlags(node->typeArguments) |
                            propagateChildFlags(node->attributes) | TransformFlags::ContainsJsx;
    if (typeArguments)
    {
        node->transformFlags |= TransformFlags::ContainsTypeScript;
    }
    return node;
}

// @api

// @api
auto NodeFactory::createJsxClosingElement(JsxTagNameExpression tagName) -> JsxClosingElement
{
    auto node = createBaseNode<JsxClosingElement>(SyntaxKind::JsxClosingElement);
    node->tagName = tagName;
    node->transformFlags |= propagateChildFlags(node->tagName) | TransformFlags::ContainsJsx;
    return node;
}

// @api

// @api
auto NodeFactory::createJsxFragment(JsxOpeningFragment openingFragment, NodeArray<JsxChild> children, JsxClosingFragment closingFragment)
    -> JsxFragment
{
    auto node = createBaseNode<JsxFragment>(SyntaxKind::JsxFragment);
    node->openingFragment = openingFragment;
    node->children = createNodeArray(children);
    node->closingFragment = closingFragment;
    node->transformFlags |= propagateChildFlags(node->openingFragment) | propagateChildrenFlags(node->children) |
                            propagateChildFlags(node->closingFragment) | TransformFlags::ContainsJsx;
    return node;
}

// @api

// @api
auto NodeFactory::createJsxText(string text, boolean containsOnlyTriviaWhiteSpaces) -> JsxText
{
    auto node = createBaseNode<JsxText>(SyntaxKind::JsxText);
    node->text = text;
    node->containsOnlyTriviaWhiteSpaces = !!containsOnlyTriviaWhiteSpaces;
    node->transformFlags |= TransformFlags::ContainsJsx;
    return node;
}

// @api

// @api
auto NodeFactory::createJsxOpeningFragment() -> JsxOpeningFragment
{
    auto node = createBaseNode<JsxOpeningFragment>(SyntaxKind::JsxOpeningFragment);
    node->transformFlags |= TransformFlags::ContainsJsx;
    return node;
}

// @api
auto NodeFactory::createJsxJsxClosingFragment() -> JsxClosingFragment
{
    auto node = createBaseNode<JsxClosingFragment>(SyntaxKind::JsxClosingFragment);
    node->transformFlags |= TransformFlags::ContainsJsx;
    return node;
}

// @api
auto NodeFactory::createJsxAttribute(Identifier name, Node initializer) -> JsxAttribute
{
    auto node = createBaseNode<JsxAttribute>(SyntaxKind::JsxAttribute);
    node->name = name;
    node->initializer = initializer;
    node->transformFlags |= propagateChildFlags(node->name) | propagateChildFlags(node->initializer) | TransformFlags::ContainsJsx;
    return node;
}

// @api

// @api
auto NodeFactory::createJsxAttributes(NodeArray<JsxAttributeLike> properties) -> JsxAttributes
{
    auto node = createBaseNode<JsxAttributes>(SyntaxKind::JsxAttributes);
    node->properties = createNodeArray(properties);
    node->transformFlags |= propagateChildrenFlags(node->properties) | TransformFlags::ContainsJsx;
    return node;
}

// @api

// @api
auto NodeFactory::createJsxSpreadAttribute(Expression expression) -> JsxSpreadAttribute
{
    auto node = createBaseNode<JsxSpreadAttribute>(SyntaxKind::JsxSpreadAttribute);
    node->expression = expression;
    node->transformFlags |= propagateChildFlags(node->expression) | TransformFlags::ContainsJsx;
    return node;
}

// @api

// @api
auto NodeFactory::createJsxExpression(DotDotDotToken dotDotDotToken, Expression expression) -> JsxExpression
{
    auto node = createBaseNode<JsxExpression>(SyntaxKind::JsxExpression);
    node->dotDotDotToken = dotDotDotToken;
    node->expression = expression;
    node->transformFlags |= propagateChildFlags(node->dotDotDotToken) | propagateChildFlags(node->expression) | TransformFlags::ContainsJsx;
    return node;
}

// @api
auto NodeFactory::createJsxNamespacedName(Identifier namespace_, Identifier name) -> JsxNamespacedName {
    auto node = createBaseNode<JsxNamespacedName>(SyntaxKind::JsxNamespacedName);
    node->_namespace = namespace_;
    node->name = name;
    node->transformFlags |= propagateChildFlags(node->_namespace) |
        propagateChildFlags(node->name) |
        TransformFlags::ContainsJsx;
    return node;
}

// @api

//
// Clauses
//

// @api
auto NodeFactory::createCaseClause(Expression expression, NodeArray<Statement> statements) -> CaseClause
{
    auto node = createBaseNode<CaseClause>(SyntaxKind::CaseClause);
    node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
    node->statements = createNodeArray(statements);
    node->transformFlags |= propagateChildFlags(node->expression) | propagateChildrenFlags(node->statements);
    return node;
}

// @api

// @api
auto NodeFactory::createDefaultClause(NodeArray<Statement> statements) -> DefaultClause
{
    auto node = createBaseNode<DefaultClause>(SyntaxKind::DefaultClause);
    node->statements = createNodeArray(statements);
    node->transformFlags = propagateChildrenFlags(node->statements);
    return node;
}

// @api

// @api
auto NodeFactory::createHeritageClause(SyntaxKind token, NodeArray<ExpressionWithTypeArguments> types) -> HeritageClause
{
    auto node = createBaseNode<HeritageClause>(SyntaxKind::HeritageClause);
    node->token = token;
    node->types = createNodeArray(types);
    node->transformFlags |= propagateChildrenFlags(node->types);
    switch (token)
    {
    case SyntaxKind::ExtendsKeyword:
        node->transformFlags |= TransformFlags::ContainsES2015;
        break;
    case SyntaxKind::ImplementsKeyword:
        node->transformFlags |= TransformFlags::ContainsTypeScript;
        break;
    default:
        Debug::_assertNever(node);
    }
    return node;
}

// @api

// @api
auto NodeFactory::createCatchClause(VariableDeclaration variableDeclaration, Block block) -> CatchClause
{
    auto node = createBaseNode<CatchClause>(SyntaxKind::CatchClause);
    // TODO: review it
    // variableDeclaration = variableDeclaration;
    node->variableDeclaration = variableDeclaration;
    node->block = block;
    node->transformFlags |= propagateChildFlags(node->variableDeclaration) | propagateChildFlags(node->block);
    if (!variableDeclaration)
        node->transformFlags |= TransformFlags::ContainsES2019;
    return node;
}

// @api

//
// Property assignments
//

// @api
auto NodeFactory::createPropertyAssignment(PropertyName name, Expression initializer) -> PropertyAssignment
{
    auto node = createBaseNamedDeclaration<PropertyAssignment>(SyntaxKind::PropertyAssignment,
                                                               
                                                               /*modifiers*/ undefined, name);
    node->initializer = parenthesizerRules.parenthesizeExpressionForDisallowedComma(initializer);
    node->transformFlags |= propagateChildFlags(node->name) | propagateChildFlags(node->initializer);
    return node;
}

// @api
auto NodeFactory::createShorthandPropertyAssignment(Identifier name, Expression objectAssignmentInitializer) -> ShorthandPropertyAssignment
{
    auto node = createBaseNamedDeclaration<ShorthandPropertyAssignment>(SyntaxKind::ShorthandPropertyAssignment,
                                                                        
                                                                        /*modifiers*/ undefined, name);
    node->objectAssignmentInitializer =
        objectAssignmentInitializer ? parenthesizerRules.parenthesizeExpressionForDisallowedComma(objectAssignmentInitializer) : undefined;
    node->transformFlags |= propagateChildFlags(node->objectAssignmentInitializer) | TransformFlags::ContainsES2015;
    return node;
}

// @api
auto NodeFactory::createSpreadAssignment(Expression expression) -> SpreadAssignment
{
    auto node = createBaseNode<SpreadAssignment>(SyntaxKind::SpreadAssignment);
    node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
    node->transformFlags |=
        propagateChildFlags(node->expression) | TransformFlags::ContainsES2018 | TransformFlags::ContainsObjectRestOrSpread;
    return node;
}

//
// Enum
//

// @api
auto NodeFactory::createEnumMember(PropertyName name, Expression initializer) -> EnumMember
{
    auto node = createBaseNode<EnumMember>(SyntaxKind::EnumMember);
    node->name = asName(name);
    node->initializer = initializer ? parenthesizerRules.parenthesizeExpressionForDisallowedComma(initializer) : undefined;
    node->transformFlags |= propagateChildFlags(node->name) | propagateChildFlags(node->initializer) | TransformFlags::ContainsTypeScript;
    return node;
}

//
// Top-level nodes
//

// @api
auto NodeFactory::createSourceFile(NodeArray<Statement> statements, EndOfFileToken endOfFileToken, NodeFlags flags) -> SourceFile
{
    auto node = /*createBaseSourceFileNode*/ createBaseNode<SourceFile>(SyntaxKind::SourceFile);
    node->statements = createNodeArray(statements);
    node->endOfFileToken = endOfFileToken;
    node->flags |= flags;
    node->fileName = string();
    node->text = string();
    node->languageVersion = ScriptTarget::ES3;
    node->languageVariant = LanguageVariant::Standard;
    node->scriptKind = ScriptKind::Unknown;
    node->isDeclarationFile = false;
    node->hasNoDefaultLib = false;
    node->transformFlags |= propagateChildrenFlags(node->statements) | propagateChildFlags(node->endOfFileToken);
    return node;
}

auto NodeFactory::cloneSourceFileWithChanges(SourceFile source, NodeArray<Statement> statements, boolean isDeclarationFile,
                                             NodeArray<FileReference> referencedFiles, NodeArray<FileReference> typeReferences,
                                             boolean hasNoDefaultLib, NodeArray<FileReference> libReferences) -> SourceFile
{
    auto node = createBaseNode<SourceFile>(SyntaxKind::SourceFile);
    // TODO: finish it
    /*
    for (const p in source) {
        if (p === "emitNode" || hasProperty(node, p) || !hasProperty(source, p)) continue;
        (node as any)[p] = (source as any)[p];
    }
    */
    node->flags |= source->flags;
    node->statements = createNodeArray(statements);
    node->endOfFileToken = source->endOfFileToken;
    node->isDeclarationFile = isDeclarationFile;
    copy(node->referencedFiles, referencedFiles);
    copy(node->typeReferenceDirectives, typeReferences);
    node->hasNoDefaultLib = hasNoDefaultLib;
    copy(node->libReferenceDirectives, libReferences);
    node->transformFlags = propagateChildrenFlags(node->statements) | propagateChildFlags(node->endOfFileToken);
    return node;
}

auto NodeFactory::updateSourceFile(SourceFile node, NodeArray<Statement> statements, boolean isDeclarationFile,
                                   NodeArray<FileReference> referencedFiles, NodeArray<FileReference> typeReferenceDirectives,
                                   boolean hasNoDefaultLib, NodeArray<FileReference> libReferenceDirectives) -> SourceFile
{
    return node->statements != statements || node->isDeclarationFile != isDeclarationFile || node->referencedFiles != referencedFiles ||
                   node->typeReferenceDirectives != typeReferenceDirectives || node->hasNoDefaultLib != hasNoDefaultLib ||
                   node->libReferenceDirectives != libReferenceDirectives
               ? update(cloneSourceFileWithChanges(node, statements, isDeclarationFile, referencedFiles, typeReferenceDirectives,
                                                   hasNoDefaultLib, libReferenceDirectives),
                        node)
               : node;
}
} // namespace ts