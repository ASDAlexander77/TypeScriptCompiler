#include "node_factory.h"

namespace ts
{
    auto NodeFactory::propagateIdentifierNameFlags(Identifier node) -> TransformFlags
    {
        // An IdentifierName is allowed to be `await`
        return propagateChildFlags(node) & ~TransformFlags::ContainsPossibleTopLevelAwait;
    }

    auto NodeFactory::propagatePropertyNameFlagsOfChild(PropertyName node, TransformFlags transformFlags) -> TransformFlags
    {
        return transformFlags | (node->transformFlags & TransformFlags::PropertyNamePropagatingFlags);
    }    

    auto NodeFactory::propagateChildFlags(Node child) -> TransformFlags
    {
        if (!child)
            return TransformFlags::None;
        auto childFlags = child->transformFlags & ~getTransformFlagsSubtreeExclusions(child->kind);
        return isNamedDeclaration(child) && isPropertyName(child.as<NamedDeclaration>()->name) ? propagatePropertyNameFlagsOfChild(child.as<NamedDeclaration>()->name, childFlags) : childFlags;
    }

    auto NodeFactory::propagateAssignmentPatternFlags(AssignmentPattern node) -> TransformFlags {
        if (!!(node->transformFlags & TransformFlags::ContainsObjectRestOrSpread)) return TransformFlags::ContainsObjectRestOrSpread;
        if (!!(node->transformFlags & TransformFlags::ContainsES2018)) {
            // check for nested spread assignments, otherwise '{ x: { a, ...b } = foo } = c'
            // will not be correctly interpreted by the ES2018 transformer
            for (auto element : getElementsOfBindingOrAssignmentPattern(node)) {
                auto target = getTargetOfBindingOrAssignmentElement(element);
                if (!!(target && isAssignmentPattern(target))) {
                    if (!!(target->transformFlags & TransformFlags::ContainsObjectRestOrSpread)) {
                        return TransformFlags::ContainsObjectRestOrSpread;
                    }
                    if (!!(target->transformFlags & TransformFlags::ContainsES2018)) {
                        auto flags = propagateAssignmentPatternFlags(target);
                        if (!!flags) return flags;
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

    /* @internal*/ auto NodeFactory::createStringLiteral(string text, boolean isSingleQuote, boolean hasExtendedUnicodeEscape) -> StringLiteral // eslint-disable-line @typescript-eslint/unified-signatures
    {
        auto node = createBaseStringLiteral(text, isSingleQuote);
        node->hasExtendedUnicodeEscape = hasExtendedUnicodeEscape;
        if (hasExtendedUnicodeEscape)
            node->transformFlags |= TransformFlags::ContainsES2015;
        return node;
    }

    auto NodeFactory::createBaseIdentifier(string text, SyntaxKind originalKeywordKind)
    {
        if (originalKeywordKind == SyntaxKind::Unknown && !text.empty())
        {
            originalKeywordKind = scanner->stringToToken(text);
        }
        if (originalKeywordKind == SyntaxKind::Identifier)
        {
            originalKeywordKind = SyntaxKind::Unknown;
        }
        auto node = createBaseNode<Identifier>(SyntaxKind::Identifier);
        node->originalKeywordKind = originalKeywordKind;
        node->escapedText = escapeLeadingUnderscores(text);
        return node;
    }

    /* @internal */ auto NodeFactory::createIdentifier(string text, NodeArray</*TypeNode | TypeParameterDeclaration*/ Node> typeArguments, SyntaxKind originalKeywordKind) -> Identifier // eslint-disable-line @typescript-eslint/unified-signatures
    {
        auto node = createBaseIdentifier(text, originalKeywordKind);
        if (!!typeArguments)
        {
            // NOTE: we do not use `setChildren` here because typeArguments in an identifier do not contribute to transformations
            copy(node->typeArguments, createNodeArray(typeArguments));
        }
        if (node->originalKeywordKind == SyntaxKind::AwaitKeyword)
        {
            node->transformFlags |= TransformFlags::ContainsPossibleTopLevelAwait;
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

    auto NodeFactory::createToken(SyntaxKind token) -> Node
    {
        Debug::_assert(token >= SyntaxKind::FirstToken && token <= SyntaxKind::LastToken, S("Invalid token"));
        Debug::_assert(token <= SyntaxKind::FirstTemplateToken || token >= SyntaxKind::LastTemplateToken, S("Invalid token. Use 'createTemplateLiteralLikeNode' to create template literals."));
        Debug::_assert(token <= SyntaxKind::FirstLiteralToken || token >= SyntaxKind::LastLiteralToken, S("Invalid token. Use 'createLiteralLikeNode' to create literals."));
        Debug::_assert(token != SyntaxKind::Identifier, S("Invalid token. Use 'createIdentifier' to create identifiers"));
        //auto node = createBaseTokenNode<Token<TKind>>(token);
        auto node = createBaseNode<Node>(token);
        auto transformFlags = TransformFlags::None;
        switch (token)
        {
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
        if (!!transformFlags)
        {
            node->transformFlags |= transformFlags;
        }
        return node;
    }

    auto NodeFactory::createQualifiedName(EntityName left, Identifier right) -> QualifiedName
    {
        auto node = createBaseNode<QualifiedName>(SyntaxKind::QualifiedName);
        node->left = left;
        node->right = asName(right);
        node->transformFlags |=
            propagateChildFlags(node->left) |
            propagateIdentifierNameFlags(node->right);
        return node;
    }

    auto NodeFactory::createComputedPropertyName(Expression expression) -> ComputedPropertyName
    {
        auto node = createBaseNode<ComputedPropertyName>(SyntaxKind::ComputedPropertyName);
        node->expression = parenthesizerRules.parenthesizeExpressionOfComputedPropertyName(expression);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsES2015 |
            TransformFlags::ContainsComputedPropertyName;
        return node;
    }

    auto NodeFactory::createTypeParameterDeclaration(Identifier name, TypeNode constraint, TypeNode defaultType) -> TypeParameterDeclaration
    {
        auto node = createBaseNamedDeclaration<TypeParameterDeclaration>(
            SyntaxKind::TypeParameter,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            name);
        node->constraint = constraint;
        node->_default = defaultType;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    auto NodeFactory::createParameterDeclaration(
        DecoratorsArray decorators, 
        ModifiersArray modifiers, 
        DotDotDotToken dotDotDotToken, 
        BindingName name, 
        QuestionToken questionToken, 
        TypeNode type, 
        Expression initializer) -> ParameterDeclaration
    {
        auto node = createBaseVariableLikeDeclaration<ParameterDeclaration>(
            SyntaxKind::Parameter,
            decorators,
            modifiers,
            name,
            type,
            initializer ? parenthesizerRules.parenthesizeExpressionForDisallowedComma(initializer) : undefined
        );
        node->dotDotDotToken = dotDotDotToken;
        node->questionToken = questionToken;
        if (isThisIdentifier(node->name)) {
            node->transformFlags = TransformFlags::ContainsTypeScript;
        }
        else {
            node->transformFlags |=
                propagateChildFlags(node->dotDotDotToken) |
                propagateChildFlags(node->questionToken);
            if (questionToken) node->transformFlags |= TransformFlags::ContainsTypeScript;
            if (!!(modifiersToFlags(node->modifiers) & ModifierFlags::ParameterPropertyModifier)) node->transformFlags |= TransformFlags::ContainsTypeScriptClassSyntax;
            if (!!initializer || !!dotDotDotToken) node->transformFlags |= TransformFlags::ContainsES2015;
        }
        return node;
    }

    // @api
    auto NodeFactory::createDecorator(Expression expression) -> Decorator {
        auto node = createBaseNode<Decorator>(SyntaxKind::Decorator);
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsTypeScript |
            TransformFlags::ContainsTypeScriptClassSyntax;
        return node;
    }

    //
    // Type Elements
    //

    // @api
    auto NodeFactory::createPropertySignature(
        ModifiersArray modifiers,
        PropertyName name,
        QuestionToken questionToken,
        TypeNode type
    ) -> PropertySignature {
        auto node = createBaseNamedDeclaration<PropertySignature>(
            SyntaxKind::PropertySignature,
            /*decorators*/ undefined,
            modifiers,
            name
        );
        node->type = type;
        node->questionToken = questionToken;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createPropertyDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        PropertyName name,
        Node questionOrExclamationToken,
        TypeNode type,
        Expression initializer
    ) -> PropertyDeclaration {
        auto node = createBaseVariableLikeDeclaration<PropertyDeclaration>(
            SyntaxKind::PropertyDeclaration,
            decorators,
            modifiers,
            name,
            type,
            initializer
        );
        node->questionToken = questionOrExclamationToken && isQuestionToken(questionOrExclamationToken) ? questionOrExclamationToken : undefined;
        node->exclamationToken = questionOrExclamationToken && isExclamationToken(questionOrExclamationToken) ? questionOrExclamationToken : undefined;
        node->transformFlags |=
            propagateChildFlags(node->questionToken) |
            propagateChildFlags(node->exclamationToken) |
            TransformFlags::ContainsClassFields;
        if (isComputedPropertyName(node->name) || (hasStaticModifier(node) && node->initializer)) {
            node->transformFlags |= TransformFlags::ContainsTypeScriptClassSyntax;
        }
        if (questionOrExclamationToken || modifiersToFlags(node->modifiers) & ModifierFlags::Ambient) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createMethodSignature(
        ModifiersArray modifiers,
        PropertyName name,
        QuestionToken questionToken,
        NodeArray<TypeParameterDeclaration> typeParameters,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type
    ) -> MethodSignature {
        auto node = createBaseSignatureDeclaration<MethodSignature>(
            SyntaxKind::MethodSignature,
            /*decorators*/ undefined,
            modifiers,
            name,
            typeParameters,
            parameters,
            type
        );
        node->questionToken = questionToken;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createMethodDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        AsteriskToken asteriskToken,
        PropertyName name,
        QuestionToken questionToken,
        NodeArray<TypeParameterDeclaration> typeParameters,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type,
        Block body
    ) -> MethodDeclaration {
        auto node = createBaseFunctionLikeDeclaration<MethodDeclaration>(
            SyntaxKind::MethodDeclaration,
            decorators,
            modifiers,
            name,
            typeParameters,
            parameters,
            type,
            body
        );
        node->asteriskToken = asteriskToken;
        node->questionToken = questionToken;
        node->transformFlags |=
            propagateChildFlags(node->asteriskToken) |
            propagateChildFlags(node->questionToken) |
            TransformFlags::ContainsES2015;
        if (questionToken) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        if (!!(modifiersToFlags(node->modifiers) & ModifierFlags::Async)) {
            if (asteriskToken) {
                node->transformFlags |= TransformFlags::ContainsES2018;
            }
            else {
                node->transformFlags |= TransformFlags::ContainsES2017;
            }
        }
        else if (asteriskToken) {
            node->transformFlags |= TransformFlags::ContainsGenerator;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createConstructorDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        NodeArray<ParameterDeclaration> parameters,
        Block body
    ) -> ConstructorDeclaration {
        auto node = createBaseFunctionLikeDeclaration<ConstructorDeclaration>(
            SyntaxKind::Constructor,
            decorators,
            modifiers,
            /*name*/ undefined,
            /*typeParameters*/ undefined,
            parameters,
            /*type*/ undefined,
            body
        );
        node->transformFlags |= TransformFlags::ContainsES2015;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createGetAccessorDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        PropertyName name,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type,
        Block body
    ) -> GetAccessorDeclaration {
        return createBaseFunctionLikeDeclaration<GetAccessorDeclaration>(
            SyntaxKind::GetAccessor,
            decorators,
            modifiers,
            name,
            /*typeParameters*/ undefined,
            parameters,
            type,
            body
        );
    }

    // @api
    

    // @api
    auto NodeFactory::createSetAccessorDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        PropertyName name,
        NodeArray<ParameterDeclaration> parameters,
        Block body
    ) -> SetAccessorDeclaration {
        return createBaseFunctionLikeDeclaration<SetAccessorDeclaration>(
            SyntaxKind::SetAccessor,
            decorators,
            modifiers,
            name,
            /*typeParameters*/ undefined,
            parameters,
            /*type*/ undefined,
            body
        );
    }

    // @api
    

    // @api
    auto NodeFactory::createCallSignature(
        NodeArray<TypeParameterDeclaration> typeParameters,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type
    ) -> CallSignatureDeclaration {
        auto node = createBaseSignatureDeclaration<CallSignatureDeclaration>(
            SyntaxKind::CallSignature,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            /*name*/ undefined,
            typeParameters,
            parameters,
            type
        );
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createConstructSignature(
        NodeArray<TypeParameterDeclaration> typeParameters,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type
    ) -> ConstructSignatureDeclaration {
        auto node = createBaseSignatureDeclaration<ConstructSignatureDeclaration>(
            SyntaxKind::ConstructSignature,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            /*name*/ undefined,
            typeParameters,
            parameters,
            type
        );
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createIndexSignature(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type
    ) -> IndexSignatureDeclaration {
        auto node = createBaseSignatureDeclaration<IndexSignatureDeclaration>(
            SyntaxKind::IndexSignature,
            decorators,
            modifiers,
            /*name*/ undefined,
            /*typeParameters*/ undefined,
            parameters,
            type
        );
        node->transformFlags = TransformFlags::ContainsTypeScript;
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
    auto NodeFactory::createFunctionTypeNode(
        NodeArray<TypeParameterDeclaration> typeParameters,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type
    ) -> FunctionTypeNode {
        auto node = createBaseSignatureDeclaration<FunctionTypeNode>(
            SyntaxKind::FunctionType,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            /*name*/ undefined,
            typeParameters,
            parameters,
            type
        );
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createTypeQueryNode(EntityName exprName) -> TypeQueryNode {
        auto node = createBaseNode<TypeQueryNode>(SyntaxKind::TypeQuery);
        node->exprName = exprName;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createTypeLiteralNode(NodeArray<TypeElement> members) -> TypeLiteralNode {
        auto node = createBaseNode<TypeLiteralNode>(SyntaxKind::TypeLiteral);
        node->members = createNodeArray(members);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createArrayTypeNode(TypeNode elementType) -> ArrayTypeNode {
        auto node = createBaseNode<ArrayTypeNode>(SyntaxKind::ArrayType);
        node->elementType = parenthesizerRules.parenthesizeElementTypeOfArrayType(elementType);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createTupleTypeNode(NodeArray</*TypeNode | NamedTupleMember*/ Node> elements) -> TupleTypeNode {
        auto node = createBaseNode<TupleTypeNode>(SyntaxKind::TupleType);
        node->elements = createNodeArray(elements);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createNamedTupleMember(DotDotDotToken dotDotDotToken, Identifier name, QuestionToken questionToken, TypeNode type) -> NamedTupleMember {
        auto node = createBaseNode<NamedTupleMember>(SyntaxKind::NamedTupleMember);
        node->dotDotDotToken = dotDotDotToken;
        node->name = name;
        node->questionToken = questionToken;
        node->type = type;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createOptionalTypeNode(TypeNode type) -> OptionalTypeNode {
        auto node = createBaseNode<OptionalTypeNode>(SyntaxKind::OptionalType);
        node->type = parenthesizerRules.parenthesizeElementTypeOfArrayType(type);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createRestTypeNode(TypeNode type) -> RestTypeNode {
        auto node = createBaseNode<RestTypeNode>(SyntaxKind::RestType);
        node->type = type;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createUnionTypeNode(NodeArray<TypeNode> types) -> UnionTypeNode {
        auto node = createBaseNode<UnionTypeNode>(SyntaxKind::UnionType);
        node->types = parenthesizerRules.parenthesizeConstituentTypesOfUnionOrIntersectionType(types);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createIntersectionTypeNode(NodeArray<TypeNode> types) -> IntersectionTypeNode {
        auto node = createBaseNode<IntersectionTypeNode>(SyntaxKind::IntersectionType);
        node->types = parenthesizerRules.parenthesizeConstituentTypesOfUnionOrIntersectionType(types);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createConditionalTypeNode(TypeNode checkType, TypeNode extendsType, TypeNode trueType, TypeNode falseType) -> ConditionalTypeNode {
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
    auto NodeFactory::createInferTypeNode(TypeParameterDeclaration typeParameter) -> InferTypeNode {
        auto node = createBaseNode<InferTypeNode>(SyntaxKind::InferType);
        node->typeParameter = typeParameter;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createTemplateLiteralType(TemplateHead head, NodeArray<TemplateLiteralTypeSpan> templateSpans) -> TemplateLiteralTypeNode {
        auto node = createBaseNode<TemplateLiteralTypeNode>(SyntaxKind::TemplateLiteralType);
        node->head = head;
        node->templateSpans = createNodeArray(templateSpans);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createImportTypeNode(TypeNode argument, EntityName qualifier, NodeArray<TypeNode> typeArguments, boolean isTypeOf) -> ImportTypeNode {
        auto node = createBaseNode<ImportTypeNode>(SyntaxKind::ImportType);
        node->argument = argument;
        node->qualifier = qualifier;
        node->typeArguments = typeArguments ? parenthesizerRules.parenthesizeTypeArguments(typeArguments) : undefined;
        node->isTypeOf = isTypeOf;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createParenthesizedType(TypeNode type) -> ParenthesizedTypeNode {
        auto node = createBaseNode<ParenthesizedTypeNode>(SyntaxKind::ParenthesizedType);
        node->type = type;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createThisTypeNode() -> ThisTypeNode {
        auto node = createBaseNode<ThisTypeNode>(SyntaxKind::ThisType);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    auto NodeFactory::createTypeOperatorNode(SyntaxKind _operator, TypeNode type) -> TypeOperatorNode {
        auto node = createBaseNode<TypeOperatorNode>(SyntaxKind::TypeOperator);
        node->_operator = _operator;
        node->type = parenthesizerRules.parenthesizeMemberOfElementType(type);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createIndexedAccessTypeNode(TypeNode objectType, TypeNode indexType) -> IndexedAccessTypeNode {
        auto node = createBaseNode<IndexedAccessTypeNode>(SyntaxKind::IndexedAccessType);
        node->objectType = parenthesizerRules.parenthesizeMemberOfElementType(objectType);
        node->indexType = indexType;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createMappedTypeNode(Node readonlyToken, TypeParameterDeclaration typeParameter, TypeNode nameType, Node questionToken, TypeNode type) -> MappedTypeNode {
        auto node = createBaseNode<MappedTypeNode>(SyntaxKind::MappedType);
        node->readonlyToken = readonlyToken;
        node->typeParameter = typeParameter;
        node->nameType = nameType;
        node->questionToken = questionToken;
        node->type = type;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createLiteralTypeNode(LiteralTypeNode literal) -> LiteralTypeNode {
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
    auto NodeFactory::createObjectBindingPattern(NodeArray<BindingElement> elements) -> ObjectBindingPattern {
        auto node = createBaseNode<ObjectBindingPattern>(SyntaxKind::ObjectBindingPattern);
        node->elements = createNodeArray(elements);
        node->transformFlags |=
            propagateChildrenFlags(node->elements) |
            TransformFlags::ContainsES2015 |
            TransformFlags::ContainsBindingPattern;
        if (!!(node->transformFlags & TransformFlags::ContainsRestOrSpread)) {
            node->transformFlags |=
                TransformFlags::ContainsES2018 |
                TransformFlags::ContainsObjectRestOrSpread;
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
            propagateChildrenFlags(node->elements) |
            TransformFlags::ContainsES2015 |
            TransformFlags::ContainsBindingPattern;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createBindingElement(DotDotDotToken dotDotDotToken, PropertyName propertyName, BindingName name, Expression initializer) -> BindingElement {
        auto node = createBaseBindingLikeDeclaration<BindingElement>(
            SyntaxKind::BindingElement,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            name,
            initializer
        );
        node->propertyName = asName(propertyName);
        node->dotDotDotToken = dotDotDotToken;
        node->transformFlags |=
            propagateChildFlags(node->dotDotDotToken) |
            TransformFlags::ContainsES2015;
        if (node->propertyName) {
            node->transformFlags |= isIdentifier(node->propertyName) ?
                propagateIdentifierNameFlags(node->propertyName) :
                propagateChildFlags(node->propertyName);
        }
        if (dotDotDotToken) node->transformFlags |= TransformFlags::ContainsRestOrSpread;
        return node;
    }

    // @api
    

    //
    // Expression
    //

    // @api
    auto NodeFactory::createArrayLiteralExpression(NodeArray<Expression> elements, boolean multiLine) -> ArrayLiteralExpression {
        auto node = createBaseExpression<ArrayLiteralExpression>(SyntaxKind::ArrayLiteralExpression);
        node->elements = parenthesizerRules.parenthesizeExpressionsOfCommaDelimitedList(createNodeArray(elements));
        node->multiLine = multiLine;
        node->transformFlags |= propagateChildrenFlags(node->elements);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createObjectLiteralExpression(NodeArray<ObjectLiteralElementLike> properties, boolean multiLine) -> ObjectLiteralExpression {
        auto node = createBaseExpression<ObjectLiteralExpression>(SyntaxKind::ObjectLiteralExpression);
        node->properties = createNodeArray(properties);
        node->multiLine = multiLine;
        node->transformFlags |= propagateChildrenFlags(node->properties);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createPropertyAccessExpression(Expression expression, MemberName name) -> PropertyAccessExpression {
        auto node = createBaseExpression<PropertyAccessExpression>(SyntaxKind::PropertyAccessExpression);
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->name = asName(name);
        node->transformFlags =
            propagateChildFlags(node->expression) |
            (isIdentifier(node->name) ?
                propagateIdentifierNameFlags(node->name) :
                propagateChildFlags(node->name));
        if (isSuperKeyword(expression)) {
            // super method calls require a lexical 'this'
            // super method calls require 'super' hoisting in ES2017 and ES2018 async functions and async generators
            node->transformFlags |=
                TransformFlags::ContainsES2017 |
                TransformFlags::ContainsES2018;
        }
        return node;
    }

    // @api
    auto NodeFactory::createPropertyAccessChain(Expression expression, QuestionDotToken questionDotToken, MemberName name) -> PropertyAccessChain {
        auto node = createBaseExpression<PropertyAccessChain>(SyntaxKind::PropertyAccessExpression);
        node->flags |= NodeFlags::OptionalChain;
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->questionDotToken = questionDotToken;
        node->name = asName(name);
        node->transformFlags |=
            TransformFlags::ContainsES2020 |
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->questionDotToken) |
            (isIdentifier(node->name) ?
                propagateIdentifierNameFlags(node->name) :
                propagateChildFlags(node->name));
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createElementAccessExpression(Expression expression, Expression index) -> ElementAccessExpression {
        auto node = createBaseExpression<ElementAccessExpression>(SyntaxKind::ElementAccessExpression);
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->argumentExpression = asExpression(index);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->argumentExpression);
        if (isSuperKeyword(expression)) {
            // super method calls require a lexical 'this'
            // super method calls require 'super' hoisting in ES2017 and ES2018 async functions and async generators
            node->transformFlags |=
                TransformFlags::ContainsES2017 |
                TransformFlags::ContainsES2018;
        }
        return node;
    }

    // @api
    auto NodeFactory::createElementAccessChain(Expression expression, QuestionDotToken questionDotToken, Expression index) -> ElementAccessChain {
        auto node = createBaseExpression<ElementAccessChain>(SyntaxKind::ElementAccessExpression);
        node->flags |= NodeFlags::OptionalChain;
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->questionDotToken = questionDotToken;
        node->argumentExpression = asExpression(index);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->questionDotToken) |
            propagateChildFlags(node->argumentExpression) |
            TransformFlags::ContainsES2020;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createCallExpression(Expression expression, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray) -> CallExpression {
        auto node = createBaseExpression<CallExpression>(SyntaxKind::CallExpression);
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->typeArguments = asNodeArray(typeArguments);
        node->arguments = parenthesizerRules.parenthesizeExpressionsOfCommaDelimitedList(createNodeArray(argumentsArray));
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildrenFlags(node->typeArguments) |
            propagateChildrenFlags(node->arguments);
        if (node->typeArguments) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        if (isImportKeyword(node->expression)) {
            node->transformFlags |= TransformFlags::ContainsDynamicImport;
        }
        else if (isSuperProperty(node->expression)) {
            node->transformFlags |= TransformFlags::ContainsLexicalThis;
        }
        return node;
    }

    // @api
    auto NodeFactory::updateCallExpression(CallExpression node, Expression expression, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray) -> CallExpression {
        return node->expression != expression
            || node->typeArguments != typeArguments
            || node->arguments != argumentsArray
            ? update(createCallExpression(expression, typeArguments, argumentsArray), node)
            : node;
    }

    // @api
    auto NodeFactory::createCallChain(Expression expression, QuestionDotToken questionDotToken, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray) -> CallChain
    {
        auto node = createBaseExpression<CallChain>(SyntaxKind::CallExpression);
        node->flags |= NodeFlags::OptionalChain;
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->questionDotToken = questionDotToken;
        node->typeArguments = asNodeArray(typeArguments);
        node->arguments = parenthesizerRules.parenthesizeExpressionsOfCommaDelimitedList(createNodeArray(argumentsArray));
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->questionDotToken) |
            propagateChildrenFlags(node->typeArguments) |
            propagateChildrenFlags(node->arguments) |
            TransformFlags::ContainsES2020;
        if (node->typeArguments) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        if (isSuperProperty(node->expression)) {
            node->transformFlags |= TransformFlags::ContainsLexicalThis;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createNewExpression(Expression expression, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray) -> NewExpression
    {
        auto node = createBaseExpression<NewExpression>(SyntaxKind::NewExpression);
        node->expression = parenthesizerRules.parenthesizeExpressionOfNew(expression);
        node->typeArguments = asNodeArray(typeArguments);
        node->arguments = argumentsArray ? parenthesizerRules.parenthesizeExpressionsOfCommaDelimitedList(argumentsArray) : undefined;
        node->transformFlags |=
            propagateChildFlags(node->expression) |
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
    auto NodeFactory::createTaggedTemplateExpression(Expression tag, NodeArray<TypeNode> typeArguments, TemplateLiteral _template) -> TaggedTemplateExpression {
        auto node = createBaseExpression<TaggedTemplateExpression>(SyntaxKind::TaggedTemplateExpression);
        node->tag = parenthesizerRules.parenthesizeLeftSideOfAccess(tag);
        node->typeArguments = asNodeArray(typeArguments);
        node->_template = _template;
        node->transformFlags |=
            propagateChildFlags(node->tag) |
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
    auto NodeFactory::createTypeAssertion(TypeNode type, Expression expression) -> TypeAssertion {
        auto node = createBaseExpression<TypeAssertion>(SyntaxKind::TypeAssertionExpression);
        node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
        node->type = type;
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->type) |
            TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createParenthesizedExpression(Expression expression) -> ParenthesizedExpression {
        auto node = createBaseExpression<ParenthesizedExpression>(SyntaxKind::ParenthesizedExpression);
        node->expression = expression;
        node->transformFlags = propagateChildFlags(node->expression);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createFunctionExpression(
        ModifiersArray modifiers,
        AsteriskToken asteriskToken,
        Identifier name,
        NodeArray<TypeParameterDeclaration> typeParameters,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type,
        Block body
    ) -> FunctionExpression {
        auto node = createBaseFunctionLikeDeclaration<FunctionExpression>(
            SyntaxKind::FunctionExpression,
            /*decorators*/ undefined,
            modifiers,
            name,
            typeParameters,
            parameters,
            type,
            body
        );
        node->asteriskToken = asteriskToken;
        node->transformFlags |= propagateChildFlags(node->asteriskToken);
        if (node->typeParameters) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        if (!!(modifiersToFlags(node->modifiers) & ModifierFlags::Async)) {
            if (node->asteriskToken) {
                node->transformFlags |= TransformFlags::ContainsES2018;
            }
            else {
                node->transformFlags |= TransformFlags::ContainsES2017;
            }
        }
        else if (node->asteriskToken) {
            node->transformFlags |= TransformFlags::ContainsGenerator;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createArrowFunction(
        ModifiersArray modifiers,
        NodeArray<TypeParameterDeclaration> typeParameters,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type,
        EqualsGreaterThanToken equalsGreaterThanToken,
        ConciseBody body
    ) -> ArrowFunction {
        auto node = createBaseFunctionLikeDeclaration<ArrowFunction>(
            SyntaxKind::ArrowFunction,
            /*decorators*/ undefined,
            modifiers,
            /*name*/ undefined,
            typeParameters,
            parameters,
            type,
            parenthesizerRules.parenthesizeConciseBodyOfArrowFunction(body)
        );
        node->equalsGreaterThanToken = equalsGreaterThanToken ? equalsGreaterThanToken : createToken(SyntaxKind::EqualsGreaterThanToken);
        node->transformFlags |=
            propagateChildFlags(node->equalsGreaterThanToken) |
            TransformFlags::ContainsES2015;
        if (!!(modifiersToFlags(node->modifiers) & ModifierFlags::Async)) {
            node->transformFlags |= TransformFlags::ContainsES2017;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createDeleteExpression(Expression expression) -> DeleteExpression {
        auto node = createBaseExpression<DeleteExpression>(SyntaxKind::DeleteExpression);
        node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
        node->transformFlags |= propagateChildFlags(node->expression);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createTypeOfExpression(Expression expression) -> TypeOfExpression {
        auto node = createBaseExpression<TypeOfExpression>(SyntaxKind::TypeOfExpression);
        node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
        node->transformFlags |= propagateChildFlags(node->expression);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createVoidExpression(Expression expression) -> VoidExpression {
        auto node = createBaseExpression<VoidExpression>(SyntaxKind::VoidExpression);
        node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
        node->transformFlags |= propagateChildFlags(node->expression);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createAwaitExpression(Expression expression) -> AwaitExpression {
        auto node = createBaseExpression<AwaitExpression>(SyntaxKind::AwaitExpression);
        node->expression = parenthesizerRules.parenthesizeOperandOfPrefixUnary(expression);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsES2017 |
            TransformFlags::ContainsES2018 |
            TransformFlags::ContainsAwait;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createPrefixUnaryExpression(PrefixUnaryOperator _operator, Expression operand) -> PrefixUnaryExpression {
        auto node = createBaseExpression<PrefixUnaryExpression>(SyntaxKind::PrefixUnaryExpression);
        node->_operator = _operator;
        node->operand = parenthesizerRules.parenthesizeOperandOfPrefixUnary(operand);
        node->transformFlags |= propagateChildFlags(node->operand);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createPostfixUnaryExpression(Expression operand, PostfixUnaryOperator _operator) -> PostfixUnaryExpression {
        auto node = createBaseExpression<PostfixUnaryExpression>(SyntaxKind::PostfixUnaryExpression);
        node->_operator = _operator;
        node->operand = parenthesizerRules.parenthesizeOperandOfPostfixUnary(operand);
        node->transformFlags = propagateChildFlags(node->operand);
        return node;
    }

    // @api

    // @api
    auto NodeFactory::createBinaryExpression(Expression left, Node _operator, Expression right) -> BinaryExpression {
        auto node = createBaseExpression<BinaryExpression>(SyntaxKind::BinaryExpression);
        auto operatorToken = asToken(_operator);
        auto operatorKind = operatorToken->kind;
        node->left = parenthesizerRules.parenthesizeLeftSideOfBinary(operatorKind, left);
        node->operatorToken = operatorToken;
        node->right = parenthesizerRules.parenthesizeRightSideOfBinary(operatorKind, node->left, right);
        node->transformFlags |=
            propagateChildFlags(node->left) |
            propagateChildFlags(node->operatorToken) |
            propagateChildFlags(node->right);
        if (operatorKind == SyntaxKind::QuestionQuestionToken) {
            node->transformFlags |= TransformFlags::ContainsES2020;
        }
        else if (operatorKind == SyntaxKind::EqualsToken) {
            if (isObjectLiteralExpression(node->left)) {
                node->transformFlags |=
                    TransformFlags::ContainsES2015 |
                    TransformFlags::ContainsES2018 |
                    TransformFlags::ContainsDestructuringAssignment |
                    propagateAssignmentPatternFlags(node->left);
            }
            else if (isArrayLiteralExpression(node->left)) {
                node->transformFlags |=
                    TransformFlags::ContainsES2015 |
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
        return node;
    }

    auto propagateAssignmentPatternFlags(AssignmentPattern node) -> TransformFlags {
        if (!!(node->transformFlags & TransformFlags::ContainsObjectRestOrSpread)) return TransformFlags::ContainsObjectRestOrSpread;
        if (!!(node->transformFlags & TransformFlags::ContainsES2018)) {
            // check for nested spread assignments, otherwise '{ x: { a, ...b } = foo } = c'
            // will not be correctly interpreted by the ES2018 transformer
            for (auto element : getElementsOfBindingOrAssignmentPattern(node)) {
                auto target = getTargetOfBindingOrAssignmentElement(element);
                if (target && isAssignmentPattern(target)) {
                    if (!!(target->transformFlags & TransformFlags::ContainsObjectRestOrSpread)) {
                        return TransformFlags::ContainsObjectRestOrSpread;
                    }
                    if (!!(target->transformFlags & TransformFlags::ContainsES2018)) {
                        auto flags = propagateAssignmentPatternFlags(target);
                        if (!!flags) return flags;
                    }
                }
            }
        }
        return TransformFlags::None;
    }

    // @api
    

    // @api
    auto NodeFactory::createConditionalExpression(Expression condition, QuestionToken questionToken, Expression whenTrue, ColonToken colonToken, Expression whenFalse) -> ConditionalExpression {
        auto node = createBaseExpression<ConditionalExpression>(SyntaxKind::ConditionalExpression);
        node->condition = parenthesizerRules.parenthesizeConditionOfConditionalExpression(condition);
        node->questionToken = questionToken ? questionToken : createToken(SyntaxKind::QuestionToken);
        node->whenTrue = parenthesizerRules.parenthesizeBranchOfConditionalExpression(whenTrue);
        node->colonToken = colonToken ? colonToken : createToken(SyntaxKind::ColonToken);
        node->whenFalse = parenthesizerRules.parenthesizeBranchOfConditionalExpression(whenFalse);
        node->transformFlags |=
            propagateChildFlags(node->condition) |
            propagateChildFlags(node->questionToken) |
            propagateChildFlags(node->whenTrue) |
            propagateChildFlags(node->colonToken) |
            propagateChildFlags(node->whenFalse);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createTemplateExpression(TemplateHead head, NodeArray<TemplateSpan> templateSpans) -> TemplateExpression {
        auto node = createBaseExpression<TemplateExpression>(SyntaxKind::TemplateExpression);
        node->head = head;
        node->templateSpans = createNodeArray(templateSpans);
        node->transformFlags |=
            propagateChildFlags(node->head) |
            propagateChildrenFlags(node->templateSpans) |
            TransformFlags::ContainsES2015;
        return node;
    }

    // @api
    

    auto NodeFactory::createTemplateLiteralLikeNodeChecked(kind: TemplateLiteralToken["kind"], text: string, rawText: string, templateFlags = TokenFlags.None) {
        Debug.assert(!(templateFlags & ~TokenFlags.TemplateLiteralLikeFlags), "Unsupported template flags.");
        // NOTE: without the assignment to `undefined`, we don't narrow the initial type of `cooked`.
        // eslint-disable-next-line no-undef-init
        let cooked: string | object = undefined;
        if (rawText !== undefined && rawText !== text) {
            cooked = getCookedText(kind, rawText);
            if (typeof cooked == "object") {
                return Debug.fail("Invalid raw text");
            }
        }
        if (text == undefined) {
            if (cooked == undefined) {
                return Debug.fail("Arguments 'text' and 'rawText' may not both be undefined.");
            }
            text = cooked;
        }
        else if (cooked !== undefined) {
            Debug.assert(text == cooked, "Expected argument 'text' to be the normalized (i.e. 'cooked') version of argument 'rawText'.");
        }
        return createTemplateLiteralLikeNode(kind, text, rawText, templateFlags);
    }

    // @api
    auto NodeFactory::createTemplateLiteralLikeNode(kind: TemplateLiteralToken["kind"], text: string, rawText: string, templateFlags: TokenFlags) {
        auto node = createBaseToken<TemplateLiteralLikeNode>(kind);
        node->text = text;
        node->rawText = rawText;
        node->templateFlags = templateFlags! & TokenFlags.TemplateLiteralLikeFlags;
        node->transformFlags |= TransformFlags::ContainsES2015;
        if (node->templateFlags) {
            node->transformFlags |= TransformFlags::ContainsES2018;
        }
        return node;
    }

    // @api
    auto NodeFactory::createTemplateHead(text: string, rawText?: string, templateFlags?: TokenFlags) {
        return <TemplateHead>createTemplateLiteralLikeNodeChecked(SyntaxKind::TemplateHead, text, rawText, templateFlags);
    }

    // @api
    auto NodeFactory::createTemplateMiddle(text: string, rawText?: string, templateFlags?: TokenFlags) {
        return <TemplateMiddle>createTemplateLiteralLikeNodeChecked(SyntaxKind::TemplateMiddle, text, rawText, templateFlags);
    }

    // @api
    auto NodeFactory::createTemplateTail(text: string, rawText?: string, templateFlags?: TokenFlags) {
        return <TemplateTail>createTemplateLiteralLikeNodeChecked(SyntaxKind::TemplateTail, text, rawText, templateFlags);
    }

    // @api
    auto NodeFactory::createNoSubstitutionTemplateLiteral(text: string, rawText?: string, templateFlags?: TokenFlags) {
        return <NoSubstitutionTemplateLiteral>createTemplateLiteralLikeNodeChecked(SyntaxKind::NoSubstitutionTemplateLiteral, text, rawText, templateFlags);
    }

    // @api
    auto NodeFactory::createYieldExpression(AsteriskToken asteriskToken, Expression expression) -> YieldExpression {
        Debug.assert(!asteriskToken || !!expression, "A `YieldExpression` with an asteriskToken must have an expression.");
        auto node = createBaseExpression<YieldExpression>(SyntaxKind::YieldExpression);
        node->expression = expression && parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
        node->asteriskToken = asteriskToken;
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->asteriskToken) |
            TransformFlags::ContainsES2015 |
            TransformFlags::ContainsES2018 |
            TransformFlags::ContainsYield;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createSpreadElement(Expression expression) {
        auto node = createBaseExpression<SpreadElement>(SyntaxKind::SpreadElement);
        node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsES2015 |
            TransformFlags::ContainsRestOrSpread;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createClassExpression(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        name: string | Identifier,
        NodeArray<TypeParameterDeclaration> typeParameters,
        heritageClauses: HeritageClause[],
        members: ClassElement[]
    ) {
        auto node = createBaseClassLikeDeclaration<ClassExpression>(
            SyntaxKind::ClassExpression,
            decorators,
            modifiers,
            name,
            typeParameters,
            heritageClauses,
            members
        );
        node->transformFlags |= TransformFlags::ContainsES2015;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createOmittedExpression() {
        return createBaseExpression<OmittedExpression>(SyntaxKind::OmittedExpression);
    }

    // @api
    auto NodeFactory::createExpressionWithTypeArguments(Expression expression, typeArguments: TypeNode[]) {
        auto node = createBaseNode<ExpressionWithTypeArguments>(SyntaxKind::ExpressionWithTypeArguments);
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->typeArguments = typeArguments && parenthesizerRules.parenthesizeTypeArguments(typeArguments);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildrenFlags(node->typeArguments) |
            TransformFlags::ContainsES2015;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createAsExpression(Expression expression, TypeNode type) {
        auto node = createBaseExpression<AsExpression>(SyntaxKind::AsExpression);
        node->expression = expression;
        node->type = type;
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->type) |
            TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createNonNullExpression(Expression expression) {
        auto node = createBaseExpression<NonNullExpression>(SyntaxKind::NonNullExpression);
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api

    // @api
    auto NodeFactory::createNonNullChain(Expression expression) {
        auto node = createBaseExpression<NonNullChain>(SyntaxKind::NonNullExpression);
        node->flags |= NodeFlags::OptionalChain;
        node->expression = parenthesizerRules.parenthesizeLeftSideOfAccess(expression);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createMetaProperty(keywordToken: MetaProperty["keywordToken"], name: Identifier) {
        auto node = createBaseExpression<MetaProperty>(SyntaxKind::MetaProperty);
        node->keywordToken = keywordToken;
        node->name = name;
        node->transformFlags |= propagateChildFlags(node->name);
        switch (keywordToken) {
            case SyntaxKind::NewKeyword:
                node->transformFlags |= TransformFlags::ContainsES2015;
                break;
            case SyntaxKind::ImportKeyword:
                node->transformFlags |= TransformFlags::ContainsESNext;
                break;
            default:
                return Debug.assertNever(keywordToken);
        }
        return node;
    }

    // @api
    

    //
    // Misc
    //

    // @api
    auto NodeFactory::createTemplateSpan(Expression expression, literal: TemplateMiddle | TemplateTail) {
        auto node = createBaseNode<TemplateSpan>(SyntaxKind::TemplateSpan);
        node->expression = expression;
        node->literal = literal;
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->literal) |
            TransformFlags::ContainsES2015;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createSemicolonClassElement() {
        auto node = createBaseNode<SemicolonClassElement>(SyntaxKind::SemicolonClassElement);
        node->transformFlags |= TransformFlags::ContainsES2015;
        return node;
    }

    //
    // Element
    //

    // @api
    auto NodeFactory::createBlock(statements: Statement[], multiLine?: boolean) -> Block {
        auto node = createBaseNode<Block>(SyntaxKind::Block);
        node->statements = createNodeArray(statements);
        node->multiLine = multiLine;
        node->transformFlags |= propagateChildrenFlags(node->statements);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createVariableStatement(ModifiersArray modifiers, declarationList: VariableDeclarationList | VariableDeclaration[]) {
        auto node = createBaseDeclaration<VariableStatement>(SyntaxKind::VariableStatement, /*decorators*/ undefined, modifiers);
        node->declarationList = isArray(declarationList) ? createVariableDeclarationList(declarationList) : declarationList;
        node->transformFlags |=
            propagateChildFlags(node->declarationList);
        if (modifiersToFlags(node->modifiers) & ModifierFlags::Ambient) {
            node->transformFlags = TransformFlags::ContainsTypeScript;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createEmptyStatement() {
        return createBaseNode<EmptyStatement>(SyntaxKind::EmptyStatement);
    }

    // @api
    auto NodeFactory::createExpressionStatement(Expression expression) -> ExpressionStatement {
        auto node = createBaseNode<ExpressionStatement>(SyntaxKind::ExpressionStatement);
        node->expression = parenthesizerRules.parenthesizeExpressionOfExpressionStatement(expression);
        node->transformFlags |= propagateChildFlags(node->expression);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createIfStatement(Expression expression, thenStatement: Statement, elseStatement?: Statement) {
        auto node = createBaseNode<IfStatement>(SyntaxKind::IfStatement);
        node->expression = expression;
        node->thenStatement = asEmbeddedStatement(thenStatement);
        node->elseStatement = asEmbeddedStatement(elseStatement);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->thenStatement) |
            propagateChildFlags(node->elseStatement);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createDoStatement(statement: Statement, Expression expression) {
        auto node = createBaseNode<DoStatement>(SyntaxKind::DoStatement);
        node->statement = asEmbeddedStatement(statement);
        node->expression = expression;
        node->transformFlags |=
            propagateChildFlags(node->statement) |
            propagateChildFlags(node->expression);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createWhileStatement(Expression expression, statement: Statement) {
        auto node = createBaseNode<WhileStatement>(SyntaxKind::WhileStatement);
        node->expression = expression;
        node->statement = asEmbeddedStatement(statement);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->statement);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createForStatement(initializer: ForInitializer, condition: Expression, incrementor: Expression, statement: Statement) {
        auto node = createBaseNode<ForStatement>(SyntaxKind::ForStatement);
        node->initializer = initializer;
        node->condition = condition;
        node->incrementor = incrementor;
        node->statement = asEmbeddedStatement(statement);
        node->transformFlags |=
            propagateChildFlags(node->initializer) |
            propagateChildFlags(node->condition) |
            propagateChildFlags(node->incrementor) |
            propagateChildFlags(node->statement);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createForInStatement(initializer: ForInitializer, Expression expression, statement: Statement) {
        auto node = createBaseNode<ForInStatement>(SyntaxKind::ForInStatement);
        node->initializer = initializer;
        node->expression = expression;
        node->statement = asEmbeddedStatement(statement);
        node->transformFlags |=
            propagateChildFlags(node->initializer) |
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->statement);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createForOfStatement(awaitModifier: AwaitKeyword, initializer: ForInitializer, Expression expression, statement: Statement) {
        auto node = createBaseNode<ForOfStatement>(SyntaxKind::ForOfStatement);
        node->awaitModifier = awaitModifier;
        node->initializer = initializer;
        node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
        node->statement = asEmbeddedStatement(statement);
        node->transformFlags |=
            propagateChildFlags(node->awaitModifier) |
            propagateChildFlags(node->initializer) |
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->statement) |
            TransformFlags::ContainsES2015;
        if (awaitModifier) node->transformFlags |= TransformFlags::ContainsES2018;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createContinueStatement(label?: string | Identifier) -> ContinueStatement {
        auto node = createBaseNode<ContinueStatement>(SyntaxKind::ContinueStatement);
        node->label = asName(label);
        node->transformFlags |=
            propagateChildFlags(node->label) |
            TransformFlags::ContainsHoistedDeclarationOrCompletion;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createBreakStatement(label?: string | Identifier) -> BreakStatement {
        auto node = createBaseNode<BreakStatement>(SyntaxKind::BreakStatement);
        node->label = asName(label);
        node->transformFlags |=
            propagateChildFlags(node->label) |
            TransformFlags::ContainsHoistedDeclarationOrCompletion;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createReturnStatement(expression?: Expression) -> ReturnStatement {
        auto node = createBaseNode<ReturnStatement>(SyntaxKind::ReturnStatement);
        node->expression = expression;
        // return in an ES2018 async generator must be awaited
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsES2018 |
            TransformFlags::ContainsHoistedDeclarationOrCompletion;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createWithStatement(Expression expression, statement: Statement) {
        auto node = createBaseNode<WithStatement>(SyntaxKind::WithStatement);
        node->expression = expression;
        node->statement = asEmbeddedStatement(statement);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->statement);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createSwitchStatement(Expression expression, caseBlock: CaseBlock) -> SwitchStatement {
        auto node = createBaseNode<SwitchStatement>(SyntaxKind::SwitchStatement);
        node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
        node->caseBlock = caseBlock;
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildFlags(node->caseBlock);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createLabeledStatement(label: string | Identifier, statement: Statement) {
        auto node = createBaseNode<LabeledStatement>(SyntaxKind::LabeledStatement);
        node->label = asName(label);
        node->statement = asEmbeddedStatement(statement);
        node->transformFlags |=
            propagateChildFlags(node->label) |
            propagateChildFlags(node->statement);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createThrowStatement(Expression expression) {
        auto node = createBaseNode<ThrowStatement>(SyntaxKind::ThrowStatement);
        node->expression = expression;
        node->transformFlags |= propagateChildFlags(node->expression);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createTryStatement(tryBlock: Block, catchClause: CatchClause, finallyBlock: Block) {
        auto node = createBaseNode<TryStatement>(SyntaxKind::TryStatement);
        node->tryBlock = tryBlock;
        node->catchClause = catchClause;
        node->finallyBlock = finallyBlock;
        node->transformFlags |=
            propagateChildFlags(node->tryBlock) |
            propagateChildFlags(node->catchClause) |
            propagateChildFlags(node->finallyBlock);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createDebuggerStatement() {
        return createBaseNode<DebuggerStatement>(SyntaxKind::DebuggerStatement);
    }

    // @api
    auto NodeFactory::createVariableDeclaration(name: string | BindingName, exclamationToken: ExclamationToken, TypeNode type, Expression initializer) {
        auto node = createBaseVariableLikeDeclaration<VariableDeclaration>(
            SyntaxKind::VariableDeclaration,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            name,
            type,
            initializer && parenthesizerRules.parenthesizeExpressionForDisallowedComma(initializer)
        );
        node->exclamationToken = exclamationToken;
        node->transformFlags |= propagateChildFlags(node->exclamationToken);
        if (exclamationToken) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createVariableDeclarationList(declarations: VariableDeclaration[], flags = NodeFlags::None) {
        auto node = createBaseNode<VariableDeclarationList>(SyntaxKind::VariableDeclarationList);
        node->flags |= flags & NodeFlags::BlockScoped;
        node->declarations = createNodeArray(declarations);
        node->transformFlags |=
            propagateChildrenFlags(node->declarations) |
            TransformFlags::ContainsHoistedDeclarationOrCompletion;
        if (flags & NodeFlags::BlockScoped) {
            node->transformFlags |=
                TransformFlags::ContainsES2015 |
                TransformFlags::ContainsBlockScopedBinding;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createFunctionDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        AsteriskToken asteriskToken,
        name: string | Identifier,
        NodeArray<TypeParameterDeclaration> typeParameters,
        NodeArray<ParameterDeclaration> parameters,
        TypeNode type,
        Block body
    ) {
        auto node = createBaseFunctionLikeDeclaration<FunctionDeclaration>(
            SyntaxKind::FunctionDeclaration,
            decorators,
            modifiers,
            name,
            typeParameters,
            parameters,
            type,
            body
        );
        node->asteriskToken = asteriskToken;
        if (!node->body || modifiersToFlags(node->modifiers) & ModifierFlags::Ambient) {
            node->transformFlags = TransformFlags::ContainsTypeScript;
        }
        else {
            node->transformFlags |=
                propagateChildFlags(node->asteriskToken) |
                TransformFlags::ContainsHoistedDeclarationOrCompletion;
            if (modifiersToFlags(node->modifiers) & ModifierFlags::Async) {
                if (node->asteriskToken) {
                    node->transformFlags |= TransformFlags::ContainsES2018;
                }
                else {
                    node->transformFlags |= TransformFlags::ContainsES2017;
                }
            }
            else if (node->asteriskToken) {
                node->transformFlags |= TransformFlags::ContainsGenerator;
            }
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createClassDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        name: string | Identifier,
        NodeArray<TypeParameterDeclaration> typeParameters,
        heritageClauses: HeritageClause[],
        members: ClassElement[]
    ) {
        auto node = createBaseClassLikeDeclaration<ClassDeclaration>(
            SyntaxKind::ClassDeclaration,
            decorators,
            modifiers,
            name,
            typeParameters,
            heritageClauses,
            members
        );
        if (modifiersToFlags(node->modifiers) & ModifierFlags::Ambient) {
            node->transformFlags = TransformFlags::ContainsTypeScript;
        }
        else {
            node->transformFlags |= TransformFlags::ContainsES2015;
            if (node->transformFlags & TransformFlags::ContainsTypeScriptClassSyntax) {
                node->transformFlags |= TransformFlags::ContainsTypeScript;
            }
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createInterfaceDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        name: string | Identifier,
        NodeArray<TypeParameterDeclaration> typeParameters,
        heritageClauses: HeritageClause[],
        NodeArray<TypeElement> members
    ) {
        auto node = createBaseInterfaceOrClassLikeDeclaration<InterfaceDeclaration>(
            SyntaxKind::InterfaceDeclaration,
            decorators,
            modifiers,
            name,
            typeParameters,
            heritageClauses
        );
        node->members = createNodeArray(members);
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createTypeAliasDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        name: string | Identifier,
        NodeArray<TypeParameterDeclaration> typeParameters,
        TypeNode type
    ) {
        auto node = createBaseGenericNamedDeclaration<TypeAliasDeclaration>(
            SyntaxKind::TypeAliasDeclaration,
            decorators,
            modifiers,
            name,
            typeParameters
        );
        node->type = type;
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createEnumDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        name: string | Identifier,
        members: EnumMember[]
    ) {
        auto node = createBaseNamedDeclaration<EnumDeclaration>(
            SyntaxKind::EnumDeclaration,
            decorators,
            modifiers,
            name
        );
        node->members = createNodeArray(members);
        node->transformFlags |=
            propagateChildrenFlags(node->members) |
            TransformFlags::ContainsTypeScript;
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // Enum declarations cannot contain `await`
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createModuleDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        name: ModuleName,
        body: ModuleBody,
        flags = NodeFlags::None
    ) {
        auto node = createBaseDeclaration<ModuleDeclaration>(
            SyntaxKind::ModuleDeclaration,
            decorators,
            modifiers
        );
        node->flags |= flags & (NodeFlags::Namespace | NodeFlags::NestedNamespace | NodeFlags::GlobalAugmentation);
        node->name = name;
        node->body = body;
        if (modifiersToFlags(node->modifiers) & ModifierFlags::Ambient) {
            node->transformFlags = TransformFlags::ContainsTypeScript;
        }
        else {
            node->transformFlags |=
                propagateChildFlags(node->name) |
                propagateChildFlags(node->body) |
                TransformFlags::ContainsTypeScript;
        }
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // Module declarations cannot contain `await`.
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createModuleBlock(statements: Statement[]) {
        auto node = createBaseNode<ModuleBlock>(SyntaxKind::ModuleBlock);
        node->statements = createNodeArray(statements);
        node->transformFlags |= propagateChildrenFlags(node->statements);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createCaseBlock(clauses: CaseOrDefaultClause[]) -> CaseBlock {
        auto node = createBaseNode<CaseBlock>(SyntaxKind::CaseBlock);
        node->clauses = createNodeArray(clauses);
        node->transformFlags |= propagateChildrenFlags(node->clauses);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createNamespaceExportDeclaration(name: string | Identifier) {
        auto node = createBaseNamedDeclaration<NamespaceExportDeclaration>(
            SyntaxKind::NamespaceExportDeclaration,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            name
        );
        node->transformFlags = TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createImportEqualsDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        isTypeOnly: boolean,
        name: string | Identifier,
        moduleReference: ModuleReference
    ) {
        auto node = createBaseNamedDeclaration<ImportEqualsDeclaration>(
            SyntaxKind::ImportEqualsDeclaration,
            decorators,
            modifiers,
            name
        );
        node->isTypeOnly = isTypeOnly;
        node->moduleReference = moduleReference;
        node->transformFlags |= propagateChildFlags(node->moduleReference);
        if (!isExternalModuleReference(node->moduleReference)) node->transformFlags |= TransformFlags::ContainsTypeScript;
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // Import= declaration is always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createImportDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        importClause: ImportClause,
        moduleSpecifier: Expression
    ) -> ImportDeclaration {
        auto node = createBaseDeclaration<ImportDeclaration>(
            SyntaxKind::ImportDeclaration,
            decorators,
            modifiers
        );
        node->importClause = importClause;
        node->moduleSpecifier = moduleSpecifier;
        node->transformFlags |=
            propagateChildFlags(node->importClause) |
            propagateChildFlags(node->moduleSpecifier);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createImportClause(isTypeOnly: boolean, name: Identifier, namedBindings: NamedImportBindings) -> ImportClause {
        auto node = createBaseNode<ImportClause>(SyntaxKind::ImportClause);
        node->isTypeOnly = isTypeOnly;
        node->name = name;
        node->namedBindings = namedBindings;
        node->transformFlags |=
            propagateChildFlags(node->name) |
            propagateChildFlags(node->namedBindings);
        if (isTypeOnly) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createNamespaceImport(name: Identifier) -> NamespaceImport {
        auto node = createBaseNode<NamespaceImport>(SyntaxKind::NamespaceImport);
        node->name = name;
        node->transformFlags |= propagateChildFlags(node->name);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createNamespaceExport(name: Identifier) -> NamespaceExport {
        auto node = createBaseNode<NamespaceExport>(SyntaxKind::NamespaceExport);
        node->name = name;
        node->transformFlags |=
            propagateChildFlags(node->name) |
            TransformFlags::ContainsESNext;
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createNamedImports(elements: ImportSpecifier[]) -> NamedImports {
        auto node = createBaseNode<NamedImports>(SyntaxKind::NamedImports);
        node->elements = createNodeArray(elements);
        node->transformFlags |= propagateChildrenFlags(node->elements);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createImportSpecifier(propertyName: Identifier, name: Identifier) {
        auto node = createBaseNode<ImportSpecifier>(SyntaxKind::ImportSpecifier);
        node->propertyName = propertyName;
        node->name = name;
        node->transformFlags |=
            propagateChildFlags(node->propertyName) |
            propagateChildFlags(node->name);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createExportAssignment(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        isExportEquals: boolean,
        Expression expression
    ) {
        auto node = createBaseDeclaration<ExportAssignment>(
            SyntaxKind::ExportAssignment,
            decorators,
            modifiers
        );
        node->isExportEquals = isExportEquals;
        node->expression = isExportEquals
            ? parenthesizerRules.parenthesizeRightSideOfBinary(SyntaxKind::EqualsToken, /*leftSide*/ undefined, expression)
            : parenthesizerRules.parenthesizeExpressionOfExportDefault(expression);
        node->transformFlags |= propagateChildFlags(node->expression);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createExportDeclaration(
        DecoratorsArray decorators,
        ModifiersArray modifiers,
        isTypeOnly: boolean,
        exportClause: NamedExportBindings,
        moduleSpecifier?: Expression
    ) {
        auto node = createBaseDeclaration<ExportDeclaration>(
            SyntaxKind::ExportDeclaration,
            decorators,
            modifiers
        );
        node->isTypeOnly = isTypeOnly;
        node->exportClause = exportClause;
        node->moduleSpecifier = moduleSpecifier;
        node->transformFlags |=
            propagateChildFlags(node->exportClause) |
            propagateChildFlags(node->moduleSpecifier);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createNamedExports(elements: ExportSpecifier[]) {
        auto node = createBaseNode<NamedExports>(SyntaxKind::NamedExports);
        node->elements = createNodeArray(elements);
        node->transformFlags |= propagateChildrenFlags(node->elements);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createExportSpecifier(propertyName: string | Identifier, name: string | Identifier) {
        auto node = createBaseNode<ExportSpecifier>(SyntaxKind::ExportSpecifier);
        node->propertyName = asName(propertyName);
        node->name = asName(name);
        node->transformFlags |=
            propagateChildFlags(node->propertyName) |
            propagateChildFlags(node->name);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createMissingDeclaration() {
        auto node = createBaseDeclaration<MissingDeclaration>(
            SyntaxKind::MissingDeclaration,
            /*decorators*/ undefined,
            /*modifiers*/ undefined
        );
        return node;
    }

    //
    // Module references
    //

    // @api
    auto NodeFactory::createExternalModuleReference(Expression expression) {
        auto node = createBaseNode<ExternalModuleReference>(SyntaxKind::ExternalModuleReference);
        node->expression = expression;
        node->transformFlags |= propagateChildFlags(node->expression);
        node->transformFlags &= ~TransformFlags::ContainsPossibleTopLevelAwait; // always parsed in an Await context
        return node;
    }

    // @api
    

    //
    // JSDoc
    //

    // @api
    // createJSDocAllType
    // createJSDocUnknownType
    auto NodeFactory::createJSDocPrimaryTypeWorker<T extends JSDocType>(kind: T["kind"]) {
        return createBaseNode(kind);
    }

    // @api
    // createJSDocNonNullableType
    // createJSDocNullableType
    // createJSDocOptionalType
    // createJSDocVariadicType
    // createJSDocNamepathType

    auto NodeFactory::createJSDocUnaryTypeWorker<T extends JSDocType & { TypeNode type; }>(kind: T["kind"], type: T["type"]) -> T {
        auto node = createBaseNode<T>(kind);
        node->type = type;
        return node;
    }

    // @api
    auto NodeFactory::createJSDocFunctionType(NodeArray<ParameterDeclaration> parameters, TypeNode type) -> JSDocFunctionType {
        auto node = createBaseSignatureDeclaration<JSDocFunctionType>(
            SyntaxKind::JSDocFunctionType,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            /*name*/ undefined,
            /*typeParameters*/ undefined,
            parameters,
            type
        );
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocTypeLiteral(propertyTags?: JSDocPropertyLikeTag[], isArrayType = false) -> JSDocTypeLiteral {
        auto node = createBaseNode<JSDocTypeLiteral>(SyntaxKind::JSDocTypeLiteral);
        node->jsDocPropertyTags = asNodeArray(propertyTags);
        node->isArrayType = isArrayType;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocTypeExpression(TypeNode type) -> JSDocTypeExpression {
        auto node = createBaseNode<JSDocTypeExpression>(SyntaxKind::JSDocTypeExpression);
        node->type = type;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocSignature(typeParameters: JSDocTemplateTag[], parameters: JSDocParameterTag[], type?: JSDocReturnTag) -> JSDocSignature {
        auto node = createBaseNode<JSDocSignature>(SyntaxKind::JSDocSignature);
        node->typeParameters = asNodeArray(typeParameters);
        node->parameters = createNodeArray(parameters);
        node->type = type;
        return node;
    }

    // @api
    

    auto getDefaultTagName(node: JSDocTag) {
        auto defaultTagName = getDefaultTagNameForKind(node->kind);
        return node->tagName.escapedText == escapeLeadingUnderscores(defaultTagName)
            ? node->tagName
            : createIdentifier(defaultTagName);
    }

    // @api
    auto NodeFactory::createBaseJSDocTag<T extends JSDocTag>(kind: T["kind"], tagName: Identifier, comment: string) {
        auto node = createBaseNode<T>(kind);
        node->tagName = tagName;
        node->comment = comment;
        return node;
    }

    // @api
    auto NodeFactory::createJSDocTemplateTag(tagName: Identifier, constraint: JSDocTypeExpression, NodeArray<TypeParameterDeclaration> typeParameters, comment?: string) -> JSDocTemplateTag {
        auto node = createBaseJSDocTag<JSDocTemplateTag>(SyntaxKind::JSDocTemplateTag, tagName ?? createIdentifier("template"), comment);
        node->constraint = constraint;
        node->typeParameters = createNodeArray(typeParameters);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocTypedefTag(tagName: Identifier, typeExpression?: JSDocTypeExpression, fullName?: Identifier | JSDocNamespaceDeclaration, comment?: string) -> JSDocTypedefTag {
        auto node = createBaseJSDocTag<JSDocTypedefTag>(SyntaxKind::JSDocTypedefTag, tagName ?? createIdentifier("typedef"), comment);
        node->typeExpression = typeExpression;
        node->fullName = fullName;
        node->name = getJSDocTypeAliasName(fullName);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocParameterTag(tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression?: JSDocTypeExpression, isNameFirst?: boolean, comment?: string) -> JSDocParameterTag {
        auto node = createBaseJSDocTag<JSDocParameterTag>(SyntaxKind::JSDocParameterTag, tagName ?? createIdentifier("param"), comment);
        node->typeExpression = typeExpression;
        node->name = name;
        node->isNameFirst = !!isNameFirst;
        node->isBracketed = isBracketed;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocPropertyTag(tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression?: JSDocTypeExpression, isNameFirst?: boolean, comment?: string) -> JSDocPropertyTag {
        auto node = createBaseJSDocTag<JSDocPropertyTag>(SyntaxKind::JSDocPropertyTag, tagName ?? createIdentifier("prop"), comment);
        node->typeExpression = typeExpression;
        node->name = name;
        node->isNameFirst = !!isNameFirst;
        node->isBracketed = isBracketed;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocCallbackTag(tagName: Identifier, typeExpression: JSDocSignature, fullName?: Identifier | JSDocNamespaceDeclaration, comment?: string) -> JSDocCallbackTag {
        auto node = createBaseJSDocTag<JSDocCallbackTag>(SyntaxKind::JSDocCallbackTag, tagName ?? createIdentifier("callback"), comment);
        node->typeExpression = typeExpression;
        node->fullName = fullName;
        node->name = getJSDocTypeAliasName(fullName);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocAugmentsTag(tagName: Identifier, className: JSDocAugmentsTag["class"], comment?: string) -> JSDocAugmentsTag {
        auto node = createBaseJSDocTag<JSDocAugmentsTag>(SyntaxKind::JSDocAugmentsTag, tagName ?? createIdentifier("augments"), comment);
        node->class = className;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocImplementsTag(tagName: Identifier, className: JSDocImplementsTag["class"], comment?: string) -> JSDocImplementsTag {
        auto node = createBaseJSDocTag<JSDocImplementsTag>(SyntaxKind::JSDocImplementsTag, tagName ?? createIdentifier("implements"), comment);
        node->class = className;
        return node;
    }

    // @api
    auto NodeFactory::createJSDocSeeTag(tagName: Identifier, name: JSDocNameReference, comment?: string) -> JSDocSeeTag {
        auto node = createBaseJSDocTag<JSDocSeeTag>(SyntaxKind::JSDocSeeTag, tagName ?? createIdentifier("see"), comment);
        node->name = name;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocNameReference(name: EntityName) -> JSDocNameReference {
        auto node = createBaseNode<JSDocNameReference>(SyntaxKind::JSDocNameReference);
        node->name = name;
        return node;
    }

    // @api
    

    // @api
    

    // @api
    // createJSDocAuthorTag
    // createJSDocClassTag
    // createJSDocPublicTag
    // createJSDocPrivateTag
    // createJSDocProtectedTag
    // createJSDocReadonlyTag
    // createJSDocDeprecatedTag
    auto NodeFactory::createJSDocSimpleTagWorker<T extends JSDocTag>(kind: T["kind"], tagName: Identifier, comment?: string) {
        auto node = createBaseJSDocTag<T>(kind, tagName ?? createIdentifier(getDefaultTagNameForKind(kind)), comment);
        return node;
    }

    // @api
    // updateJSDocAuthorTag
    // updateJSDocClassTag
    // updateJSDocPublicTag
    // updateJSDocPrivateTag
    // updateJSDocProtectedTag
    // updateJSDocReadonlyTag
    // updateJSDocDeprecatedTag
    

    // @api
    // createJSDocTypeTag
    // createJSDocReturnTag
    // createJSDocThisTag
    // createJSDocEnumTag
    auto NodeFactory::createJSDocTypeLikeTagWorker<T extends JSDocTag & { typeExpression?: JSDocTypeExpression }>(kind: T["kind"], tagName: Identifier, typeExpression?: JSDocTypeExpression, comment?: string) {
        auto node = createBaseJSDocTag<T>(kind, tagName ?? createIdentifier(getDefaultTagNameForKind(kind)), comment);
        node->typeExpression = typeExpression;
        return node;
    }


    // @api
    auto NodeFactory::createJSDocUnknownTag(tagName: Identifier, comment?: string) -> JSDocUnknownTag {
        auto node = createBaseJSDocTag<JSDocUnknownTag>(SyntaxKind::JSDocTag, tagName, comment);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJSDocComment(comment?: string, tags?: JSDocTag[]) {
        auto node = createBaseNode<JSDoc>(SyntaxKind::JSDocComment);
        node->comment = comment;
        node->tags = asNodeArray(tags);
        return node;
    }

    // @api
    

    //
    // JSX
    //

    // @api
    auto NodeFactory::createJsxElement(openingElement: JsxOpeningElement, children: JsxChild[], closingElement: JsxClosingElement) {
        auto node = createBaseNode<JsxElement>(SyntaxKind::JsxElement);
        node->openingElement = openingElement;
        node->children = createNodeArray(children);
        node->closingElement = closingElement;
        node->transformFlags |=
            propagateChildFlags(node->openingElement) |
            propagateChildrenFlags(node->children) |
            propagateChildFlags(node->closingElement) |
            TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxSelfClosingElement(tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) {
        auto node = createBaseNode<JsxSelfClosingElement>(SyntaxKind::JsxSelfClosingElement);
        node->tagName = tagName;
        node->typeArguments = asNodeArray(typeArguments);
        node->attributes = attributes;
        node->transformFlags |=
            propagateChildFlags(node->tagName) |
            propagateChildrenFlags(node->typeArguments) |
            propagateChildFlags(node->attributes) |
            TransformFlags::ContainsJsx;
        if (node->typeArguments) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxOpeningElement(tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) {
        auto node = createBaseNode<JsxOpeningElement>(SyntaxKind::JsxOpeningElement);
        node->tagName = tagName;
        node->typeArguments = asNodeArray(typeArguments);
        node->attributes = attributes;
        node->transformFlags |=
            propagateChildFlags(node->tagName) |
            propagateChildrenFlags(node->typeArguments) |
            propagateChildFlags(node->attributes) |
            TransformFlags::ContainsJsx;
        if (typeArguments) {
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxClosingElement(tagName: JsxTagNameExpression) {
        auto node = createBaseNode<JsxClosingElement>(SyntaxKind::JsxClosingElement);
        node->tagName = tagName;
        node->transformFlags |=
            propagateChildFlags(node->tagName) |
            TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxFragment(openingFragment: JsxOpeningFragment, children: JsxChild[], closingFragment: JsxClosingFragment) {
        auto node = createBaseNode<JsxFragment>(SyntaxKind::JsxFragment);
        node->openingFragment = openingFragment;
        node->children = createNodeArray(children);
        node->closingFragment = closingFragment;
        node->transformFlags |=
            propagateChildFlags(node->openingFragment) |
            propagateChildrenFlags(node->children) |
            propagateChildFlags(node->closingFragment) |
            TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxText(text: string, containsOnlyTriviaWhiteSpaces?: boolean) {
        auto node = createBaseNode<JsxText>(SyntaxKind::JsxText);
        node->text = text;
        node->containsOnlyTriviaWhiteSpaces = !!containsOnlyTriviaWhiteSpaces;
        node->transformFlags |= TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxOpeningFragment() {
        auto node = createBaseNode<JsxOpeningFragment>(SyntaxKind::JsxOpeningFragment);
        node->transformFlags |= TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    auto NodeFactory::createJsxJsxClosingFragment() {
        auto node = createBaseNode<JsxClosingFragment>(SyntaxKind::JsxClosingFragment);
        node->transformFlags |= TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    auto NodeFactory::createJsxAttribute(name: Identifier, initializer: StringLiteral | JsxExpression) {
        auto node = createBaseNode<JsxAttribute>(SyntaxKind::JsxAttribute);
        node->name = name;
        node->initializer = initializer;
        node->transformFlags |=
            propagateChildFlags(node->name) |
            propagateChildFlags(node->initializer) |
            TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxAttributes(properties: JsxAttributeLike[]) {
        auto node = createBaseNode<JsxAttributes>(SyntaxKind::JsxAttributes);
        node->properties = createNodeArray(properties);
        node->transformFlags |=
            propagateChildrenFlags(node->properties) |
            TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxSpreadAttribute(Expression expression) {
        auto node = createBaseNode<JsxSpreadAttribute>(SyntaxKind::JsxSpreadAttribute);
        node->expression = expression;
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createJsxExpression(dotDotDotToken: DotDotDotToken, Expression expression) {
        auto node = createBaseNode<JsxExpression>(SyntaxKind::JsxExpression);
        node->dotDotDotToken = dotDotDotToken;
        node->expression = expression;
        node->transformFlags |=
            propagateChildFlags(node->dotDotDotToken) |
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsJsx;
        return node;
    }

    // @api
    

    //
    // Clauses
    //

    // @api
    auto NodeFactory::createCaseClause(Expression expression, statements: Statement[]) {
        auto node = createBaseNode<CaseClause>(SyntaxKind::CaseClause);
        node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
        node->statements = createNodeArray(statements);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            propagateChildrenFlags(node->statements);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createDefaultClause(statements: Statement[]) {
        auto node = createBaseNode<DefaultClause>(SyntaxKind::DefaultClause);
        node->statements = createNodeArray(statements);
        node->transformFlags = propagateChildrenFlags(node->statements);
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createHeritageClause(token: HeritageClause["token"], types: ExpressionWithTypeArguments[]) {
        auto node = createBaseNode<HeritageClause>(SyntaxKind::HeritageClause);
        node->token = token;
        node->types = createNodeArray(types);
        node->transformFlags |= propagateChildrenFlags(node->types);
        switch (token) {
            case SyntaxKind::ExtendsKeyword:
                node->transformFlags |= TransformFlags::ContainsES2015;
                break;
            case SyntaxKind::ImplementsKeyword:
                node->transformFlags |= TransformFlags::ContainsTypeScript;
                break;
            default:
                return Debug.assertNever(token);
        }
        return node;
    }

    // @api
    

    // @api
    auto NodeFactory::createCatchClause(variableDeclaration: string | VariableDeclaration, block: Block) {
        auto node = createBaseNode<CatchClause>(SyntaxKind::CatchClause);
        variableDeclaration = !isString(variableDeclaration) ? variableDeclaration : createVariableDeclaration(
            variableDeclaration,
            /*exclamationToken*/ undefined,
            /*type*/ undefined,
            /*initializer*/ undefined
        );
        node->variableDeclaration = variableDeclaration;
        node->block = block;
        node->transformFlags |=
            propagateChildFlags(node->variableDeclaration) |
            propagateChildFlags(node->block);
        if (!variableDeclaration) node->transformFlags |= TransformFlags::ContainsES2019;
        return node;
    }

    // @api
    

    //
    // Property assignments
    //

    // @api
    auto NodeFactory::createPropertyAssignment(PropertyName name, Expression initializer) {
        auto node = createBaseNamedDeclaration<PropertyAssignment>(
            SyntaxKind::PropertyAssignment,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            name
        );
        node->initializer = parenthesizerRules.parenthesizeExpressionForDisallowedComma(initializer);
        node->transformFlags |=
            propagateChildFlags(node->name) |
            propagateChildFlags(node->initializer);
        return node;
    }

    auto finishUpdatePropertyAssignment(updated: Mutable<PropertyAssignment>, original: PropertyAssignment) {
        // copy children used only for error reporting
        if (original->decorators) updated.decorators = original->decorators;
        if (original->modifiers) updated.modifiers = original->modifiers;
        if (original->questionToken) updated.questionToken = original->questionToken;
        if (original->exclamationToken) updated.exclamationToken = original->exclamationToken;
        return update(updated, original);
    }

    // @api
    

    // @api
    auto NodeFactory::createShorthandPropertyAssignment(name: string | Identifier, objectAssignmentInitializer?: Expression) {
        auto node = createBaseNamedDeclaration<ShorthandPropertyAssignment>(
            SyntaxKind::ShorthandPropertyAssignment,
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            name
        );
        node->objectAssignmentInitializer = objectAssignmentInitializer && parenthesizerRules.parenthesizeExpressionForDisallowedComma(objectAssignmentInitializer);
        node->transformFlags |=
            propagateChildFlags(node->objectAssignmentInitializer) |
            TransformFlags::ContainsES2015;
        return node;
    }

    auto finishUpdateShorthandPropertyAssignment(updated: Mutable<ShorthandPropertyAssignment>, original: ShorthandPropertyAssignment) {
        // copy children used only for error reporting
        if (original->decorators) updated.decorators = original->decorators;
        if (original->modifiers) updated.modifiers = original->modifiers;
        if (original->equalsToken) updated.equalsToken = original->equalsToken;
        if (original->questionToken) updated.questionToken = original->questionToken;
        if (original->exclamationToken) updated.exclamationToken = original->exclamationToken;
        return update(updated, original);
    }

    // @api
    

    // @api
    auto NodeFactory::createSpreadAssignment(Expression expression) {
        auto node = createBaseNode<SpreadAssignment>(SyntaxKind::SpreadAssignment);
        node->expression = parenthesizerRules.parenthesizeExpressionForDisallowedComma(expression);
        node->transformFlags |=
            propagateChildFlags(node->expression) |
            TransformFlags::ContainsES2018 |
            TransformFlags::ContainsObjectRestOrSpread;
        return node;
    }

    // @api
    

    //
    // Enum
    //

    // @api
    auto NodeFactory::createEnumMember(PropertyName name, initializer?: Expression) {
        auto node = createBaseNode<EnumMember>(SyntaxKind::EnumMember);
        node->name = asName(name);
        node->initializer = initializer && parenthesizerRules.parenthesizeExpressionForDisallowedComma(initializer);
        node->transformFlags |=
            propagateChildFlags(node->name) |
            propagateChildFlags(node->initializer) |
            TransformFlags::ContainsTypeScript;
        return node;
    }

    // @api
    

    //
    // Top-level nodes
    //

    // @api
    auto NodeFactory::createSourceFile(
        statements: Statement[],
        endOfFileToken: EndOfFileToken,
        flags: NodeFlags
    ) {
        auto node = baseFactory.createBaseSourceFileNode(SyntaxKind::SourceFile) as Mutable<SourceFile>;
        node->statements = createNodeArray(statements);
        node->endOfFileToken = endOfFileToken;
        node->flags |= flags;
        node->fileName = string();
        node->text = string();
        node->languageVersion = 0;
        node->languageVariant = 0;
        node->scriptKind = 0;
        node->isDeclarationFile = false;
        node->hasNoDefaultLib = false;
        node->transformFlags |=
            propagateChildrenFlags(node->statements) |
            propagateChildFlags(node->endOfFileToken);
        return node;
    }


}