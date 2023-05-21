#ifndef NODEFACTORY_H
#define NODEFACTORY_H

#include "node_test.h"
#include "parenthesizer_rules.h"
#include "parser_fwd_types.h"
#include "scanner.h"
#include "utilities.h"

namespace ts
{
class NodeFactory
{
    Scanner *scanner;
    Scanner rawTextScanner;
    ParenthesizerRules parenthesizerRules;
    NodeFactoryFlags flags;
    NodeCreateCallbackFunc createNodeCallback;

  public:
    NodeFactory(ts::Scanner *scanner, NodeFactoryFlags nodeFactoryFlags, NodeCreateCallbackFunc createNodeCallback)
        : scanner(scanner), rawTextScanner(ScriptTarget::Latest, /*skipTrivia*/ false, LanguageVariant::Standard), parenthesizerRules(this),
          flags(nodeFactoryFlags), createNodeCallback(createNodeCallback)
    {
    }

    NodeFactory(NodeFactoryFlags nodeFactoryFlags)
        : scanner(nullptr), rawTextScanner(ScriptTarget::Latest, /*skipTrivia*/ false, LanguageVariant::Standard), parenthesizerRules(this),
          flags(nodeFactoryFlags), createNodeCallback((NodeCreateCallbackFunc)[](Node){})
    {
    }

    auto NoParenthesizerRules() -> boolean;

    template <typename T> auto update(T updated, T original) -> T
    {
        if (!!(flags & NodeFactoryFlags::NoOriginalNode))
        {
            return updateWithoutOriginal(updated, original);
        }

        return updateWithOriginal(updated, original);
    }

    template <typename T> auto updateWithoutOriginal(T updated, T original) -> T
    {
        if (updated != original)
        {
            setTextRange(updated, original);
        }
        return updated;
    }

    template <typename T> auto updateWithOriginal(T updated, T original) -> T
    {
        if (updated != original)
        {
            setOriginalNode(updated, original);
            setTextRange(updated, original);
        }
        return updated;
    }

    inline auto asName(Identifier name) -> Identifier
    {
        return name;
    }

    template <typename T> inline auto asNodeArray(NodeArray<T> array) -> NodeArray<T>
    {
        return createNodeArray(array);
    }

    inline auto asExpression(Expression value) -> Expression
    {
        return value;
    }

    inline auto asToken(Node value) -> Node
    {
        return value;
    }

    auto mergeEmitNode(/*EmitNode*/ Node sourceEmitNode, /*EmitNode*/ Node destEmitNode) -> /* EmitNode */ Node
    {
        // TODO: finish it
        return destEmitNode;
    }

    template <typename T> auto setOriginalNode(T node, Node original) -> T
    {
        node->original = original;
        if (original)
        {
            // TODO: review it
            // auto emitNode = original->emitNode;
            // if (emitNode) node->emitNode = mergeEmitNode(emitNode, node->emitNode);
        }
        return node;
    }

    template <typename T> inline auto asEmbeddedStatement(T statement) -> T
    {
        return statement && isNotEmittedStatement(statement)
                   ? setTextRange(setOriginalNode(createEmptyStatement(), statement), statement).template as<Statement>()
                   : statement;
    }

    auto propagateIdentifierNameFlags(Identifier node) -> TransformFlags;

    auto propagateAssignmentPatternFlags(Node node) -> TransformFlags;

    auto getTransformFlagsSubtreeExclusions(SyntaxKind kind) -> TransformFlags;

    auto propagatePropertyNameFlagsOfChild(PropertyName node, TransformFlags transformFlags) -> TransformFlags;

    auto propagateChildFlags(Node child) -> TransformFlags;

    template <typename T> auto aggregateChildrenFlags(NodeArray<T> children) -> void
    {
        auto subtreeFlags = TransformFlags::None;
        for (auto &child : children)
        {
            subtreeFlags |= propagateChildFlags(child);
        }
        children->transformFlags = subtreeFlags;
    }

    template <typename T> auto propagateChildrenFlags(NodeArray<T> children) -> TransformFlags
    {
        return !!children ? children->transformFlags : TransformFlags::None;
    }

    auto getCookedText(SyntaxKind kind, string rawText) -> std::pair<string, boolean>;

    template <typename T, typename D = typename T::data> auto createBaseNode(SyntaxKind kind)
    {
        auto newNode = T(D());
        newNode->_kind = kind;
        createNodeCallback(newNode);
        return newNode;
    }

    template <typename T> auto createBaseToken(SyntaxKind kind)
    {
        return createBaseNode<T>(kind);
    }

    template <typename T> auto createBaseLiteral(SyntaxKind kind, string text)
    {
        auto node = createBaseNode<T>(kind);
        node->text = text;
        return node;
    }

    // auto createNodeArray(Node elements = undefined, boolean hasTrailingComma = false) -> Node;
    template <typename T> auto createNodeArray(NodeArray<T> elements, boolean hasTrailingComma = false) -> NodeArray<T>
    {
        setTextRangePosEnd(elements, -1, -1);
        elements.hasTrailingComma = !!hasTrailingComma;
        aggregateChildrenFlags(elements);
        Debug::attachNodeArrayDebugInfo(elements);
        return elements;
    }

    template <typename T = Node> auto createToken(SyntaxKind token) -> T
    {
        Debug::_assert(token >= SyntaxKind::FirstToken && token <= SyntaxKind::LastToken, S("Invalid token"));
        Debug::_assert(token <= SyntaxKind::FirstTemplateToken || token >= SyntaxKind::LastTemplateToken,
                       S("Invalid token. Use 'createTemplateLiteralLikeNode' to create template literals."));
        Debug::_assert(token <= SyntaxKind::FirstLiteralToken || token >= SyntaxKind::LastLiteralToken,
                       S("Invalid token. Use 'createLiteralLikeNode' to create literals."));
        Debug::_assert(token != SyntaxKind::Identifier, S("Invalid token. Use 'createIdentifier' to create identifiers"));
        // auto node = createBaseTokenNode<Token<TKind>>(token);
        auto node = createBaseNode<T>(token);
        auto transformFlags = TransformFlags::None;
        switch (token)
        {
        case SyntaxKind::AsyncKeyword:
            // 'async' modifier is ES2017 (async functions) or ES2018 (async generators)
            transformFlags = TransformFlags::ContainsES2017 | TransformFlags::ContainsES2018;
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

    template <typename T> auto createJSDocPrimaryTypeWorker(SyntaxKind kind)
    {
        return createBaseNode<T>(kind);
    }

    template <typename T> auto createJSDocUnaryTypeWorker(SyntaxKind kind, decltype(T()->type) type) -> T
    {
        auto node = createBaseNode<T>(kind);
        node->type = type;
        return node;
    }

    template <typename T> auto createBaseJSDocTag(SyntaxKind kind, Identifier tagName, string comment)
    {
        auto node = createBaseNode<T>(kind);
        node->tagName = tagName;
        node->comment = comment;
        return node;
    }

    auto static getDefaultTagNameForKind(SyntaxKind kind) -> string;

    template <typename T> auto createJSDocSimpleTagWorker(SyntaxKind kind, Identifier tagName, string comment)
    {
        auto node = createBaseJSDocTag<T>(kind, tagName ? tagName : createIdentifier(getDefaultTagNameForKind(kind)), comment);
        return node;
    }

    template <typename T>
    auto createJSDocTypeLikeTagWorker(SyntaxKind kind, Identifier tagName, JSDocTypeExpression typeExpression, string comment)
    {
        auto node = createBaseJSDocTag<T>(kind, tagName ? tagName : createIdentifier(getDefaultTagNameForKind(kind)), comment);
        node->typeExpression = typeExpression;
        return node;
    }

    template <typename T> auto createBaseDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers)
    {
        auto node = createBaseNode<T>(kind);
        node->decorators = asNodeArray(decorators);
        node->modifiers = asNodeArray(modifiers);
        node->transformFlags |= propagateChildrenFlags(node->decorators) | propagateChildrenFlags(node->modifiers);
        // NOTE: The following properties are commonly set by the binder and are added here to
        // ensure declarations have a stable shape.
        node->symbol = undefined;        // initialized by binder
        node->localSymbol = undefined;   // initialized by binder
        node->locals.clear();            // initialized by binder
        //node->nextContainer = undefined; // initialized by binder
        return node;
    }

    template <typename T>
    auto createBaseNamedDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers, Identifier name) -> T
    {
        auto node = createBaseDeclaration<T>(kind, decorators, modifiers);
        name = asName(name);
        node->name = name;

        // The PropertyName of a member is allowed to be `await`.
        // We don't need to exclude `await` for type signatures since types
        // don't propagate child flags.
        if (name)
        {
            switch ((SyntaxKind)node)
            {
            case SyntaxKind::MethodDeclaration:
            case SyntaxKind::GetAccessor:
            case SyntaxKind::SetAccessor:
            case SyntaxKind::PropertyDeclaration:
            case SyntaxKind::PropertyAssignment:
                if (isIdentifier(name))
                {
                    node->transformFlags |= propagateIdentifierNameFlags(name);
                    break;
                }
                // fall through
            default:
                node->transformFlags |= propagateChildFlags(name);
                break;
            }
        }
        return node;
    }

    template <typename T>
    auto createBaseBindingLikeDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers, BindingName name,
                                          Expression initializer) -> T
    {
        auto node = createBaseNamedDeclaration<T>(kind, decorators, modifiers, name);
        node->initializer = initializer;
        node->transformFlags |= propagateChildFlags(node->initializer);
        return node;
    }

    template <typename T>
    auto createBaseVariableLikeDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers, BindingName name,
                                           TypeNode type, Expression initializer) -> T
    {
        auto node = createBaseBindingLikeDeclaration<T>(kind, decorators, modifiers, name, initializer);
        node->type = type;
        node->transformFlags |= propagateChildFlags(type);
        if (type)
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        return node;
    }

    template <typename T>
    auto createBaseGenericNamedDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers, PropertyName name,
                                           NodeArray<TypeParameterDeclaration> typeParameters)
    {
        auto node = createBaseNamedDeclaration<T>(kind, decorators, modifiers, name);
        node->typeParameters = asNodeArray(typeParameters);
        node->transformFlags |= propagateChildrenFlags(node->typeParameters);
        if (!typeParameters.empty())
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        return node;
    }

    template <typename T>
    auto createBaseSignatureDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers, PropertyName name,
                                        NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                        TypeNode type)
    {
        auto node = createBaseGenericNamedDeclaration<T>(kind, decorators, modifiers, name, typeParameters);
        node->parameters = createNodeArray(parameters);
        node->type = type;
        node->transformFlags |= propagateChildrenFlags(node->parameters) | propagateChildFlags(node->type);
        if (type)
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        return node;
    }

    template <typename T>
    auto createBaseFunctionLikeDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers, PropertyName name,
                                           NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                           TypeNode type, Block body)
    {
        auto node = createBaseSignatureDeclaration<T>(kind, decorators, modifiers, name, typeParameters, parameters, type);
        node->body = body;
        node->transformFlags |= propagateChildFlags(node->body) & ~TransformFlags::ContainsPossibleTopLevelAwait;
        if (!body)
            node->transformFlags |= TransformFlags::ContainsTypeScript;
        return node;
    }

    template <typename T> auto createBaseExpression(SyntaxKind kind)
    {
        auto node = createBaseNode<T>(kind);
        // the following properties are commonly set by the checker/binder
        return node;
    }

    template <typename T>
    auto createBaseInterfaceOrClassLikeDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
                                                   NodeArray<TypeParameterDeclaration> typeParameters,
                                                   NodeArray<HeritageClause> heritageClauses)
    {
        auto node = createBaseGenericNamedDeclaration<T>(kind, decorators, modifiers, name, typeParameters);
        node->heritageClauses = asNodeArray(heritageClauses);
        node->transformFlags |= propagateChildrenFlags(node->heritageClauses);
        return node;
    }

    template <typename T>
    auto createBaseClassLikeDeclaration(SyntaxKind kind, DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
                                        NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses,
                                        NodeArray<ClassElement> members)
    {
        auto node = createBaseInterfaceOrClassLikeDeclaration<T>(kind, decorators, modifiers, name, typeParameters, heritageClauses);
        node->members = createNodeArray(members);
        node->transformFlags |= propagateChildrenFlags(node->members);
        return node;
    }

    //
    // Literals
    //

    auto createNumericLiteral(string value, TokenFlags numericLiteralFlags = TokenFlags::None) -> NumericLiteral;
    // auto createNumericLiteral(number value, TokenFlags numericLiteralFlags = TokenFlags::None) -> NumericLiteral;
    auto createBigIntLiteral(string value) -> BigIntLiteral;
    // auto createBigIntLiteral(PseudoBigInt value) -> BigIntLiteral;
    auto createBaseStringLiteral(string text, boolean isSingleQuote = false) -> StringLiteral;
    /* @internal*/ auto createStringLiteral(string text, boolean isSingleQuote = false, boolean hasExtendedUnicodeEscape = false)
        -> StringLiteral; // eslint-disable-line @typescript-eslint/unified-signatures
    // auto createStringLiteralFromNode(PropertyNameLiteral sourceNode, boolean isSingleQuote = false) -> StringLiteral;
    auto createRegularExpressionLiteral(string text) -> RegularExpressionLiteral;

    //
    // Identifiers
    //

    auto createBaseIdentifier(string text, SyntaxKind originalKeywordKind = SyntaxKind::Unknown);
    /* @internal */ auto createIdentifier(string text, NodeArray</*TypeNode | TypeParameterDeclaration*/ Node> typeArguments = undefined,
                                          SyntaxKind originalKeywordKind = SyntaxKind::Unknown)
        -> Identifier; // eslint-disable-line @typescript-eslint/unified-signatures
                       ///* @internal */ auto updateIdentifier(Identifier node, NodeArray</*TypeNode | TypeParameterDeclaration*/Node>
                       /// typeArguments) -> Identifier;

    /**
     * Create a unique temporary variable.
     * @param recordTempVariable An optional callback used to record the temporary variable name. This
     * should usually be a reference to `hoistVariableDeclaration` from a `TransformationContext`, but
     * can be `undefined` if you plan to record the temporary variable manually.
     * @param reservedInNestedScopes When `true`, reserves the temporary variable name in all nested scopes
     * during emit so that the variable can be referenced in a nested function body. This is an alternative to
     * setting `EmitFlags.ReuseTempVariableScope` on the nested function itself.
     */
    // auto createTempVariable(std::function<void(Identifier)> recordTempVariable, boolean reservedInNestedScopes = false) -> Identifier;

    /**
     * Create a unique temporary variable for use in a loop.
     * @param reservedInNestedScopes When `true`, reserves the temporary variable name in all nested scopes
     * during emit so that the variable can be referenced in a nested function body. This is an alternative to
     * setting `EmitFlags.ReuseTempVariableScope` on the nested function itself.
     */
    // auto createLoopVariable(boolean reservedInNestedScopes = false) -> Identifier;

    // /** Create a unique name based on the supplied text. */
    // auto createUniqueName(string text, GeneratedIdentifierFlags flags = (GeneratedIdentifierFlags)0) -> Identifier;

    // /** Create a unique name generated for a node. */
    // auto getGeneratedNameForNode(Node node, GeneratedIdentifierFlags flags = (GeneratedIdentifierFlags)0) -> Identifier;

    auto createPrivateIdentifier(string text) -> PrivateIdentifier;

    // //
    // // Punctuation
    // //

    // //
    // // Reserved words
    // //

    // auto createSuper() -> SuperExpression;
    // auto createThis() -> ThisExpression;
    // auto createNull() -> NullLiteral;
    // auto createTrue() -> TrueLiteral;
    // auto createFalse() -> FalseLiteral;

    // //
    // // Modifiers
    // //

    // template <typename T/*extends ModifierSyntaxKind*/> auto createModifier(SyntaxKind kind) -> Node;
    // auto createModifiersFromModifierFlags(ModifierFlags flags) -> ModifiersArray;

    // //
    // // Names
    // //

    // auto createQualifiedName(EntityName left, string right) -> QualifiedName;
    auto createQualifiedName(EntityName left, Identifier right) -> QualifiedName;
    // auto updateQualifiedName(QualifiedName node, EntityName left, Identifier right) -> QualifiedName;
    auto createComputedPropertyName(Expression expression) -> ComputedPropertyName;
    // auto updateComputedPropertyName(ComputedPropertyName node, Expression expression) -> ComputedPropertyName;

    // //
    // // Signature elements
    // //

    // auto createTypeParameterDeclaration(string name, TypeNode constraint = undefined, TypeNode defaultType = undefined) ->
    // TypeParameterDeclaration;
    auto createTypeParameterDeclaration(Identifier name, TypeNode constraint = undefined, TypeNode defaultType = undefined)
        -> TypeParameterDeclaration;
    // auto updateTypeParameterDeclaration(TypeParameterDeclaration node, Identifier name, TypeNode constraint, TypeNode defaultType) ->
    // TypeParameterDeclaration;

    // auto createParameterDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, DotDotDotToken dotDotDotToken, string name,
    // QuestionToken questionToken = undefined, TypeNode type = undefined, Expression initializer = undefined) -> ParameterDeclaration;
    auto createParameterDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, DotDotDotToken dotDotDotToken, BindingName name,
                                    QuestionToken questionToken = undefined, TypeNode type = undefined, Expression initializer = undefined)
        -> ParameterDeclaration;
    // auto updateParameterDeclaration(ParameterDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, DotDotDotToken
    // dotDotDotToken, string name, QuestionToken questionToken, TypeNode type, Expression initializer) -> ParameterDeclaration; auto
    // updateParameterDeclaration(ParameterDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, DotDotDotToken
    // dotDotDotToken, BindingName name, QuestionToken questionToken, TypeNode type, Expression initializer) -> ParameterDeclaration;
    auto createDecorator(Expression expression) -> Decorator;
    // auto updateDecorator(Decorator node, Expression expression) -> Decorator;

    // //
    // // Type Elements
    // //

    // auto createPropertySignature(ModifiersArray modifiers, string name, QuestionToken questionToken, TypeNode type) -> PropertySignature;
    auto createPropertySignature(ModifiersArray modifiers, PropertyName name, QuestionToken questionToken, TypeNode type)
        -> PropertySignature;
    // auto updatePropertySignature(PropertySignature node, ModifiersArray modifiers, PropertyName name, QuestionToken questionToken,
    // TypeNode type) -> PropertySignature; auto createPropertyDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, string
    // name, Node questionOrExclamationToken, TypeNode type, Expression initializer) -> PropertyDeclaration;
    auto createPropertyDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, PropertyName name, Node questionOrExclamationToken,
                                   TypeNode type, Expression initializer) -> PropertyDeclaration;
    // auto updatePropertyDeclaration(PropertyDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, string name, Node
    // questionOrExclamationToken, TypeNode type, Expression initializer) -> PropertyDeclaration; auto
    // updatePropertyDeclaration(PropertyDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, PropertyName name, Node
    // questionOrExclamationToken, TypeNode type, Expression initializer) -> PropertyDeclaration;

    // auto createMethodSignature(ModifiersArray modifiers, string name, QuestionToken questionToken, NodeArray<TypeParameterDeclaration>
    // typeParameters, NodeArray<ParameterDeclaration> parameters, TypeNode type) -> MethodSignature;
    auto createMethodSignature(ModifiersArray modifiers, PropertyName name, QuestionToken questionToken,
                               NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                               TypeNode type) -> MethodSignature;
    // auto updateMethodSignature(MethodSignature node, ModifiersArray modifiers, PropertyName name, QuestionToken questionToken,
    // NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters, TypeNode type) -> MethodSignature;
    // auto createMethodDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, AsteriskToken asteriskToken, string name,
    // QuestionToken questionToken, NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters, TypeNode
    // type, Block body) -> MethodDeclaration;

    auto createMethodDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, AsteriskToken asteriskToken, PropertyName name,
                                 QuestionToken questionToken, NodeArray<TypeParameterDeclaration> typeParameters,
                                 NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body) -> MethodDeclaration;
    // auto updateMethodDeclaration(MethodDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, AsteriskToken
    // asteriskToken, PropertyName name, QuestionToken questionToken, NodeArray<TypeParameterDeclaration> typeParameters,
    // NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body) -> MethodDeclaration;
    auto createConstructorDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, NodeArray<ParameterDeclaration> parameters,
                                      Block body) -> ConstructorDeclaration;
    // auto updateConstructorDeclaration(ConstructorDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers,
    // NodeArray<ParameterDeclaration> parameters, Block body) -> ConstructorDeclaration; auto createGetAccessorDeclaration(DecoratorsArray
    // decorators, ModifiersArray modifiers, string name, NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body) ->
    // GetAccessorDeclaration;
    auto createGetAccessorDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, PropertyName name,
                                      NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body) -> GetAccessorDeclaration;
    // auto updateGetAccessorDeclaration(GetAccessorDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, PropertyName
    // name, NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body) -> GetAccessorDeclaration; auto
    // createSetAccessorDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, string name, NodeArray<ParameterDeclaration>
    // parameters, Block body) -> SetAccessorDeclaration;
    auto createSetAccessorDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, PropertyName name,
                                      NodeArray<ParameterDeclaration> parameters, Block body) -> SetAccessorDeclaration;
    // auto updateSetAccessorDeclaration(SetAccessorDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, PropertyName
    // name, NodeArray<ParameterDeclaration> parameters, Block body) -> SetAccessorDeclaration;
    auto createCallSignature(NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters, TypeNode type)
        -> CallSignatureDeclaration;
    // auto updateCallSignature(CallSignatureDeclaration node, NodeArray<TypeParameterDeclaration> typeParameters,
    // NodeArray<ParameterDeclaration> parameters, TypeNode type) -> CallSignatureDeclaration;
    auto createConstructSignature(NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                  TypeNode type) -> ConstructSignatureDeclaration;
    // auto updateConstructSignature(ConstructSignatureDeclaration node, NodeArray<TypeParameterDeclaration> typeParameters,
    // NodeArray<ParameterDeclaration> parameters, TypeNode type) -> ConstructSignatureDeclaration;
    auto createIndexSignature(DecoratorsArray decorators, ModifiersArray modifiers, NodeArray<ParameterDeclaration> parameters,
                              TypeNode type) -> IndexSignatureDeclaration;
    // auto updateIndexSignature(IndexSignatureDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers,
    // NodeArray<ParameterDeclaration> parameters, TypeNode type) -> IndexSignatureDeclaration;
    auto createTemplateLiteralTypeSpan(TypeNode type, Node literal) -> TemplateLiteralTypeSpan;
    // auto updateTemplateLiteralTypeSpan(TemplateLiteralTypeSpan node, TypeNode type, Node literal) -> TemplateLiteralTypeSpan;

    // //
    // // Types
    // //

    // auto createKeywordTypeNode(SyntaxKind kind) -> Node;
    // auto createTypePredicateNode(AssertsKeyword assertsModifier, string parameterName, TypeNode type) -> TypePredicateNode;
    auto createTypePredicateNode(AssertsKeyword assertsModifier, Node parameterName, TypeNode type) -> TypePredicateNode;
    // auto updateTypePredicateNode(TypePredicateNode node, AssertsKeyword assertsModifier, Node parameterName, TypeNode type) ->
    // TypePredicateNode; auto createTypeReferenceNode(string typeName, NodeArray<TypeNode> typeArguments = undefined) -> TypeReferenceNode;
    auto createTypeReferenceNode(EntityName typeName, NodeArray<TypeNode> typeArguments = undefined) -> TypeReferenceNode;
    // auto updateTypeReferenceNode(TypeReferenceNode node, EntityName typeName, NodeArray<TypeNode> typeArguments) -> TypeReferenceNode;
    auto createFunctionTypeNode(NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                TypeNode type) -> FunctionTypeNode;
    // auto updateFunctionTypeNode(FunctionTypeNode node, NodeArray<TypeParameterDeclaration> typeParameters,
    // NodeArray<ParameterDeclaration> parameters, TypeNode type) -> FunctionTypeNode;
    auto createConstructorTypeNode(ModifiersArray modifiers, NodeArray<TypeParameterDeclaration> typeParameters,
                                   NodeArray<ParameterDeclaration> parameters, TypeNode type) -> ConstructorTypeNode;
    // /** @deprecated */
    // auto createConstructorTypeNode(NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
    // TypeNode type) -> ConstructorTypeNode; auto updateConstructorTypeNode(ConstructorTypeNode node, ModifiersArray modifiers,
    // NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters, TypeNode type) ->
    // ConstructorTypeNode;
    // /** @deprecated */
    // auto updateConstructorTypeNode(ConstructorTypeNode node, NodeArray<TypeParameterDeclaration> typeParameters,
    // NodeArray<ParameterDeclaration> parameters, TypeNode type) -> ConstructorTypeNode;
    auto createTypeQueryNode(EntityName exprName) -> TypeQueryNode;
    // auto updateTypeQueryNode(TypeQueryNode node, EntityName exprName) -> TypeQueryNode;
    auto createTypeLiteralNode(NodeArray<TypeElement> members) -> TypeLiteralNode;
    // auto updateTypeLiteralNode(TypeLiteralNode node, NodeArray<TypeElement> members) -> TypeLiteralNode;
    auto createArrayTypeNode(TypeNode elementType) -> ArrayTypeNode;
    // auto updateArrayTypeNode(ArrayTypeNode node, TypeNode elementType) -> ArrayTypeNode;
    auto createTupleTypeNode(NodeArray</*TypeNode | NamedTupleMember*/ Node> elements) -> TupleTypeNode;
    // auto updateTupleTypeNode(TupleTypeNode node, NodeArray</*TypeNode | NamedTupleMember*/Node> elements) -> TupleTypeNode;
    auto createNamedTupleMember(DotDotDotToken dotDotDotToken, Identifier name, QuestionToken questionToken, TypeNode type)
        -> NamedTupleMember;
    // auto updateNamedTupleMember(NamedTupleMember node, DotDotDotToken dotDotDotToken, Identifier name, QuestionToken questionToken,
    // TypeNode type) -> NamedTupleMember;
    auto createOptionalTypeNode(TypeNode type) -> OptionalTypeNode;
    // auto updateOptionalTypeNode(OptionalTypeNode node, TypeNode type) -> OptionalTypeNode;
    auto createRestTypeNode(TypeNode type) -> RestTypeNode;
    // auto updateRestTypeNode(RestTypeNode node, TypeNode type) -> RestTypeNode;
    auto createUnionTypeNode(NodeArray<TypeNode> types) -> UnionTypeNode;
    // auto updateUnionTypeNode(UnionTypeNode node, NodeArray<TypeNode> types) -> UnionTypeNode;
    auto createIntersectionTypeNode(NodeArray<TypeNode> types) -> IntersectionTypeNode;
    // auto updateIntersectionTypeNode(IntersectionTypeNode node, NodeArray<TypeNode> types) -> IntersectionTypeNode;
    auto createConditionalTypeNode(TypeNode checkType, TypeNode extendsType, TypeNode trueType, TypeNode falseType) -> ConditionalTypeNode;
    // auto updateConditionalTypeNode(ConditionalTypeNode node, TypeNode checkType, TypeNode extendsType, TypeNode trueType, TypeNode
    // falseType) -> ConditionalTypeNode;
    auto createInferTypeNode(TypeParameterDeclaration typeParameter) -> InferTypeNode;
    // auto updateInferTypeNode(InferTypeNode node, TypeParameterDeclaration typeParameter) -> InferTypeNode;
    auto createImportTypeNode(TypeNode argument, EntityName qualifier = undefined, NodeArray<TypeNode> typeArguments = undefined,
                              boolean isTypeOf = false) -> ImportTypeNode;
    // auto updateImportTypeNode(ImportTypeNode node, TypeNode argument, EntityName qualifier, NodeArray<TypeNode> typeArguments, boolean
    // isTypeOf = false) -> ImportTypeNode;
    auto createParenthesizedType(TypeNode type) -> ParenthesizedTypeNode;
    // auto updateParenthesizedType(ParenthesizedTypeNode node, TypeNode type) -> ParenthesizedTypeNode;
    auto createThisTypeNode() -> ThisTypeNode;
    auto createTypeOperatorNode(SyntaxKind kind, TypeNode type) -> TypeOperatorNode;
    // auto updateTypeOperatorNode(TypeOperatorNode node, TypeNode type) -> TypeOperatorNode;
    auto createIndexedAccessTypeNode(TypeNode objectType, TypeNode indexType) -> IndexedAccessTypeNode;
    // auto updateIndexedAccessTypeNode(IndexedAccessTypeNode node, TypeNode objectType, TypeNode indexType) -> IndexedAccessTypeNode;
    auto createMappedTypeNode(Node readonlyToken, TypeParameterDeclaration typeParameter, TypeNode nameType, Node questionToken,
                              TypeNode type) -> MappedTypeNode;
    // auto updateMappedTypeNode(MappedTypeNode node, Node token, TypeParameterDeclaration typeParameter, TypeNode nameType, Node
    // questionToken, TypeNode type) -> MappedTypeNode;
    auto createLiteralTypeNode(LiteralTypeNode literal) -> LiteralTypeNode;
    // auto updateLiteralTypeNode(LiteralTypeNode node, LiteralTypeNode literal) -> LiteralTypeNode;
    auto createTemplateLiteralType(TemplateHead head, NodeArray<TemplateLiteralTypeSpan> templateSpans) -> TemplateLiteralTypeNode;
    // auto updateTemplateLiteralType(TemplateLiteralTypeNode node, TemplateHead head, NodeArray<TemplateLiteralTypeSpan> templateSpans) ->
    // TemplateLiteralTypeNode;

    // //
    // // Binding Patterns
    // //

    auto createObjectBindingPattern(NodeArray<BindingElement> elements) -> ObjectBindingPattern;
    // auto updateObjectBindingPattern(ObjectBindingPattern node, NodeArray<BindingElement> elements) -> ObjectBindingPattern;
    auto createArrayBindingPattern(NodeArray<ArrayBindingElement> elements) -> ArrayBindingPattern;
    // auto updateArrayBindingPattern(ArrayBindingPattern node, NodeArray<ArrayBindingElement> elements) -> ArrayBindingPattern;
    // auto createBindingElement(DotDotDotToken dotDotDotToken, string propertyName, string name, Expression initializer = undefined) ->
    // BindingElement;
    auto createBindingElement(DotDotDotToken dotDotDotToken, PropertyName propertyName, BindingName name,
                              Expression initializer = undefined) -> BindingElement;
    // auto updateBindingElement(BindingElement node, DotDotDotToken dotDotDotToken, PropertyName propertyName, BindingName name, Expression
    // initializer) -> BindingElement;

    // //
    // // Expression
    // //

    auto createArrayLiteralExpression(NodeArray<Expression> elements = undefined, boolean multiLine = false) -> ArrayLiteralExpression;
    // auto updateArrayLiteralExpression(ArrayLiteralExpression node, NodeArray<Expression> elements) -> ArrayLiteralExpression;
    auto createObjectLiteralExpression(NodeArray<ObjectLiteralElementLike> properties = undefined, boolean multiLine = false)
        -> ObjectLiteralExpression;
    // auto updateObjectLiteralExpression(ObjectLiteralExpression node, NodeArray<ObjectLiteralElementLike> properties) ->
    // ObjectLiteralExpression; auto createPropertyAccessExpression(Expression expression, string name) -> PropertyAccessExpression;
    auto createPropertyAccessExpression(Expression expression, MemberName name) -> PropertyAccessExpression;
    // auto updatePropertyAccessExpression(PropertyAccessExpression node, Expression expression, MemberName name) ->
    // PropertyAccessExpression; auto createPropertyAccessChain(Expression expression, QuestionDotToken questionDotToken, string name) ->
    // PropertyAccessChain;
    auto createPropertyAccessChain(Expression expression, QuestionDotToken questionDotToken, MemberName name) -> PropertyAccessChain;
    // auto updatePropertyAccessChain(PropertyAccessChain node, Expression expression, QuestionDotToken questionDotToken, MemberName name)
    // -> PropertyAccessChain; auto createElementAccessExpression(Expression expression, number index) -> ElementAccessExpression;
    auto createElementAccessExpression(Expression expression, Expression index) -> ElementAccessExpression;
    // auto updateElementAccessExpression(ElementAccessExpression node, Expression expression, Expression argumentExpression) ->
    // ElementAccessExpression; auto createElementAccessChain(Expression expression, QuestionDotToken questionDotToken, number index) ->
    // ElementAccessChain;
    auto createElementAccessChain(Expression expression, QuestionDotToken questionDotToken, Expression index) -> ElementAccessChain;
    // auto updateElementAccessChain(ElementAccessChain node, Expression expression, QuestionDotToken questionDotToken, Expression
    // argumentExpression) -> ElementAccessChain;
    auto createCallExpression(Expression expression, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray)
        -> CallExpression;
    auto updateCallExpression(CallExpression node, Expression expression, NodeArray<TypeNode> typeArguments,
                              NodeArray<Expression> argumentsArray) -> CallExpression;
    auto createCallChain(Expression expression, QuestionDotToken questionDotToken, NodeArray<TypeNode> typeArguments,
                         NodeArray<Expression> argumentsArray) -> CallChain;
    // auto updateCallChain(CallChain node, Expression expression, QuestionDotToken questionDotToken, NodeArray<TypeNode> typeArguments,
    // NodeArray<Expression> argumentsArray) -> CallChain;
    auto createNewExpression(Expression expression, NodeArray<TypeNode> typeArguments, NodeArray<Expression> argumentsArray)
        -> NewExpression;
    // auto updateNewExpression(NewExpression node, Expression expression, NodeArray<TypeNode> typeArguments, NodeArray<Expression>
    // argumentsArray) -> NewExpression;
    auto createTaggedTemplateExpression(Expression tag, NodeArray<TypeNode> typeArguments, TemplateLiteral _template)
        -> TaggedTemplateExpression;
    // auto updateTaggedTemplateExpression(TaggedTemplateExpression node, Expression tag, NodeArray<TypeNode> typeArguments, TemplateLiteral
    // _template) -> TaggedTemplateExpression;
    auto createTypeAssertion(TypeNode type, Expression expression) -> TypeAssertion;
    // auto updateTypeAssertion(TypeAssertion node, TypeNode type, Expression expression) -> TypeAssertion;
    auto createParenthesizedExpression(Expression expression) -> ParenthesizedExpression;
    // auto updateParenthesizedExpression(ParenthesizedExpression node, Expression expression) -> ParenthesizedExpression;
    // auto createFunctionExpression(ModifiersArray modifiers, AsteriskToken asteriskToken, string name, NodeArray<TypeParameterDeclaration>
    // typeParameters, NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body) -> FunctionExpression;
    auto createFunctionExpression(ModifiersArray modifiers, AsteriskToken asteriskToken, Identifier name,
                                  NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                  TypeNode type, Block body) -> FunctionExpression;
    // auto updateFunctionExpression(FunctionExpression node, ModifiersArray modifiers, AsteriskToken asteriskToken, Identifier name,
    // NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters, TypeNode type, Block body) ->
    // FunctionExpression;
    auto createArrowFunction(ModifiersArray modifiers, NodeArray<TypeParameterDeclaration> typeParameters,
                             NodeArray<ParameterDeclaration> parameters, TypeNode type, EqualsGreaterThanToken equalsGreaterThanToken,
                             ConciseBody body) -> ArrowFunction;
    // auto updateArrowFunction(ArrowFunction node, ModifiersArray modifiers, NodeArray<TypeParameterDeclaration> typeParameters,
    // NodeArray<ParameterDeclaration> parameters, TypeNode type, EqualsGreaterThanToken equalsGreaterThanToken, ConciseBody body) ->
    // ArrowFunction;
    auto createDeleteExpression(Expression expression) -> DeleteExpression;
    // auto updateDeleteExpression(DeleteExpression node, Expression expression) -> DeleteExpression;
    auto createTypeOfExpression(Expression expression) -> TypeOfExpression;
    // auto updateTypeOfExpression(TypeOfExpression node, Expression expression) -> TypeOfExpression;
    auto createVoidExpression(Expression expression) -> VoidExpression;
    // auto updateVoidExpression(VoidExpression node, Expression expression) -> VoidExpression;
    auto createAwaitExpression(Expression expression) -> AwaitExpression;
    // auto updateAwaitExpression(AwaitExpression node, Expression expression) -> AwaitExpression;
    auto createPrefixUnaryExpression(PrefixUnaryOperator _operator, Expression operand) -> PrefixUnaryExpression;
    // auto updatePrefixUnaryExpression(PrefixUnaryExpression node, Expression operand) -> PrefixUnaryExpression;
    auto createPostfixUnaryExpression(Expression operand, PostfixUnaryOperator _operator) -> PostfixUnaryExpression;
    // auto updatePostfixUnaryExpression(PostfixUnaryExpression node, Expression operand) -> PostfixUnaryExpression;
    auto createBinaryExpression(Expression left, Node _operator, Expression right) -> BinaryExpression;
    // auto updateBinaryExpression(BinaryExpression node, Expression left, Node _operator, Expression right) -> BinaryExpression;
    auto createConditionalExpression(Expression condition, QuestionToken questionToken, Expression whenTrue, ColonToken colonToken,
                                     Expression whenFalse) -> ConditionalExpression;
    // auto updateConditionalExpression(ConditionalExpression node, Expression condition, QuestionToken questionToken, Expression whenTrue,
    // ColonToken colonToken, Expression whenFalse) -> ConditionalExpression;
    auto createTemplateExpression(TemplateHead head, NodeArray<TemplateSpan> templateSpans) -> TemplateExpression;
    // auto updateTemplateExpression(TemplateExpression node, TemplateHead head, NodeArray<TemplateSpan> templateSpans) ->
    // TemplateExpression;
    auto createTemplateHead(string text, string rawText = string(), TokenFlags templateFlags = TokenFlags::None) -> TemplateHead;
    auto createTemplateMiddle(string text, string rawText = string(), TokenFlags templateFlags = TokenFlags::None) -> TemplateMiddle;
    auto createTemplateTail(string text, string rawText = string(), TokenFlags templateFlags = TokenFlags::None) -> TemplateTail;
    auto createNoSubstitutionTemplateLiteral(string text, string rawText = string(), TokenFlags templateFlags = TokenFlags::None)
        -> NoSubstitutionTemplateLiteral;
    /* @internal */ auto createLiteralLikeNode(SyntaxKind kind, string text) -> LiteralLikeNode;
    /* @internal */ auto createTemplateLiteralLikeNodeChecked(SyntaxKind kind, string text, string rawText = string(),
                                                              TokenFlags templateFlags = TokenFlags::None) -> TemplateLiteralLikeNode;
    /* @internal */ auto createTemplateLiteralLikeNode(SyntaxKind kind, string text, string rawText = string(),
                                                       TokenFlags templateFlags = TokenFlags::None) -> TemplateLiteralLikeNode;
    auto createYieldExpression(AsteriskToken asteriskToken, Expression expression) -> YieldExpression;
    // auto updateYieldExpression(YieldExpression node, AsteriskToken asteriskToken, Expression expression) -> YieldExpression;
    auto createSpreadElement(Expression expression) -> SpreadElement;
    // auto updateSpreadElement(SpreadElement node, Expression expression) -> SpreadElement;
    // auto createClassExpression(DecoratorsArray decorators, ModifiersArray modifiers, string name, NodeArray<TypeParameterDeclaration>
    // typeParameters, NodeArray<HeritageClause> heritageClauses, NodeArray<ClassElement> members) -> ClassExpression;
    auto createClassExpression(DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
                               NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses,
                               NodeArray<ClassElement> members) -> ClassExpression;
    // auto updateClassExpression(ClassExpression node, DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
    // NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses, NodeArray<ClassElement> members) ->
    // ClassExpression;
    auto createOmittedExpression() -> OmittedExpression;
    auto createExpressionWithTypeArguments(Expression expression, NodeArray<TypeNode> typeArguments) -> ExpressionWithTypeArguments;
    // auto updateExpressionWithTypeArguments(ExpressionWithTypeArguments node, Expression expression, NodeArray<TypeNode> typeArguments) ->
    // ExpressionWithTypeArguments;
    auto createAsExpression(Expression expression, TypeNode type) -> AsExpression;
    // auto updateAsExpression(AsExpression node, Expression expression, TypeNode type) -> AsExpression;
    auto createNonNullExpression(Expression expression) -> NonNullExpression;
    // auto updateNonNullExpression(NonNullExpression node, Expression expression) -> NonNullExpression;
    auto createNonNullChain(Expression expression) -> NonNullChain;
    // auto updateNonNullChain(NonNullChain node, Expression expression) -> NonNullChain;
    auto createMetaProperty(SyntaxKind keywordToken, Identifier name) -> MetaProperty;
    // auto updateMetaProperty(MetaProperty node, Identifier name) -> MetaProperty;

    // //
    // // Misc
    // //

    auto createTemplateSpan(Expression expression, Node literal) -> TemplateSpan;
    // auto updateTemplateSpan(TemplateSpan node, Expression expression, Node literal) -> TemplateSpan;
    auto createSemicolonClassElement() -> SemicolonClassElement;

    // //
    // // Element
    // //

    auto createBlock(NodeArray<Statement> statements, boolean multiLine = false) -> Block;
    // auto updateBlock(Block node, NodeArray<Statement> statements) -> Block;
    // auto createVariableStatement(ModifiersArray modifiers, NodeArray<VariableDeclaration> declarationList) -> VariableStatement;
    auto createVariableStatement(ModifiersArray modifiers, VariableDeclarationList declarationList) -> VariableStatement;
    // auto updateVariableStatement(VariableStatement node, ModifiersArray modifiers, VariableDeclarationList declarationList) ->
    // VariableStatement;
    auto createEmptyStatement() -> EmptyStatement;
    auto createExpressionStatement(Expression expression) -> ExpressionStatement;
    // auto updateExpressionStatement(ExpressionStatement node, Expression expression) -> ExpressionStatement;
    auto createIfStatement(Expression expression, Statement thenStatement, Statement elseStatement = undefined) -> IfStatement;
    // auto updateIfStatement(IfStatement node, Expression expression, Statement thenStatement, Statement elseStatement) -> IfStatement;
    auto createDoStatement(Statement statement, Expression expression) -> DoStatement;
    // auto updateDoStatement(DoStatement node, Statement statement, Expression expression) -> DoStatement;
    auto createWhileStatement(Expression expression, Statement statement) -> WhileStatement;
    // auto updateWhileStatement(WhileStatement node, Expression expression, Statement statement) -> WhileStatement;
    auto createForStatement(ForInitializer initializer, Expression condition, Expression incrementor, Statement statement) -> ForStatement;
    // auto updateForStatement(ForStatement node, ForInitializer initializer, Expression condition, Expression incrementor, Statement
    // statement) -> ForStatement;
    auto createForInStatement(ForInitializer initializer, Expression expression, Statement statement) -> ForInStatement;
    // auto updateForInStatement(ForInStatement node, ForInitializer initializer, Expression expression, Statement statement) ->
    // ForInStatement;
    auto createForOfStatement(AwaitKeyword awaitModifier, ForInitializer initializer, Expression expression, Statement statement)
        -> ForOfStatement;
    // auto updateForOfStatement(ForOfStatement node, AwaitKeyword awaitModifier, ForInitializer initializer, Expression expression,
    // Statement statement) -> ForOfStatement;
    auto createContinueStatement(Identifier label = undefined) -> ContinueStatement;
    // auto createContinueStatement(string label) -> ContinueStatement;
    // auto updateContinueStatement(ContinueStatement node, Identifier label) -> ContinueStatement;
    // auto createBreakStatement(string label) -> BreakStatement;
    auto createBreakStatement(Identifier label = undefined) -> BreakStatement;
    // auto updateBreakStatement(BreakStatement node, Identifier label) -> BreakStatement;
    auto createReturnStatement(Expression expression = undefined) -> ReturnStatement;
    // auto updateReturnStatement(ReturnStatement node, Expression expression) -> ReturnStatement;
    auto createWithStatement(Expression expression, Statement statement) -> WithStatement;
    // auto updateWithStatement(WithStatement node, Expression expression, Statement statement) -> WithStatement;
    auto createSwitchStatement(Expression expression, CaseBlock caseBlock) -> SwitchStatement;
    // auto updateSwitchStatement(SwitchStatement node, Expression expression, CaseBlock caseBlock) -> SwitchStatement;
    // auto createLabeledStatement(string label, Statement statement) -> LabeledStatement;
    auto createLabeledStatement(Identifier label, Statement statement) -> LabeledStatement;
    // auto updateLabeledStatement(LabeledStatement node, Identifier label, Statement statement) -> LabeledStatement;
    auto createThrowStatement(Expression expression) -> ThrowStatement;
    // auto updateThrowStatement(ThrowStatement node, Expression expression) -> ThrowStatement;
    auto createTryStatement(Block tryBlock, CatchClause catchClause, Block finallyBlock) -> TryStatement;
    // auto updateTryStatement(TryStatement node, Block tryBlock, CatchClause catchClause, Block finallyBlock) -> TryStatement;
    auto createDebuggerStatement() -> DebuggerStatement;
    // auto createVariableDeclaration(string name, ExclamationToken exclamationToken = undefined, TypeNode type = undefined, Expression
    // initializer = undefined) -> VariableDeclaration;
    auto createVariableDeclaration(BindingName name, ExclamationToken exclamationToken = undefined, TypeNode type = undefined,
                                   Expression initializer = undefined) -> VariableDeclaration;
    // auto updateVariableDeclaration(VariableDeclaration node, BindingName name, ExclamationToken exclamationToken, TypeNode type,
    // Expression initializer) -> VariableDeclaration;
    auto createVariableDeclarationList(NodeArray<VariableDeclaration> declarations, NodeFlags flags = (NodeFlags)0)
        -> VariableDeclarationList;
    // auto updateVariableDeclarationList(VariableDeclarationList node, NodeArray<VariableDeclaration> declarations) ->
    // VariableDeclarationList; auto createFunctionDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, AsteriskToken
    // asteriskToken, string name, NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters, TypeNode
    // type, Block body) -> FunctionDeclaration;
    auto createFunctionDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, AsteriskToken asteriskToken, Identifier name,
                                   NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
                                   TypeNode type, Block body) -> FunctionDeclaration;
    // auto updateFunctionDeclaration(FunctionDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, AsteriskToken
    // asteriskToken, Identifier name, NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<ParameterDeclaration> parameters,
    // TypeNode type, Block body) -> FunctionDeclaration; auto createClassDeclaration(DecoratorsArray decorators, ModifiersArray modifiers,
    // string name, NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses, NodeArray<ClassElement>
    // members) -> ClassDeclaration;
    auto createClassDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
                                NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses,
                                NodeArray<ClassElement> members) -> ClassDeclaration;
    // auto updateClassDeclaration(ClassDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
    // NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses, NodeArray<ClassElement> members) ->
    // ClassDeclaration; auto createInterfaceDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, string name,
    // NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses, NodeArray<TypeElement> members) ->
    // InterfaceDeclaration;
    auto createInterfaceDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
                                    NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses,
                                    NodeArray<TypeElement> members) -> InterfaceDeclaration;
    // auto updateInterfaceDeclaration(InterfaceDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
    // NodeArray<TypeParameterDeclaration> typeParameters, NodeArray<HeritageClause> heritageClauses, NodeArray<TypeElement> members) ->
    // InterfaceDeclaration; auto createTypeAliasDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, string name,
    // NodeArray<TypeParameterDeclaration> typeParameters, TypeNode type) -> TypeAliasDeclaration;
    auto createTypeAliasDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
                                    NodeArray<TypeParameterDeclaration> typeParameters, TypeNode type) -> TypeAliasDeclaration;
    // auto updateTypeAliasDeclaration(TypeAliasDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
    // NodeArray<TypeParameterDeclaration> typeParameters, TypeNode type) -> TypeAliasDeclaration; auto
    // createEnumDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, string name, NodeArray<EnumMember> members) ->
    // EnumDeclaration;
    auto createEnumDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, Identifier name, NodeArray<EnumMember> members)
        -> EnumDeclaration;
    // auto updateEnumDeclaration(EnumDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, Identifier name,
    // NodeArray<EnumMember> members) -> EnumDeclaration;
    auto createModuleDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, ModuleName name, ModuleBody body,
                                 NodeFlags flags = (NodeFlags)0) -> ModuleDeclaration;
    // auto updateModuleDeclaration(ModuleDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, ModuleName name,
    // ModuleBody body) -> ModuleDeclaration;
    auto createModuleBlock(NodeArray<Statement> statements) -> ModuleBlock;
    // auto updateModuleBlock(ModuleBlock node, NodeArray<Statement> statements) -> ModuleBlock;
    auto createCaseBlock(NodeArray<CaseOrDefaultClause> clauses) -> CaseBlock;
    // auto updateCaseBlock(CaseBlock node, NodeArray<CaseOrDefaultClause> clauses) -> CaseBlock;
    // auto createNamespaceExportDeclaration(string name) -> NamespaceExportDeclaration;
    auto createNamespaceExportDeclaration(Identifier name) -> NamespaceExportDeclaration;
    // auto updateNamespaceExportDeclaration(NamespaceExportDeclaration node, Identifier name) -> NamespaceExportDeclaration;
    // auto createImportEqualsDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, boolean isTypeOnly, string name,
    // ModuleReference moduleReference) -> ImportEqualsDeclaration;
    auto createImportEqualsDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, boolean isTypeOnly, Identifier name,
                                       ModuleReference moduleReference) -> ImportEqualsDeclaration;
    // auto updateImportEqualsDeclaration(ImportEqualsDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, boolean
    // isTypeOnly, Identifier name, ModuleReference moduleReference) -> ImportEqualsDeclaration;
    auto createImportDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, ImportClause importClause,
                                 Expression moduleSpecifier) -> ImportDeclaration;
    // auto updateImportDeclaration(ImportDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, ImportClause importClause,
    // Expression moduleSpecifier) -> ImportDeclaration;
    auto createImportClause(boolean isTypeOnly, Identifier name, NamedImportBindings namedBindings) -> ImportClause;
    // auto updateImportClause(ImportClause node, boolean isTypeOnly, Identifier name, NamedImportBindings namedBindings) -> ImportClause;
    auto createNamespaceImport(Identifier name) -> NamespaceImport;
    // auto updateNamespaceImport(NamespaceImport node, Identifier name) -> NamespaceImport;
    auto createNamespaceExport(Identifier name) -> NamespaceExport;
    // auto updateNamespaceExport(NamespaceExport node, Identifier name) -> NamespaceExport;
    auto createNamedImports(NodeArray<ImportSpecifier> elements) -> NamedImports;
    // auto updateNamedImports(NamedImports node, NodeArray<ImportSpecifier> elements) -> NamedImports;
    auto createImportSpecifier(Identifier propertyName, Identifier name) -> ImportSpecifier;
    // auto updateImportSpecifier(ImportSpecifier node, Identifier propertyName, Identifier name) -> ImportSpecifier;
    auto createExportAssignment(DecoratorsArray decorators, ModifiersArray modifiers, boolean isExportEquals, Expression expression)
        -> ExportAssignment;
    // auto updateExportAssignment(ExportAssignment node, DecoratorsArray decorators, ModifiersArray modifiers, Expression expression) ->
    // ExportAssignment;
    auto createExportDeclaration(DecoratorsArray decorators, ModifiersArray modifiers, boolean isTypeOnly, NamedExportBindings exportClause,
                                 Expression moduleSpecifier = undefined) -> ExportDeclaration;
    // auto updateExportDeclaration(ExportDeclaration node, DecoratorsArray decorators, ModifiersArray modifiers, boolean isTypeOnly,
    // NamedExportBindings exportClause, Expression moduleSpecifier) -> ExportDeclaration;
    auto createNamedExports(NodeArray<ExportSpecifier> elements) -> NamedExports;
    // auto updateNamedExports(NamedExports node, NodeArray<ExportSpecifier> elements) -> NamedExports;
    // auto createExportSpecifier(string propertyName, string name) -> ExportSpecifier;
    auto createExportSpecifier(Identifier propertyName, Identifier name) -> ExportSpecifier;
    // auto updateExportSpecifier(ExportSpecifier node, Identifier propertyName, Identifier name) -> ExportSpecifier;
    /* @internal*/ auto createMissingDeclaration() -> MissingDeclaration;

    // //
    // // Module references
    // //

    auto createExternalModuleReference(Expression expression) -> ExternalModuleReference;
    // auto updateExternalModuleReference(ExternalModuleReference node, Expression expression) -> ExternalModuleReference;

    // //
    // // JSDoc
    // //

    auto getDefaultTagName(JSDocTag node) -> Identifier;
    auto createJSDocAllType() -> JSDocAllType;
    auto createJSDocUnknownType() -> JSDocUnknownType;
    auto createJSDocNonNullableType(TypeNode type) -> JSDocNonNullableType;
    // auto updateJSDocNonNullableType(JSDocNonNullableType node, TypeNode type) -> JSDocNonNullableType;
    auto createJSDocNullableType(TypeNode type) -> JSDocNullableType;
    // auto updateJSDocNullableType(JSDocNullableType node, TypeNode type) -> JSDocNullableType;
    auto createJSDocOptionalType(TypeNode type) -> JSDocOptionalType;
    // auto updateJSDocOptionalType(JSDocOptionalType node, TypeNode type) -> JSDocOptionalType;
    auto createJSDocFunctionType(NodeArray<ParameterDeclaration> parameters, TypeNode type) -> JSDocFunctionType;
    // auto updateJSDocFunctionType(JSDocFunctionType node, NodeArray<ParameterDeclaration> parameters, TypeNode type) -> JSDocFunctionType;
    auto createJSDocVariadicType(TypeNode type) -> JSDocVariadicType;
    // auto updateJSDocVariadicType(JSDocVariadicType node, TypeNode type) -> JSDocVariadicType;
    auto createJSDocNamepathType(TypeNode type) -> JSDocNamepathType;
    // auto updateJSDocNamepathType(JSDocNamepathType node, TypeNode type) -> JSDocNamepathType;
    auto createJSDocTypeExpression(TypeNode type) -> JSDocTypeExpression;
    // auto updateJSDocTypeExpression(JSDocTypeExpression node, TypeNode type) -> JSDocTypeExpression;
    auto createJSDocNameReference(EntityName name) -> JSDocNameReference;
    // auto updateJSDocNameReference(JSDocNameReference node, EntityName name) -> JSDocNameReference;
    auto createJSDocTypeLiteral(NodeArray<JSDocPropertyLikeTag> jsDocPropertyTags = undefined, boolean isArrayType = false)
        -> JSDocTypeLiteral;
    // auto updateJSDocTypeLiteral(JSDocTypeLiteral node, NodeArray<JSDocPropertyLikeTag> jsDocPropertyTags, boolean isArrayType) ->
    // JSDocTypeLiteral;
    auto createJSDocSignature(NodeArray<JSDocTemplateTag> typeParameters, NodeArray<JSDocParameterTag> parameters,
                              JSDocReturnTag type = undefined) -> JSDocSignature;
    // auto updateJSDocSignature(JSDocSignature node, NodeArray<JSDocTemplateTag> typeParameters, NodeArray<JSDocParameterTag> parameters,
    // JSDocReturnTag type) -> JSDocSignature;
    auto createJSDocTemplateTag(Identifier tagName, JSDocTypeExpression constraint, NodeArray<TypeParameterDeclaration> typeParameters,
                                string comment = string()) -> JSDocTemplateTag;
    // auto updateJSDocTemplateTag(JSDocTemplateTag node, Identifier tagName, JSDocTypeExpression constraint,
    // NodeArray<TypeParameterDeclaration> typeParameters, string comment) -> JSDocTemplateTag;
    auto createJSDocTypedefTag(Identifier tagName, /*JSDocTypeExpression | JSDocTypeLiteral*/ Node typeExpression = undefined,
                               /*Identifier | JSDocNamespaceDeclaration*/ Node fullName = undefined, string comment = string())
        -> JSDocTypedefTag;
    // auto updateJSDocTypedefTag(JSDocTypedefTag node, Identifier tagName, /*JSDocTypeExpression | JSDocTypeLiteral*/Node typeExpression,
    // /*Identifier | JSDocNamespaceDeclaration*/Node fullName, string comment) -> JSDocTypedefTag;
    auto createJSDocParameterTag(Identifier tagName, EntityName name, boolean isBracketed, JSDocTypeExpression typeExpression = undefined,
                                 boolean isNameFirst = false, string comment = string()) -> JSDocParameterTag;
    // auto updateJSDocParameterTag(JSDocParameterTag node, Identifier tagName, EntityName name, boolean isBracketed, JSDocTypeExpression
    // typeExpression, boolean isNameFirst, string comment) -> JSDocParameterTag;
    auto createJSDocPropertyTag(Identifier tagName, EntityName name, boolean isBracketed, JSDocTypeExpression typeExpression = undefined,
                                boolean isNameFirst = false, string comment = string()) -> JSDocPropertyTag;
    // auto updateJSDocPropertyTag(JSDocPropertyTag node, Identifier tagName, EntityName name, boolean isBracketed, JSDocTypeExpression
    // typeExpression, boolean isNameFirst, string comment) -> JSDocPropertyTag;
    auto createJSDocTypeTag(Identifier tagName, JSDocTypeExpression typeExpression, string comment = string()) -> JSDocTypeTag;
    // auto updateJSDocTypeTag(JSDocTypeTag node, Identifier tagName, JSDocTypeExpression typeExpression, string comment) -> JSDocTypeTag;
    auto createJSDocSeeTag(Identifier tagName, JSDocNameReference nameExpression, string comment = string()) -> JSDocSeeTag;
    // auto updateJSDocSeeTag(JSDocSeeTag node, Identifier tagName, JSDocNameReference nameExpression, string comment = string()) ->
    // JSDocSeeTag;
    auto createJSDocReturnTag(Identifier tagName, JSDocTypeExpression typeExpression = undefined, string comment = string())
        -> JSDocReturnTag;
    // auto updateJSDocReturnTag(JSDocReturnTag node, Identifier tagName, JSDocTypeExpression typeExpression, string comment) ->
    // JSDocReturnTag;
    auto createJSDocThisTag(Identifier tagName, JSDocTypeExpression typeExpression, string comment = string()) -> JSDocThisTag;
    // auto updateJSDocThisTag(JSDocThisTag node, Identifier tagName, JSDocTypeExpression typeExpression, string comment) -> JSDocThisTag;
    auto createJSDocEnumTag(Identifier tagName, JSDocTypeExpression typeExpression, string comment = string()) -> JSDocEnumTag;
    // auto updateJSDocEnumTag(JSDocEnumTag node, Identifier tagName, JSDocTypeExpression typeExpression, string comment) -> JSDocEnumTag;
    auto createJSDocCallbackTag(Identifier tagName, JSDocSignature typeExpression, /*Identifier | JSDocNamespaceDeclaration*/ Node fullName,
                                string comment = string()) -> JSDocCallbackTag;
    // auto updateJSDocCallbackTag(JSDocCallbackTag node, Identifier tagName, JSDocSignature typeExpression, /*Identifier |
    // JSDocNamespaceDeclaration*/Node fullName, string comment) -> JSDocCallbackTag;
    auto createJSDocAugmentsTag(Identifier tagName, JSDocAugmentsTag className, string comment = string()) -> JSDocAugmentsTag;
    // auto updateJSDocAugmentsTag(JSDocAugmentsTag node, Identifier tagName, JSDocAugmentsTag className, string comment) ->
    // JSDocAugmentsTag;
    auto createJSDocImplementsTag(Identifier tagName, JSDocImplementsTag className, string comment = string()) -> JSDocImplementsTag;
    // auto updateJSDocImplementsTag(JSDocImplementsTag node, Identifier tagName, JSDocImplementsTag className, string comment) ->
    // JSDocImplementsTag;
    auto createJSDocAuthorTag(Identifier tagName, string comment = string()) -> JSDocAuthorTag;
    // auto updateJSDocAuthorTag(JSDocAuthorTag node, Identifier tagName, string comment) -> JSDocAuthorTag;
    auto createJSDocClassTag(Identifier tagName, string comment = string()) -> JSDocClassTag;
    // auto updateJSDocClassTag(JSDocClassTag node, Identifier tagName, string comment) -> JSDocClassTag;
    auto createJSDocPublicTag(Identifier tagName, string comment = string()) -> JSDocPublicTag;
    // auto updateJSDocPublicTag(JSDocPublicTag node, Identifier tagName, string comment) -> JSDocPublicTag;
    auto createJSDocPrivateTag(Identifier tagName, string comment = string()) -> JSDocPrivateTag;
    // auto updateJSDocPrivateTag(JSDocPrivateTag node, Identifier tagName, string comment) -> JSDocPrivateTag;
    auto createJSDocProtectedTag(Identifier tagName, string comment = string()) -> JSDocProtectedTag;
    // auto updateJSDocProtectedTag(JSDocProtectedTag node, Identifier tagName, string comment) -> JSDocProtectedTag;
    auto createJSDocReadonlyTag(Identifier tagName, string comment = string()) -> JSDocReadonlyTag;
    // auto updateJSDocReadonlyTag(JSDocReadonlyTag node, Identifier tagName, string comment) -> JSDocReadonlyTag;
    auto createJSDocUnknownTag(Identifier tagName, string comment = string()) -> JSDocUnknownTag;
    // auto updateJSDocUnknownTag(JSDocUnknownTag node, Identifier tagName, string comment) -> JSDocUnknownTag;
    auto createJSDocDeprecatedTag(Identifier tagName, string comment = string()) -> JSDocDeprecatedTag;
    // auto updateJSDocDeprecatedTag(JSDocDeprecatedTag node, Identifier tagName, string comment = string()) -> JSDocDeprecatedTag;
    auto createJSDocComment(string comment = string(), NodeArray<JSDocTag> tags = undefined) -> JSDoc;
    // auto updateJSDocComment(JSDoc node, string comment, NodeArray<JSDocTag> tags) -> JSDoc;

    // //
    // // JSX
    // //

    auto createJsxElement(JsxOpeningElement openingElement, NodeArray<JsxChild> children, JsxClosingElement closingElement) -> JsxElement;
    // auto updateJsxElement(JsxElement node, JsxOpeningElement openingElement, NodeArray<JsxChild> children, JsxClosingElement
    // closingElement) -> JsxElement;
    auto createJsxSelfClosingElement(JsxTagNameExpression tagName, NodeArray<TypeNode> typeArguments, JsxAttributes attributes)
        -> JsxSelfClosingElement;
    // auto updateJsxSelfClosingElement(JsxSelfClosingElement node, JsxTagNameExpression tagName, NodeArray<TypeNode> typeArguments,
    // JsxAttributes attributes) -> JsxSelfClosingElement;
    auto createJsxOpeningElement(JsxTagNameExpression tagName, NodeArray<TypeNode> typeArguments, JsxAttributes attributes)
        -> JsxOpeningElement;
    // auto updateJsxOpeningElement(JsxOpeningElement node, JsxTagNameExpression tagName, NodeArray<TypeNode> typeArguments, JsxAttributes
    // attributes) -> JsxOpeningElement;
    auto createJsxClosingElement(JsxTagNameExpression tagName) -> JsxClosingElement;
    // auto updateJsxClosingElement(JsxClosingElement node, JsxTagNameExpression tagName) -> JsxClosingElement;
    auto createJsxFragment(JsxOpeningFragment openingFragment, NodeArray<JsxChild> children, JsxClosingFragment closingFragment)
        -> JsxFragment;
    auto createJsxText(string text, boolean containsOnlyTriviaWhiteSpaces = false) -> JsxText;
    // auto updateJsxText(JsxText node, string text, boolean containsOnlyTriviaWhiteSpaces = false) -> JsxText;
    auto createJsxOpeningFragment() -> JsxOpeningFragment;
    auto createJsxJsxClosingFragment() -> JsxClosingFragment;
    // auto updateJsxFragment(JsxFragment node, JsxOpeningFragment openingFragment, NodeArray<JsxChild> children, JsxClosingFragment
    // closingFragment) -> JsxFragment;
    auto createJsxAttribute(Identifier name, /*StringLiteral | JsxExpression*/ Node initializer) -> JsxAttribute;
    // auto updateJsxAttribute(JsxAttribute node, Identifier name, /*StringLiteral | JsxExpression*/Node initializer) -> JsxAttribute;
    auto createJsxAttributes(NodeArray<JsxAttributeLike> properties) -> JsxAttributes;
    // auto updateJsxAttributes(JsxAttributes node, NodeArray<JsxAttributeLike> properties) -> JsxAttributes;
    auto createJsxSpreadAttribute(Expression expression) -> JsxSpreadAttribute;
    // auto updateJsxSpreadAttribute(JsxSpreadAttribute node, Expression expression) -> JsxSpreadAttribute;
    auto createJsxExpression(DotDotDotToken dotDotDotToken, Expression expression) -> JsxExpression;
    // auto updateJsxExpression(JsxExpression node, Expression expression) -> JsxExpression;

    // //
    // // Clauses
    // //

    auto createCaseClause(Expression expression, NodeArray<Statement> statements) -> CaseClause;
    // auto updateCaseClause(CaseClause node, Expression expression, NodeArray<Statement> statements) -> CaseClause;
    auto createDefaultClause(NodeArray<Statement> statements) -> DefaultClause;
    // auto updateDefaultClause(DefaultClause node, NodeArray<Statement> statements) -> DefaultClause;
    auto createHeritageClause(/*HeritageClause*/ SyntaxKind token, NodeArray<ExpressionWithTypeArguments> types) -> HeritageClause;
    // auto updateHeritageClause(HeritageClause node, NodeArray<ExpressionWithTypeArguments> types) -> HeritageClause;
    // auto createCatchClause(string variableDeclaration, Block block) -> CatchClause;
    auto createCatchClause(VariableDeclaration variableDeclaration, Block block) -> CatchClause;
    // auto updateCatchClause(CatchClause node, VariableDeclaration variableDeclaration, Block block) -> CatchClause;

    // //
    // // Property assignments
    // //

    // auto createPropertyAssignment(string name, Expression initializer) -> PropertyAssignment;
    auto createPropertyAssignment(PropertyName name, Expression initializer) -> PropertyAssignment;
    // auto updatePropertyAssignment(PropertyAssignment node, PropertyName name, Expression initializer) -> PropertyAssignment;
    // auto createShorthandPropertyAssignment(string name, Expression objectAssignmentInitializer = undefined) ->
    // ShorthandPropertyAssignment;
    auto createShorthandPropertyAssignment(Identifier name, Expression objectAssignmentInitializer = undefined)
        -> ShorthandPropertyAssignment;
    // auto updateShorthandPropertyAssignment(ShorthandPropertyAssignment node, Identifier name, Expression objectAssignmentInitializer) ->
    // ShorthandPropertyAssignment;
    auto createSpreadAssignment(Expression expression) -> SpreadAssignment;
    // auto updateSpreadAssignment(SpreadAssignment node, Expression expression) -> SpreadAssignment;

    // //
    // // Enum
    // //

    // auto createEnumMember(string name, Expression initializer = undefined) -> EnumMember;
    auto createEnumMember(PropertyName name, Expression initializer = undefined) -> EnumMember;
    // auto updateEnumMember(EnumMember node, PropertyName name, Expression initializer) -> EnumMember;

    // //
    // // Top-level nodes
    // //

    auto createSourceFile(NodeArray<Statement> statements, EndOfFileToken endOfFileToken, NodeFlags flags) -> SourceFile;
    auto cloneSourceFileWithChanges(SourceFile node, NodeArray<Statement> statements, boolean isDeclarationFile,
                                    NodeArray<FileReference> referencedFiles, NodeArray<FileReference> typeReferences,
                                    boolean hasNoDefaultLib, NodeArray<FileReference> libReferences) -> SourceFile;
    auto updateSourceFile(SourceFile node, NodeArray<Statement> statements, boolean isDeclarationFile = false,
                          NodeArray<FileReference> referencedFiles = undefined, NodeArray<FileReference> typeReferences = undefined,
                          boolean hasNoDefaultLib = false, NodeArray<FileReference> libReferences = undefined) -> SourceFile;

    // /* @internal */ auto createUnparsedSource(NodeArray<UnparsedPrologue> prologues, NodeArray<UnparsedSyntheticReference>
    // syntheticReferences, NodeArray<UnparsedSourceText> texts) -> UnparsedSource;
    // /* @internal */ auto createUnparsedPrologue(string data = string()) -> UnparsedPrologue;
    // /* @internal */ auto createUnparsedPrepend(string data, NodeArray<UnparsedSourceText> texts) -> UnparsedPrepend;
    // /* @internal */ auto createUnparsedTextLike(string data, boolean internal) -> UnparsedTextLike;
    // /* @internal */ auto createUnparsedSyntheticReference(/*BundleFileHasNoDefaultLib | BundleFileReference*/Node section) ->
    // UnparsedSyntheticReference;
    // /* @internal */ auto createInputFiles() -> InputFiles;

    // //
    // // Synthetic Nodes
    // //
    // /* @internal */ auto createSyntheticExpression(SignatureFlags type, boolean isSpread = false, /*NamedTupleMember |
    // ParameterDeclaration*/Node tupleNameSource = undefined) -> SyntheticExpression;
    // /* @internal */ auto createSyntaxList(NodeArray<Node> children) -> SyntaxList;

    // //
    // // Transformation nodes
    // //

    // auto createNotEmittedStatement(Node original) -> NotEmittedStatement;
    // /* @internal */ auto createEndOfDeclarationMarker(Node original) -> EndOfDeclarationMarker;
    // /* @internal */ auto createMergeDeclarationMarker(Node original) -> MergeDeclarationMarker;
    // auto createPartiallyEmittedExpression(Expression expression, Node original = undefined) -> PartiallyEmittedExpression;
    // auto updatePartiallyEmittedExpression(PartiallyEmittedExpression node, Expression expression) -> PartiallyEmittedExpression;
    // /* @internal */ auto createSyntheticReferenceExpression(Expression expression, Expression thisArg) -> SyntheticReferenceExpression;
    // /* @internal */ auto updateSyntheticReferenceExpression(SyntheticReferenceExpression node, Expression expression, Expression thisArg)
    // -> SyntheticReferenceExpression; auto createCommaListExpression(NodeArray<Expression> elements) -> CommaListExpression; auto
    // updateCommaListExpression(CommaListExpression node, NodeArray<Expression> elements) -> CommaListExpression; auto
    // createBundle(NodeArray<SourceFile> sourceFiles, NodeArray</*UnparsedSource | InputFiles*/Node> prepends = undefined) -> Bundle; auto
    // updateBundle(Bundle node, NodeArray<SourceFile> sourceFiles, NodeArray</*UnparsedSource | InputFiles*/Node> prepends = undefined) ->
    // Bundle;

    // //
    // // Common operators
    // //

    // auto createComma(Expression left, Expression right) -> BinaryExpression;
    // auto createAssignment(Expression left, Expression right) -> /*DestructuringAssignment | AssignmentExpression<EqualsToken>*/Node;
    // auto createLogicalOr(Expression left, Expression right) -> BinaryExpression;
    // auto createLogicalAnd(Expression left, Expression right) -> BinaryExpression;
    // auto createBitwiseOr(Expression left, Expression right) -> BinaryExpression;
    // auto createBitwiseXor(Expression left, Expression right) -> BinaryExpression;
    // auto createBitwiseAnd(Expression left, Expression right) -> BinaryExpression;
    // auto createStrictEquality(Expression left, Expression right) -> BinaryExpression;
    // auto createStrictInequality(Expression left, Expression right) -> BinaryExpression;
    // auto createEquality(Expression left, Expression right) -> BinaryExpression;
    // auto createInequality(Expression left, Expression right) -> BinaryExpression;
    // auto createLessThan(Expression left, Expression right) -> BinaryExpression;
    // auto createLessThanEquals(Expression left, Expression right) -> BinaryExpression;
    // auto createGreaterThan(Expression left, Expression right) -> BinaryExpression;
    // auto createGreaterThanEquals(Expression left, Expression right) -> BinaryExpression;
    // auto createLeftShift(Expression left, Expression right) -> BinaryExpression;
    // auto createRightShift(Expression left, Expression right) -> BinaryExpression;
    // auto createUnsignedRightShift(Expression left, Expression right) -> BinaryExpression;
    // auto createAdd(Expression left, Expression right) -> BinaryExpression;
    // auto createSubtract(Expression left, Expression right) -> BinaryExpression;
    // auto createMultiply(Expression left, Expression right) -> BinaryExpression;
    // auto createDivide(Expression left, Expression right) -> BinaryExpression;
    // auto createModulo(Expression left, Expression right) -> BinaryExpression;
    // auto createExponent(Expression left, Expression right) -> BinaryExpression;
    // auto createPrefixPlus(Expression operand) -> PrefixUnaryExpression;
    // auto createPrefixMinus(Expression operand) -> PrefixUnaryExpression;
    // auto createPrefixIncrement(Expression operand) -> PrefixUnaryExpression;
    // auto createPrefixDecrement(Expression operand) -> PrefixUnaryExpression;
    // auto createBitwiseNot(Expression operand) -> PrefixUnaryExpression;
    // auto createLogicalNot(Expression operand) -> PrefixUnaryExpression;
    // auto createPostfixIncrement(Expression operand) -> PostfixUnaryExpression;
    // auto createPostfixDecrement(Expression operand) -> PostfixUnaryExpression;

    // //
    // // Compound Nodes
    // //

    // auto createImmediatelyInvokedFunctionExpression(NodeArray<Statement> statements) -> CallExpression;
    // auto createImmediatelyInvokedFunctionExpression(NodeArray<Statement> statements, ParameterDeclaration param, Expression paramValue)
    // -> CallExpression; auto createImmediatelyInvokedArrowFunction(NodeArray<Statement> statements) -> CallExpression; auto
    // createImmediatelyInvokedArrowFunction(NodeArray<Statement> statements, ParameterDeclaration param, Expression paramValue) ->
    // CallExpression;

    // auto createVoidZero() -> VoidExpression;
    // auto createExportDefault(Expression expression) -> ExportAssignment;
    // auto createExternalModuleExport(Identifier exportName) -> ExportDeclaration;

    // /* @internal */ auto createTypeCheck(Expression value, string tag) -> Expression;
    // /* @internal */ auto createMethodCall(Expression object, string methodName, NodeArray<Expression> argumentsList) -> CallExpression;
    // /* @internal */ auto createMethodCall(Expression object, Identifier methodName, NodeArray<Expression> argumentsList) ->
    // CallExpression;
    // /* @internal */ auto createGlobalMethodCall(string globalObjectName, string globalMethodName, NodeArray<Expression> argumentsList) ->
    // CallExpression;
    // /* @internal */ auto createFunctionBindCall(Expression target, Expression thisArg, NodeArray<Expression> argumentsList) ->
    // CallExpression;
    // /* @internal */ auto createFunctionCallCall(Expression target, Expression thisArg, NodeArray<Expression> argumentsList) ->
    // CallExpression;
    // /* @internal */ auto createFunctionApplyCall(Expression target, Expression thisArg, Expression argumentsExpression) ->
    // CallExpression;
    // /* @internal */ auto createObjectDefinePropertyCall(Expression target, string propertyName, Expression attributes) -> CallExpression;
    // /* @internal */ auto createObjectDefinePropertyCall(Expression target, Expression propertyName, Expression attributes) ->
    // CallExpression;
    // /* @internal */ auto createPropertyDescriptor(PropertyDescriptorAttributes attributes, boolean singleLine = false) ->
    // ObjectLiteralExpression;
    // /* @internal */ auto createArraySliceCall(Expression array, number start) -> CallExpression;
    // /* @internal */ auto createArraySliceCall(Expression array, Expression start = undefined) -> CallExpression;
    // /* @internal */ auto createArrayConcatCall(Expression array, NodeArray<Expression> values) -> CallExpression;
    // /* @internal */ auto createCallBinding(Expression expression, std::function<void(Identifier)> recordTempVariable, ScriptTarget
    // languageVersion = (ScriptTarget)0, boolean cacheIdentifiers = false) -> CallBinding;
    // /* @internal */ auto inlineExpressions(NodeArray<Expression> expressions) -> Expression;
    // /**
    //  * Gets the internal name of a declaration. This is primarily used for declarations that can be
    //  * referred to by name in the body of an ES5 class function body. An internal name will *never*
    //  * be prefixed with an module or namespace export modifier like "exports." when emitted as an
    //  * expression. An internal name will also *never* be renamed due to a collision with a block
    //  * scoped variable.
    //  *
    //  * @param node The declaration.
    //  * @param allowComments A value indicating whether comments may be emitted for the name.
    //  * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
    //  */
    // /* @internal */ auto getInternalName(Declaration node, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    // /**
    //  * Gets the local name of a declaration. This is primarily used for declarations that can be
    //  * referred to by name in the declaration's immediate scope (classes, enums, namespaces). A
    //  * local name will *never* be prefixed with an module or namespace export modifier like
    //  * "exports." when emitted as an expression.
    //  *
    //  * @param node The declaration.
    //  * @param allowComments A value indicating whether comments may be emitted for the name.
    //  * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
    //  */
    // /* @internal */ auto getLocalName(Declaration node, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    // /**
    //  * Gets the export name of a declaration. This is primarily used for declarations that can be
    //  * referred to by name in the declaration's immediate scope (classes, enums, namespaces). An
    //  * export name will *always* be prefixed with a module or namespace export modifier like
    //  * `"exports."` when emitted as an expression if the name points to an exported symbol.
    //  *
    //  * @param node The declaration.
    //  * @param allowComments A value indicating whether comments may be emitted for the name.
    //  * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
    //  */
    // /* @internal */ auto getExportName(Declaration node, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    // /**
    //  * Gets the name of a declaration for use in declarations.
    //  *
    //  * @param node The declaration.
    //  * @param allowComments A value indicating whether comments may be emitted for the name.
    //  * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
    //  */
    // /* @internal */ auto getDeclarationName(Declaration node, boolean allowComments = false, boolean allowSourceMaps = false) ->
    // Identifier;
    // /**
    //  * Gets a namespace-qualified name for use in expressions.
    //  *
    //  * @param ns The namespace identifier.
    //  * @param name The name.
    //  * @param allowComments A value indicating whether comments may be emitted for the name.
    //  * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
    //  */
    // /* @internal */ auto getNamespaceMemberName(Identifier ns, Identifier name, boolean allowComments = false, boolean allowSourceMaps =
    // false) -> PropertyAccessExpression;
    // /**
    //  * Gets the exported name of a declaration for use in expressions.
    //  *
    //  * An exported name will *always* be prefixed with an module or namespace export modifier like
    //  * "exports." if the name points to an exported symbol.
    //  *
    //  * @param ns The namespace identifier.
    //  * @param node The declaration.
    //  * @param allowComments A value indicating whether comments may be emitted for the name.
    //  * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
    //  */
    // /* @internal */ auto getExternalModuleOrNamespaceExportName(Identifier ns, Declaration node, boolean allowComments = false, boolean
    // allowSourceMaps = false) -> /*Identifier | PropertyAccessExpression*/Node;

    // //
    // // Utilities
    // //

    // auto restoreOuterExpressions(Expression outerExpression, Expression innerExpression, OuterExpressionKinds kinds =
    // OuterExpressionKinds::None) -> Expression;
    // /* @internal */ auto restoreEnclosingLabel(Statement node, LabeledStatement outermostLabeledStatement,
    // std::function<void(LabeledStatement)> afterRestoreLabelCallback = nullptr) -> Statement;
    // /* @internal */ auto createUseStrictPrologue() -> PrologueDirective;
    // /**
    //  * Copies any necessary standard and custom prologue-directives into target array.
    //  * @param source origin statements array
    //  * @param target result statements array
    //  * @param ensureUseStrict boolean determining whether the function need to add prologue-directives
    //  * @param visitor Optional callback used to visit any custom prologue directives.
    //  */
    // /* @internal */ auto copyPrologue(NodeArray<Statement> source, Push<Statement> target, boolean ensureUseStrict = false,
    // std::function<VisitResult<Node>(Node)> visitor = nullptr) -> number;
    // /**
    //  * Copies only the standard (string-expression) prologue-directives into the target statement-array.
    //  * @param source origin statements array
    //  * @param target result statements array
    //  * @param ensureUseStrict boolean determining whether the function need to add prologue-directives
    //  */
    // /* @internal */ auto copyStandardPrologue(NodeArray<Statement> source, Push<Statement> target, boolean ensureUseStrict = false) ->
    // number;
    // /**
    //  * Copies only the custom prologue-directives into target statement-array.
    //  * @param source origin statements array
    //  * @param target result statements array
    //  * @param statementOffset The offset at which to begin the copy.
    //  * @param visitor Optional callback used to visit any custom prologue directives.
    //  */
    // /* @internal */ auto copyCustomPrologue(NodeArray<Statement> source, Push<Statement> target, number statementOffset,
    // std::function<VisitResult<Node>(Node)> visitor = nullptr, std::function<boolean(Node)> filter = nullptr) -> number;
    // /* @internal */ auto ensureUseStrict(NodeArray<Statement> statements) -> NodeArray<Statement>;
    // /* @internal */ auto liftToBlock(NodeArray<Node> nodes) -> Statement;
    // /**
    //  * Merges generated lexical declarations into a new statement list.
    //  */
    // /* @internal */ auto mergeLexicalEnvironment(NodeArray<Statement> statements, NodeArray<Statement> declarations) ->
    // NodeArray<Statement>;
    // /**
    //  * Creates a shallow, memberwise clone of a node.
    //  * - The result will have its `original` pointer set to `node`.
    //  * - The result will have its `pos` and `end` set to `-1`.
    //  * - *DO NOT USE THIS* if a more appropriate function is available.
    //  */
    // /* @internal */ template <typename T/*extends Node*/> auto cloneNode(T node) -> T;
    // /* @internal */ template <typename T/*extends HasModifiers*/> auto updateModifiers(T node, ModifierFlags modifiers) -> T;
    // /* @internal */ template <typename T/*extends HasModifiers*/> auto updateModifiers(T node, ModifiersArray modifiers) -> T;
};
} // namespace ts

#endif // NODEFACTORY_H