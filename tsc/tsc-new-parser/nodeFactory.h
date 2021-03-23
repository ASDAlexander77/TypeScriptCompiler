#ifndef NODEFACTORY_H
#define NODEFACTORY_H

#include "enums.h"
#include "scanner.h"

typedef std::function<Node(SyntaxKind)> NodeCreate;

enum class NodeFactoryFlags : number {
    None = 0,
    // Disables the parenthesizer rules for the factory.
    NoParenthesizerRules = 1 << 0,
    // Disables the node converters for the factory.
    NoNodeConverters = 1 << 1,
    // Ensures new `PropertyAccessExpression` nodes are created with the `NoIndentation` emit flag set.
    NoIndentationOnFreshPropertyAccess = 1 << 2,
    // Do not set an `original` pointer when updating a node.
    NoOriginalNode = 1 << 3,
};

static NodeFactoryFlags operator |(NodeFactoryFlags lhs, NodeFactoryFlags rhs)
{
    return (NodeFactoryFlags) ((number) lhs | (number) rhs);
}

struct BaseNodeFactory
{
    BaseNodeFactory() = default;

    NodeCreate createBaseSourceFileNode;
    NodeCreate createBaseIdentifierNode;
    NodeCreate createBasePrivateIdentifierNode;
    NodeCreate createBaseTokenNode;
    NodeCreate createBaseNode;
};

class NodeFactory
{
    NodeFactoryFlags nodeFactoryFlags;
    BaseNodeFactory baseNodeFactory;

public:
    NodeFactory(NodeFactoryFlags nodeFactoryFlags, BaseNodeFactory baseNodeFactory) : nodeFactoryFlags(nodeFactoryFlags), baseNodeFactory(baseNodeFactory) {}

    /* @internal */ parenthesizer: ParenthesizerRules;
    /* @internal */ converters: NodeConverters;
    template <typename T> createNodeArray(elements?: T[], boolean hasTrailingComma = false) -> NodeArray<T>;

    //
    // Literals
    //

    createNumericLiteral(value: string | number, TokenFlags numericLiteralFlags = (TokenFlags)0) -> NumericLiteral;
    createBigIntLiteral(value: string | PseudoBigInt) -> BigIntLiteral;
    createStringLiteral(text: string, boolean isSingleQuote = false) -> StringLiteral;
    /* @internal*/ createStringLiteral(text: string, boolean isSingleQuote = false, boolean hasExtendedUnicodeEscape = false) -> StringLiteral; // eslint-disable-line @typescript-eslint/unified-signatures
    createStringLiteralFromNode(sourceNode: PropertyNameLiteral, boolean isSingleQuote = false) -> StringLiteral;
    createRegularExpressionLiteral(text: string) -> RegularExpressionLiteral;

    //
    // Identifiers
    //

    createIdentifier(text: string) -> Identifier;
    /* @internal */ createIdentifier(text: string, typeArguments?: (TypeNode | TypeParameterDeclaration)[], originalKeywordKind?: SyntaxKind) -> Identifier; // eslint-disable-line @typescript-eslint/unified-signatures
    /* @internal */ updateIdentifier(node: Identifier, typeArguments: NodeArray</*TypeNode | TypeParameterDeclaration*/Node>) -> Identifier;

    /**
     * Create a unique temporary variable.
     * @param recordTempVariable An optional callback used to record the temporary variable name. This
     * should usually be a reference to `hoistVariableDeclaration` from a `TransformationContext`, but
     * can be `undefined` if you plan to record the temporary variable manually.
     * @param reservedInNestedScopes When `true`, reserves the temporary variable name in all nested scopes
     * during emit so that the variable can be referenced in a nested function body. This is an alternative to
     * setting `EmitFlags.ReuseTempVariableScope` on the nested function itself.
     */
    createTempVariable(recordTempVariable: ((node: Identifier) => void), boolean reservedInNestedScopes = false) -> Identifier;

    /**
     * Create a unique temporary variable for use in a loop.
     * @param reservedInNestedScopes When `true`, reserves the temporary variable name in all nested scopes
     * during emit so that the variable can be referenced in a nested function body. This is an alternative to
     * setting `EmitFlags.ReuseTempVariableScope` on the nested function itself.
     */
    createLoopVariable(boolean reservedInNestedScopes = false) -> Identifier;

    /** Create a unique name based on the supplied text. */
    createUniqueName(text: string, GeneratedIdentifierFlags flags = (GeneratedIdentifierFlags)0) -> Identifier;

    /** Create a unique name generated for a node. */
    getGeneratedNameForNode(node: Node, GeneratedIdentifierFlags flags = (GeneratedIdentifierFlags)0) -> Identifier;

    createPrivateIdentifier(text: string) -> PrivateIdentifier

    //
    // Punctuation
    //

    createToken(token: SyntaxKind.SuperKeyword) -> SuperExpression;
    createToken(token: SyntaxKind.ThisKeyword) -> ThisExpression;
    createToken(token: SyntaxKind.NullKeyword) -> NullLiteral;
    createToken(token: SyntaxKind.TrueKeyword) -> TrueLiteral;
    createToken(token: SyntaxKind.FalseKeyword) -> FalseLiteral;
    createToken<TKind extends PunctuationSyntaxKind>(token: TKind) -> PunctuationToken<TKind>;
    createToken<TKind extends KeywordTypeSyntaxKind>(token: TKind) -> KeywordTypeNode<TKind>;
    createToken<TKind extends ModifierSyntaxKind>(token: TKind) -> ModifierToken<TKind>;
    createToken<TKind extends KeywordSyntaxKind>(token: TKind) -> KeywordToken<TKind>;
    createToken<TKind extends SyntaxKind.Unknown | SyntaxKind.EndOfFileToken>(token: TKind) -> Token<TKind>;
    /*@internal*/ createToken<TKind extends SyntaxKind>(token: TKind) -> Token<TKind>;

    //
    // Reserved words
    //

    createSuper() -> SuperExpression;
    createThis() -> ThisExpression;
    createNull() -> NullLiteral;
    createTrue() -> TrueLiteral;
    createFalse() -> FalseLiteral;

    //
    // Modifiers
    //

    template <typename T/*extends ModifierSyntaxKind*/> createModifier(kind: T) -> ModifierToken<T>;
    createModifiersFromModifierFlags(flags: ModifierFlags) -> Modifier[];

    //
    // Names
    //

    createQualifiedName(left: EntityName, right: string | Identifier) -> QualifiedName;
    updateQualifiedName(node: QualifiedName, left: EntityName, right: Identifier) -> QualifiedName;
    createComputedPropertyName(expression: Expression) -> ComputedPropertyName;
    updateComputedPropertyName(node: ComputedPropertyName, expression: Expression) -> ComputedPropertyName;

    //
    // Signature elements
    //

    createTypeParameterDeclaration(name: string | Identifier, constraint?: TypeNode, defaultType?: TypeNode) -> TypeParameterDeclaration;
    updateTypeParameterDeclaration(node: TypeParameterDeclaration, name: Identifier, constraint: TypeNode, defaultType: TypeNode) -> TypeParameterDeclaration;
    createParameterDeclaration(decorators: Decorator[], modifiers: Modifier[], dotDotDotToken: DotDotDotToken, name: string | BindingName, questionToken?: QuestionToken, type?: TypeNode, initializer?: Expression) -> ParameterDeclaration;
    updateParameterDeclaration(node: ParameterDeclaration, decorators: Decorator[], modifiers: Modifier[], dotDotDotToken: DotDotDotToken, name: string | BindingName, questionToken: QuestionToken, type: TypeNode, initializer: Expression) -> ParameterDeclaration;
    createDecorator(expression: Expression) -> Decorator;
    updateDecorator(node: Decorator, expression: Expression) -> Decorator;

    //
    // Type Elements
    //

    createPropertySignature(modifiers: Modifier[], name: PropertyName | string, questionToken: QuestionToken, type: TypeNode) -> PropertySignature;
    updatePropertySignature(node: PropertySignature, modifiers: Modifier[], name: PropertyName, questionToken: QuestionToken, type: TypeNode) -> PropertySignature;
    createPropertyDeclaration(decorators: Decorator[], modifiers: Modifier[], name: string | PropertyName, questionOrExclamationToken: QuestionToken | ExclamationToken, type: TypeNode, initializer: Expression) -> PropertyDeclaration;
    updatePropertyDeclaration(node: PropertyDeclaration, decorators: Decorator[], modifiers: Modifier[], name: string | PropertyName, questionOrExclamationToken: QuestionToken | ExclamationToken, type: TypeNode, initializer: Expression) -> PropertyDeclaration;
    createMethodSignature(modifiers: Modifier[], name: string | PropertyName, questionToken: QuestionToken, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> MethodSignature;
    updateMethodSignature(node: MethodSignature, modifiers: Modifier[], name: PropertyName, questionToken: QuestionToken, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> MethodSignature;
    createMethodDeclaration(decorators: Decorator[], modifiers: Modifier[], asteriskToken: AsteriskToken, name: string | PropertyName, questionToken: QuestionToken, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> MethodDeclaration;
    updateMethodDeclaration(node: MethodDeclaration, decorators: Decorator[], modifiers: Modifier[], asteriskToken: AsteriskToken, name: PropertyName, questionToken: QuestionToken, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> MethodDeclaration;
    createConstructorDeclaration(decorators: Decorator[], modifiers: Modifier[], parameters: ParameterDeclaration[], body: Block) -> ConstructorDeclaration;
    updateConstructorDeclaration(node: ConstructorDeclaration, decorators: Decorator[], modifiers: Modifier[], parameters: ParameterDeclaration[], body: Block) -> ConstructorDeclaration;
    createGetAccessorDeclaration(decorators: Decorator[], modifiers: Modifier[], name: string | PropertyName, parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> GetAccessorDeclaration;
    updateGetAccessorDeclaration(node: GetAccessorDeclaration, decorators: Decorator[], modifiers: Modifier[], name: PropertyName, parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> GetAccessorDeclaration;
    createSetAccessorDeclaration(decorators: Decorator[], modifiers: Modifier[], name: string | PropertyName, parameters: ParameterDeclaration[], body: Block) -> SetAccessorDeclaration;
    updateSetAccessorDeclaration(node: SetAccessorDeclaration, decorators: Decorator[], modifiers: Modifier[], name: PropertyName, parameters: ParameterDeclaration[], body: Block) -> SetAccessorDeclaration;
    createCallSignature(typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> CallSignatureDeclaration;
    updateCallSignature(node: CallSignatureDeclaration, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> CallSignatureDeclaration;
    createConstructSignature(typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> ConstructSignatureDeclaration;
    updateConstructSignature(node: ConstructSignatureDeclaration, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> ConstructSignatureDeclaration;
    createIndexSignature(decorators: Decorator[], modifiers: Modifier[], parameters: ParameterDeclaration[], type: TypeNode) -> IndexSignatureDeclaration;
    /* @internal */ createIndexSignature(decorators: Decorator[], modifiers: Modifier[], parameters: ParameterDeclaration[], type: TypeNode) -> IndexSignatureDeclaration; // eslint-disable-line @typescript-eslint/unified-signatures
    updateIndexSignature(node: IndexSignatureDeclaration, decorators: Decorator[], modifiers: Modifier[], parameters: ParameterDeclaration[], type: TypeNode) -> IndexSignatureDeclaration;
    createTemplateLiteralTypeSpan(type: TypeNode, literal: TemplateMiddle | TemplateTail) -> TemplateLiteralTypeSpan;
    updateTemplateLiteralTypeSpan(node: TemplateLiteralTypeSpan, type: TypeNode, literal: TemplateMiddle | TemplateTail) -> TemplateLiteralTypeSpan;

    //
    // Types
    //

    createKeywordTypeNode<TKind extends KeywordTypeSyntaxKind>(kind: TKind) -> KeywordTypeNode<TKind>;
    createTypePredicateNode(assertsModifier: AssertsKeyword, parameterName: Identifier | ThisTypeNode | string, type: TypeNode) -> TypePredicateNode;
    updateTypePredicateNode(node: TypePredicateNode, assertsModifier: AssertsKeyword, parameterName: Identifier | ThisTypeNode, type: TypeNode) -> TypePredicateNode;
    createTypeReferenceNode(typeName: string | EntityName, typeArguments?: TypeNode[]) -> TypeReferenceNode;
    updateTypeReferenceNode(node: TypeReferenceNode, typeName: EntityName, typeArguments: NodeArray<TypeNode>) -> TypeReferenceNode;
    createFunctionTypeNode(typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> FunctionTypeNode;
    updateFunctionTypeNode(node: FunctionTypeNode, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> FunctionTypeNode;
    createConstructorTypeNode(modifiers: Modifier[], typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> ConstructorTypeNode;
    /** @deprecated */
    createConstructorTypeNode(typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> ConstructorTypeNode;
    updateConstructorTypeNode(node: ConstructorTypeNode, modifiers: Modifier[], typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> ConstructorTypeNode;
    /** @deprecated */
    updateConstructorTypeNode(node: ConstructorTypeNode, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> ConstructorTypeNode;
    createTypeQueryNode(exprName: EntityName) -> TypeQueryNode;
    updateTypeQueryNode(node: TypeQueryNode, exprName: EntityName) -> TypeQueryNode;
    createTypeLiteralNode(members: TypeElement[]) -> TypeLiteralNode;
    updateTypeLiteralNode(node: TypeLiteralNode, members: NodeArray<TypeElement>) -> TypeLiteralNode;
    createArrayTypeNode(elementType: TypeNode) -> ArrayTypeNode;
    updateArrayTypeNode(node: ArrayTypeNode, elementType: TypeNode) -> ArrayTypeNode;
    createTupleTypeNode(elements: (TypeNode | NamedTupleMember)[]) -> TupleTypeNode;
    updateTupleTypeNode(node: TupleTypeNode, elements: (TypeNode | NamedTupleMember)[]) -> TupleTypeNode;
    createNamedTupleMember(dotDotDotToken: DotDotDotToken, name: Identifier, questionToken: QuestionToken, type: TypeNode) -> NamedTupleMember;
    updateNamedTupleMember(node: NamedTupleMember, dotDotDotToken: DotDotDotToken, name: Identifier, questionToken: QuestionToken, type: TypeNode) -> NamedTupleMember;
    createOptionalTypeNode(type: TypeNode) -> OptionalTypeNode;
    updateOptionalTypeNode(node: OptionalTypeNode, type: TypeNode) -> OptionalTypeNode;
    createRestTypeNode(type: TypeNode) -> RestTypeNode;
    updateRestTypeNode(node: RestTypeNode, type: TypeNode) -> RestTypeNode;
    createUnionTypeNode(types: TypeNode[]) -> UnionTypeNode;
    updateUnionTypeNode(node: UnionTypeNode, types: NodeArray<TypeNode>) -> UnionTypeNode;
    createIntersectionTypeNode(types: TypeNode[]) -> IntersectionTypeNode;
    updateIntersectionTypeNode(node: IntersectionTypeNode, types: NodeArray<TypeNode>) -> IntersectionTypeNode;
    createConditionalTypeNode(checkType: TypeNode, extendsType: TypeNode, trueType: TypeNode, falseType: TypeNode) -> ConditionalTypeNode;
    updateConditionalTypeNode(node: ConditionalTypeNode, checkType: TypeNode, extendsType: TypeNode, trueType: TypeNode, falseType: TypeNode) -> ConditionalTypeNode;
    createInferTypeNode(typeParameter: TypeParameterDeclaration) -> InferTypeNode;
    updateInferTypeNode(node: InferTypeNode, typeParameter: TypeParameterDeclaration) -> InferTypeNode;
    createImportTypeNode(argument: TypeNode, qualifier?: EntityName, typeArguments?: TypeNode[], boolean isTypeOf = false) -> ImportTypeNode;
    updateImportTypeNode(node: ImportTypeNode, argument: TypeNode, qualifier: EntityName, typeArguments: TypeNode[], boolean isTypeOf = false) -> ImportTypeNode;
    createParenthesizedType(type: TypeNode) -> ParenthesizedTypeNode;
    updateParenthesizedType(node: ParenthesizedTypeNode, type: TypeNode) -> ParenthesizedTypeNode;
    createThisTypeNode() -> ThisTypeNode;
    createTypeOperatorNode(operator: SyntaxKind.KeyOfKeyword | SyntaxKind.UniqueKeyword | SyntaxKind.ReadonlyKeyword, type: TypeNode) -> TypeOperatorNode;
    updateTypeOperatorNode(node: TypeOperatorNode, type: TypeNode) -> TypeOperatorNode;
    createIndexedAccessTypeNode(objectType: TypeNode, indexType: TypeNode) -> IndexedAccessTypeNode;
    updateIndexedAccessTypeNode(node: IndexedAccessTypeNode, objectType: TypeNode, indexType: TypeNode) -> IndexedAccessTypeNode;
    createMappedTypeNode(readonlyToken:Keyword | PlusToken | MinusToken, typeParameter: TypeParameterDeclaration, nameType: TypeNode, questionToken: QuestionToken | PlusToken | MinusToken, type: TypeNode) -> MappedTypeNode;
    updateMappedTypeNode(node: MappedTypeNode,Token:Keyword | PlusToken | MinusToken, typeParameter: TypeParameterDeclaration, nameType: TypeNode, questionToken: QuestionToken | PlusToken | MinusToken, type: TypeNode) -> MappedTypeNode;
    createLiteralTypeNode(literal: LiteralTypeNode["literal"]) -> LiteralTypeNode;
    updateLiteralTypeNode(node: LiteralTypeNode, literal: LiteralTypeNode["literal"]) -> LiteralTypeNode;
    createTemplateLiteralType(head: TemplateHead, templateSpans: TemplateLiteralTypeSpan[]) -> TemplateLiteralTypeNode;
    updateTemplateLiteralType(node: TemplateLiteralTypeNode, head: TemplateHead, templateSpans: TemplateLiteralTypeSpan[]) -> TemplateLiteralTypeNode;

    //
    // Binding Patterns
    //

    createObjectBindingPattern(elements: BindingElement[]) -> ObjectBindingPattern;
    updateObjectBindingPattern(node: ObjectBindingPattern, elements: BindingElement[]) -> ObjectBindingPattern;
    createArrayBindingPattern(elements: ArrayBindingElement[]) -> ArrayBindingPattern;
    updateArrayBindingPattern(node: ArrayBindingPattern, elements: ArrayBindingElement[]) -> ArrayBindingPattern;
    createBindingElement(dotDotDotToken: DotDotDotToken, propertyName: string | PropertyName, name: string | BindingName, initializer?: Expression) -> BindingElement;
    updateBindingElement(node: BindingElement, dotDotDotToken: DotDotDotToken, propertyName: PropertyName, name: BindingName, initializer: Expression) -> BindingElement;

    //
    // Expression
    //

    createArrayLiteralExpression(elements?: Expression[], boolean multiLine = false) -> ArrayLiteralExpression;
    updateArrayLiteralExpression(node: ArrayLiteralExpression, elements: Expression[]) -> ArrayLiteralExpression;
    createObjectLiteralExpression(properties?: ObjectLiteralElementLike[], boolean multiLine = false) -> ObjectLiteralExpression;
    updateObjectLiteralExpression(node: ObjectLiteralExpression, properties: ObjectLiteralElementLike[]) -> ObjectLiteralExpression;
    createPropertyAccessExpression(expression: Expression, name: string | MemberName) -> PropertyAccessExpression;
    updatePropertyAccessExpression(node: PropertyAccessExpression, expression: Expression, name: MemberName) -> PropertyAccessExpression;
    createPropertyAccessChain(expression: Expression, questionDotToken: QuestionDotToken, name: string | MemberName) -> PropertyAccessChain;
    updatePropertyAccessChain(node: PropertyAccessChain, expression: Expression, questionDotToken: QuestionDotToken, name: MemberName) -> PropertyAccessChain;
    createElementAccessExpression(expression: Expression, index: number | Expression) -> ElementAccessExpression;
    updateElementAccessExpression(node: ElementAccessExpression, expression: Expression, argumentExpression: Expression) -> ElementAccessExpression;
    createElementAccessChain(expression: Expression, questionDotToken: QuestionDotToken, index: number | Expression) -> ElementAccessChain;
    updateElementAccessChain(node: ElementAccessChain, expression: Expression, questionDotToken: QuestionDotToken, argumentExpression: Expression) -> ElementAccessChain;
    createCallExpression(expression: Expression, typeArguments: TypeNode[], argumentsArray: Expression[]) -> CallExpression;
    updateCallExpression(node: CallExpression, expression: Expression, typeArguments: TypeNode[], argumentsArray: Expression[]) -> CallExpression;
    createCallChain(expression: Expression, questionDotToken: QuestionDotToken, typeArguments: TypeNode[], argumentsArray: Expression[]) -> CallChain;
    updateCallChain(node: CallChain, expression: Expression, questionDotToken: QuestionDotToken, typeArguments: TypeNode[], argumentsArray: Expression[]) -> CallChain;
    createNewExpression(expression: Expression, typeArguments: TypeNode[], argumentsArray: Expression[]) -> NewExpression;
    updateNewExpression(node: NewExpression, expression: Expression, typeArguments: TypeNode[], argumentsArray: Expression[]) -> NewExpression;
    createTaggedTemplateExpression(tag: Expression, typeArguments: TypeNode[], template: TemplateLiteral) -> TaggedTemplateExpression;
    updateTaggedTemplateExpression(node: TaggedTemplateExpression, tag: Expression, typeArguments: TypeNode[], template: TemplateLiteral) -> TaggedTemplateExpression;
    createTypeAssertion(type: TypeNode, expression: Expression) -> TypeAssertion;
    updateTypeAssertion(node: TypeAssertion, type: TypeNode, expression: Expression) -> TypeAssertion;
    createParenthesizedExpression(expression: Expression) -> ParenthesizedExpression;
    updateParenthesizedExpression(node: ParenthesizedExpression, expression: Expression) -> ParenthesizedExpression;
    createFunctionExpression(modifiers: Modifier[], asteriskToken: AsteriskToken, name: string | Identifier, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> FunctionExpression;
    updateFunctionExpression(node: FunctionExpression, modifiers: Modifier[], asteriskToken: AsteriskToken, name: Identifier, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> FunctionExpression;
    createArrowFunction(modifiers: Modifier[], typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, equalsGreaterThanToken: EqualsGreaterThanToken, body: ConciseBody) -> ArrowFunction;
    updateArrowFunction(node: ArrowFunction, modifiers: Modifier[], typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, equalsGreaterThanToken: EqualsGreaterThanToken, body: ConciseBody) -> ArrowFunction;
    createDeleteExpression(expression: Expression) -> DeleteExpression;
    updateDeleteExpression(node: DeleteExpression, expression: Expression) -> DeleteExpression;
    createTypeOfExpression(expression: Expression) -> TypeOfExpression;
    updateTypeOfExpression(node: TypeOfExpression, expression: Expression) -> TypeOfExpression;
    createVoidExpression(expression: Expression) -> VoidExpression;
    updateVoidExpression(node: VoidExpression, expression: Expression) -> VoidExpression;
    createAwaitExpression(expression: Expression) -> AwaitExpression;
    updateAwaitExpression(node: AwaitExpression, expression: Expression) -> AwaitExpression;
    createPrefixUnaryExpression(operator: PrefixUnaryOperator, operand: Expression) -> PrefixUnaryExpression;
    updatePrefixUnaryExpression(node: PrefixUnaryExpression, operand: Expression) -> PrefixUnaryExpression;
    createPostfixUnaryExpression(operand: Expression, operator: PostfixUnaryOperator) -> PostfixUnaryExpression;
    updatePostfixUnaryExpression(node: PostfixUnaryExpression, operand: Expression) -> PostfixUnaryExpression;
    createBinaryExpression(left: Expression, operator: BinaryOperator | BinaryOperatorToken, right: Expression) -> BinaryExpression;
    updateBinaryExpression(node: BinaryExpression, left: Expression, operator: BinaryOperator | BinaryOperatorToken, right: Expression) -> BinaryExpression;
    createConditionalExpression(condition: Expression, questionToken: QuestionToken, whenTrue: Expression, colonToken: ColonToken, whenFalse: Expression) -> ConditionalExpression;
    updateConditionalExpression(node: ConditionalExpression, condition: Expression, questionToken: QuestionToken, whenTrue: Expression, colonToken: ColonToken, whenFalse: Expression) -> ConditionalExpression;
    createTemplateExpression(head: TemplateHead, templateSpans: TemplateSpan[]) -> TemplateExpression;
    updateTemplateExpression(node: TemplateExpression, head: TemplateHead, templateSpans: TemplateSpan[]) -> TemplateExpression;
    createTemplateHead(text: string, rawText?: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateHead;
    createTemplateHead(text: string, rawText: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateHead;
    createTemplateMiddle(text: string, rawText?: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateMiddle;
    createTemplateMiddle(text: string, rawText: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateMiddle;
    createTemplateTail(text: string, rawText?: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateTail;
    createTemplateTail(text: string, rawText: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateTail;
    createNoSubstitutionTemplateLiteral(text: string, rawText?: string) -> NoSubstitutionTemplateLiteral;
    createNoSubstitutionTemplateLiteral(text: string, rawText: string) -> NoSubstitutionTemplateLiteral;
    /* @internal */ createLiteralLikeNode(kind: LiteralToken["kind"] | SyntaxKind.JsxTextAllWhiteSpaces, text: string) -> LiteralToken;
    /* @internal */ createTemplateLiteralLikeNode(kind: TemplateLiteralToken["kind"], text: string, rawText: string, templateFlags: TokenFlags) -> TemplateLiteralLikeNode;
    createYieldExpression(asteriskToken: AsteriskToken, expression: Expression) -> YieldExpression;
    createYieldExpression(asteriskToken: undefined, expression: Expression) -> YieldExpression;
    /* @internal */ createYieldExpression(asteriskToken: AsteriskToken, expression: Expression) -> YieldExpression; // eslint-disable-line @typescript-eslint/unified-signatures
    updateYieldExpression(node: YieldExpression, asteriskToken: AsteriskToken, expression: Expression) -> YieldExpression;
    createSpreadElement(expression: Expression) -> SpreadElement;
    updateSpreadElement(node: SpreadElement, expression: Expression) -> SpreadElement;
    createClassExpression(decorators: Decorator[], modifiers: Modifier[], name: string | Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: ClassElement[]) -> ClassExpression;
    updateClassExpression(node: ClassExpression, decorators: Decorator[], modifiers: Modifier[], name: Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: ClassElement[]) -> ClassExpression;
    createOmittedExpression() -> OmittedExpression;
    createExpressionWithTypeArguments(expression: Expression, typeArguments: TypeNode[]) -> ExpressionWithTypeArguments;
    updateExpressionWithTypeArguments(node: ExpressionWithTypeArguments, expression: Expression, typeArguments: TypeNode[]) -> ExpressionWithTypeArguments;
    createAsExpression(expression: Expression, type: TypeNode) -> AsExpression;
    updateAsExpression(node: AsExpression, expression: Expression, type: TypeNode) -> AsExpression;
    createNonNullExpression(expression: Expression) -> NonNullExpression;
    updateNonNullExpression(node: NonNullExpression, expression: Expression) -> NonNullExpression;
    createNonNullChain(expression: Expression) -> NonNullChain;
    updateNonNullChain(node: NonNullChain, expression: Expression) -> NonNullChain;
    createMetaProperty(keywordToken: MetaProperty["keywordToken"], name: Identifier) -> MetaProperty;
    updateMetaProperty(node: MetaProperty, name: Identifier) -> MetaProperty;

    //
    // Misc
    //

    createTemplateSpan(expression: Expression, literal: TemplateMiddle | TemplateTail) -> TemplateSpan;
    updateTemplateSpan(node: TemplateSpan, expression: Expression, literal: TemplateMiddle | TemplateTail) -> TemplateSpan;
    createSemicolonClassElement() -> SemicolonClassElement;

    //
    // Element
    //

    createBlock(statements: Statement[], boolean multiLine = false) -> Block;
    updateBlock(node: Block, statements: Statement[]) -> Block;
    createVariableStatement(modifiers: Modifier[], declarationList: VariableDeclarationList | VariableDeclaration[]) -> VariableStatement;
    updateVariableStatement(node: VariableStatement, modifiers: Modifier[], declarationList: VariableDeclarationList) -> VariableStatement;
    createEmptyStatement() -> EmptyStatement;
    createExpressionStatement(expression: Expression) -> ExpressionStatement;
    updateExpressionStatement(node: ExpressionStatement, expression: Expression) -> ExpressionStatement;
    createIfStatement(expression: Expression, thenStatement: Statement, elseStatement?: Statement) -> IfStatement;
    updateIfStatement(node: IfStatement, expression: Expression, thenStatement: Statement, elseStatement: Statement) -> IfStatement;
    createDoStatement(statement: Statement, expression: Expression) -> DoStatement;
    updateDoStatement(node: DoStatement, statement: Statement, expression: Expression) -> DoStatement;
    createWhileStatement(expression: Expression, statement: Statement) -> WhileStatement;
    updateWhileStatement(node: WhileStatement, expression: Expression, statement: Statement) -> WhileStatement;
    createForStatement(initializer: ForInitializer, condition: Expression, incrementor: Expression, statement: Statement) -> ForStatement;
    updateForStatement(node: ForStatement, initializer: ForInitializer, condition: Expression, incrementor: Expression, statement: Statement) -> ForStatement;
    createForInStatement(initializer: ForInitializer, expression: Expression, statement: Statement) -> ForInStatement;
    updateForInStatement(node: ForInStatement, initializer: ForInitializer, expression: Expression, statement: Statement) -> ForInStatement;
    createForOfStatement(awaitModifier: AwaitKeyword, initializer: ForInitializer, expression: Expression, statement: Statement) -> ForOfStatement;
    updateForOfStatement(node: ForOfStatement, awaitModifier: AwaitKeyword, initializer: ForInitializer, expression: Expression, statement: Statement) -> ForOfStatement;
    createContinueStatement(label?: string | Identifier) -> ContinueStatement;
    updateContinueStatement(node: ContinueStatement, label: Identifier) -> ContinueStatement;
    createBreakStatement(label?: string | Identifier) -> BreakStatement;
    updateBreakStatement(node: BreakStatement, label: Identifier) -> BreakStatement;
    createReturnStatement(expression?: Expression) -> ReturnStatement;
    updateReturnStatement(node: ReturnStatement, expression: Expression) -> ReturnStatement;
    createWithStatement(expression: Expression, statement: Statement) -> WithStatement;
    updateWithStatement(node: WithStatement, expression: Expression, statement: Statement) -> WithStatement;
    createSwitchStatement(expression: Expression, caseBlock: CaseBlock) -> SwitchStatement;
    updateSwitchStatement(node: SwitchStatement, expression: Expression, caseBlock: CaseBlock) -> SwitchStatement;
    createLabeledStatement(label: string | Identifier, statement: Statement) -> LabeledStatement;
    updateLabeledStatement(node: LabeledStatement, label: Identifier, statement: Statement) -> LabeledStatement;
    createThrowStatement(expression: Expression) -> ThrowStatement;
    updateThrowStatement(node: ThrowStatement, expression: Expression) -> ThrowStatement;
    createTryStatement(tryBlock: Block, catchClause: CatchClause, finallyBlock: Block) -> TryStatement;
    updateTryStatement(node: TryStatement, tryBlock: Block, catchClause: CatchClause, finallyBlock: Block) -> TryStatement;
    createDebuggerStatement() -> DebuggerStatement;
    createVariableDeclaration(name: string | BindingName, exclamationToken?: ExclamationToken, type?: TypeNode, initializer?: Expression) -> VariableDeclaration;
    updateVariableDeclaration(node: VariableDeclaration, name: BindingName, exclamationToken: ExclamationToken, type: TypeNode, initializer: Expression) -> VariableDeclaration;
    createVariableDeclarationList(declarations: VariableDeclaration[], NodeFlags flags = (NodeFlags)0) -> VariableDeclarationList;
    updateVariableDeclarationList(node: VariableDeclarationList, declarations: VariableDeclaration[]) -> VariableDeclarationList;
    createFunctionDeclaration(decorators: Decorator[], modifiers: Modifier[], asteriskToken: AsteriskToken, name: string | Identifier, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> FunctionDeclaration;
    updateFunctionDeclaration(node: FunctionDeclaration, decorators: Decorator[], modifiers: Modifier[], asteriskToken: AsteriskToken, name: Identifier, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> FunctionDeclaration;
    createClassDeclaration(decorators: Decorator[], modifiers: Modifier[], name: string | Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: ClassElement[]) -> ClassDeclaration;
    updateClassDeclaration(node: ClassDeclaration, decorators: Decorator[], modifiers: Modifier[], name: Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: ClassElement[]) -> ClassDeclaration;
    createInterfaceDeclaration(decorators: Decorator[], modifiers: Modifier[], name: string | Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: TypeElement[]) -> InterfaceDeclaration;
    updateInterfaceDeclaration(node: InterfaceDeclaration, decorators: Decorator[], modifiers: Modifier[], name: Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: TypeElement[]) -> InterfaceDeclaration;
    createTypeAliasDeclaration(decorators: Decorator[], modifiers: Modifier[], name: string | Identifier, typeParameters: TypeParameterDeclaration[], type: TypeNode) -> TypeAliasDeclaration;
    updateTypeAliasDeclaration(node: TypeAliasDeclaration, decorators: Decorator[], modifiers: Modifier[], name: Identifier, typeParameters: TypeParameterDeclaration[], type: TypeNode) -> TypeAliasDeclaration;
    createEnumDeclaration(decorators: Decorator[], modifiers: Modifier[], name: string | Identifier, members: EnumMember[]) -> EnumDeclaration;
    updateEnumDeclaration(node: EnumDeclaration, decorators: Decorator[], modifiers: Modifier[], name: Identifier, members: EnumMember[]) -> EnumDeclaration;
    createModuleDeclaration(decorators: Decorator[], modifiers: Modifier[], name: ModuleName, body: ModuleBody, NodeFlags flags = (NodeFlags)0) -> ModuleDeclaration;
    updateModuleDeclaration(node: ModuleDeclaration, decorators: Decorator[], modifiers: Modifier[], name: ModuleName, body: ModuleBody) -> ModuleDeclaration;
    createModuleBlock(statements: Statement[]) -> ModuleBlock;
    updateModuleBlock(node: ModuleBlock, statements: Statement[]) -> ModuleBlock;
    createCaseBlock(clauses: CaseOrDefaultClause[]) -> CaseBlock;
    updateCaseBlock(node: CaseBlock, clauses: CaseOrDefaultClause[]) -> CaseBlock;
    createNamespaceExportDeclaration(name: string | Identifier) -> NamespaceExportDeclaration;
    updateNamespaceExportDeclaration(node: NamespaceExportDeclaration, name: Identifier) -> NamespaceExportDeclaration;
    createImportEqualsDeclaration(decorators: Decorator[], modifiers: Modifier[], isTypeOnly: boolean, name: string | Identifier, moduleReference: ModuleReference) -> ImportEqualsDeclaration;
    updateImportEqualsDeclaration(node: ImportEqualsDeclaration, decorators: Decorator[], modifiers: Modifier[], isTypeOnly: boolean, name: Identifier, moduleReference: ModuleReference) -> ImportEqualsDeclaration;
    createImportDeclaration(decorators: Decorator[], modifiers: Modifier[], importClause: ImportClause, moduleSpecifier: Expression) -> ImportDeclaration;
    updateImportDeclaration(node: ImportDeclaration, decorators: Decorator[], modifiers: Modifier[], importClause: ImportClause, moduleSpecifier: Expression) -> ImportDeclaration;
    createImportClause(isTypeOnly: boolean, name: Identifier, namedBindings: NamedImportBindings) -> ImportClause;
    updateImportClause(node: ImportClause, isTypeOnly: boolean, name: Identifier, namedBindings: NamedImportBindings) -> ImportClause;
    createNamespaceImport(name: Identifier) -> NamespaceImport;
    updateNamespaceImport(node: NamespaceImport, name: Identifier) -> NamespaceImport;
    createNamespaceExport(name: Identifier) -> NamespaceExport;
    updateNamespaceExport(node: NamespaceExport, name: Identifier) -> NamespaceExport;
    createNamedImports(elements: ImportSpecifier[]) -> NamedImports;
    updateNamedImports(node: NamedImports, elements: ImportSpecifier[]) -> NamedImports;
    createImportSpecifier(propertyName: Identifier, name: Identifier) -> ImportSpecifier;
    updateImportSpecifier(node: ImportSpecifier, propertyName: Identifier, name: Identifier) -> ImportSpecifier;
    createExportAssignment(decorators: Decorator[], modifiers: Modifier[], isExportEquals: boolean, expression: Expression) -> ExportAssignment;
    updateExportAssignment(node: ExportAssignment, decorators: Decorator[], modifiers: Modifier[], expression: Expression) -> ExportAssignment;
    createExportDeclaration(decorators: Decorator[], modifiers: Modifier[], isTypeOnly: boolean, exportClause: NamedExportBindings, moduleSpecifier?: Expression) -> ExportDeclaration;
    updateExportDeclaration(node: ExportDeclaration, decorators: Decorator[], modifiers: Modifier[], isTypeOnly: boolean, exportClause: NamedExportBindings, moduleSpecifier: Expression) -> ExportDeclaration;
    createNamedExports(elements: ExportSpecifier[]) -> NamedExports;
    updateNamedExports(node: NamedExports, elements: ExportSpecifier[]) -> NamedExports;
    createExportSpecifier(propertyName: string | Identifier, name: string | Identifier) -> ExportSpecifier;
    updateExportSpecifier(node: ExportSpecifier, propertyName: Identifier, name: Identifier) -> ExportSpecifier;
    /* @internal*/ createMissingDeclaration() -> MissingDeclaration;

    //
    // Module references
    //

    createExternalModuleReference(expression: Expression) -> ExternalModuleReference;
    updateExternalModuleReference(node: ExternalModuleReference, expression: Expression) -> ExternalModuleReference;

    //
    // JSDoc
    //

    createJSDocAllType() -> JSDocAllType;
    createJSDocUnknownType() -> JSDocUnknownType;
    createJSDocNonNullableType(type: TypeNode) -> JSDocNonNullableType;
    updateJSDocNonNullableType(node: JSDocNonNullableType, type: TypeNode) -> JSDocNonNullableType;
    createJSDocNullableType(type: TypeNode) -> JSDocNullableType;
    updateJSDocNullableType(node: JSDocNullableType, type: TypeNode) -> JSDocNullableType;
    createJSDocOptionalType(type: TypeNode) -> JSDocOptionalType;
    updateJSDocOptionalType(node: JSDocOptionalType, type: TypeNode) -> JSDocOptionalType;
    createJSDocFunctionType(parameters: ParameterDeclaration[], type: TypeNode) -> JSDocFunctionType;
    updateJSDocFunctionType(node: JSDocFunctionType, parameters: ParameterDeclaration[], type: TypeNode) -> JSDocFunctionType;
    createJSDocVariadicType(type: TypeNode) -> JSDocVariadicType;
    updateJSDocVariadicType(node: JSDocVariadicType, type: TypeNode) -> JSDocVariadicType;
    createJSDocNamepathType(type: TypeNode) -> JSDocNamepathType;
    updateJSDocNamepathType(node: JSDocNamepathType, type: TypeNode) -> JSDocNamepathType;
    createJSDocTypeExpression(type: TypeNode) -> JSDocTypeExpression;
    updateJSDocTypeExpression(node: JSDocTypeExpression, type: TypeNode) -> JSDocTypeExpression;
    createJSDocNameReference(name: EntityName) -> JSDocNameReference;
    updateJSDocNameReference(node: JSDocNameReference, name: EntityName) -> JSDocNameReference;
    createJSDocTypeLiteral(jsDocPropertyTags?: JSDocPropertyLikeTag[], boolean isArrayType = false) -> JSDocTypeLiteral;
    updateJSDocTypeLiteral(node: JSDocTypeLiteral, jsDocPropertyTags: JSDocPropertyLikeTag[], isArrayType: boolean) -> JSDocTypeLiteral;
    createJSDocSignature(typeParameters: JSDocTemplateTag[], parameters: JSDocParameterTag[], type?: JSDocReturnTag) -> JSDocSignature;
    updateJSDocSignature(node: JSDocSignature, typeParameters: JSDocTemplateTag[], parameters: JSDocParameterTag[], type: JSDocReturnTag) -> JSDocSignature;
    createJSDocTemplateTag(tagName: Identifier, constraint: JSDocTypeExpression, typeParameters: TypeParameterDeclaration[], comment?: string) -> JSDocTemplateTag;
    updateJSDocTemplateTag(node: JSDocTemplateTag, tagName: Identifier, constraint: JSDocTypeExpression, typeParameters: TypeParameterDeclaration[], comment: string) -> JSDocTemplateTag;
    createJSDocTypedefTag(tagName: Identifier, typeExpression?: JSDocTypeExpression | JSDocTypeLiteral, fullName?: Identifier | JSDocNamespaceDeclaration, comment?: string) -> JSDocTypedefTag;
    updateJSDocTypedefTag(node: JSDocTypedefTag, tagName: Identifier, typeExpression: JSDocTypeExpression | JSDocTypeLiteral, fullName: Identifier | JSDocNamespaceDeclaration, comment: string) -> JSDocTypedefTag;
    createJSDocParameterTag(tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression?: JSDocTypeExpression, boolean isNameFirst = false, comment?: string) -> JSDocParameterTag;
    updateJSDocParameterTag(node: JSDocParameterTag, tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression: JSDocTypeExpression, isNameFirst: boolean, comment: string) -> JSDocParameterTag;
    createJSDocPropertyTag(tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression?: JSDocTypeExpression, boolean isNameFirst = false, comment?: string) -> JSDocPropertyTag;
    updateJSDocPropertyTag(node: JSDocPropertyTag, tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression: JSDocTypeExpression, isNameFirst: boolean, comment: string) -> JSDocPropertyTag;
    createJSDocTypeTag(tagName: Identifier, typeExpression: JSDocTypeExpression, comment?: string) -> JSDocTypeTag;
    updateJSDocTypeTag(node: JSDocTypeTag, tagName: Identifier, typeExpression: JSDocTypeExpression, comment: string) -> JSDocTypeTag;
    createJSDocSeeTag(tagName: Identifier, nameExpression: JSDocNameReference, comment?: string) -> JSDocSeeTag;
    updateJSDocSeeTag(node: JSDocSeeTag, tagName: Identifier, nameExpression: JSDocNameReference, comment?: string) -> JSDocSeeTag;
    createJSDocReturnTag(tagName: Identifier, typeExpression?: JSDocTypeExpression, comment?: string) -> JSDocReturnTag;
    updateJSDocReturnTag(node: JSDocReturnTag, tagName: Identifier, typeExpression: JSDocTypeExpression, comment: string) -> JSDocReturnTag;
    createJSDocThisTag(tagName: Identifier, typeExpression: JSDocTypeExpression, comment?: string) -> JSDocThisTag;
    updateJSDocThisTag(node: JSDocThisTag, tagName: Identifier, typeExpression: JSDocTypeExpression, comment: string) -> JSDocThisTag;
    createJSDocEnumTag(tagName: Identifier, typeExpression: JSDocTypeExpression, comment?: string) -> JSDocEnumTag;
    updateJSDocEnumTag(node: JSDocEnumTag, tagName: Identifier, typeExpression: JSDocTypeExpression, comment: string) -> JSDocEnumTag;
    createJSDocCallbackTag(tagName: Identifier, typeExpression: JSDocSignature, fullName?: Identifier | JSDocNamespaceDeclaration, comment?: string) -> JSDocCallbackTag;
    updateJSDocCallbackTag(node: JSDocCallbackTag, tagName: Identifier, typeExpression: JSDocSignature, fullName: Identifier | JSDocNamespaceDeclaration, comment: string) -> JSDocCallbackTag;
    createJSDocAugmentsTag(tagName: Identifier, className: JSDocAugmentsTag["class"], comment?: string) -> JSDocAugmentsTag;
    updateJSDocAugmentsTag(node: JSDocAugmentsTag, tagName: Identifier, className: JSDocAugmentsTag["class"], comment: string) -> JSDocAugmentsTag;
    createJSDocImplementsTag(tagName: Identifier, className: JSDocImplementsTag["class"], comment?: string) -> JSDocImplementsTag;
    updateJSDocImplementsTag(node: JSDocImplementsTag, tagName: Identifier, className: JSDocImplementsTag["class"], comment: string) -> JSDocImplementsTag;
    createJSDocAuthorTag(tagName: Identifier, comment?: string) -> JSDocAuthorTag;
    updateJSDocAuthorTag(node: JSDocAuthorTag, tagName: Identifier, comment: string) -> JSDocAuthorTag;
    createJSDocClassTag(tagName: Identifier, comment?: string) -> JSDocClassTag;
    updateJSDocClassTag(node: JSDocClassTag, tagName: Identifier, comment: string) -> JSDocClassTag;
    createJSDocPublicTag(tagName: Identifier, comment?: string) -> JSDocPublicTag;
    updateJSDocPublicTag(node: JSDocPublicTag, tagName: Identifier, comment: string) -> JSDocPublicTag;
    createJSDocPrivateTag(tagName: Identifier, comment?: string) -> JSDocPrivateTag;
    updateJSDocPrivateTag(node: JSDocPrivateTag, tagName: Identifier, comment: string) -> JSDocPrivateTag;
    createJSDocProtectedTag(tagName: Identifier, comment?: string) -> JSDocProtectedTag;
    updateJSDocProtectedTag(node: JSDocProtectedTag, tagName: Identifier, comment: string) -> JSDocProtectedTag;
    createJSDocReadonlyTag(tagName: Identifier, comment?: string) -> JSDocReadonlyTag;
    updateJSDocReadonlyTag(node: JSDocReadonlyTag, tagName: Identifier, comment: string) -> JSDocReadonlyTag;
    createJSDocUnknownTag(tagName: Identifier, comment?: string) -> JSDocUnknownTag;
    updateJSDocUnknownTag(node: JSDocUnknownTag, tagName: Identifier, comment: string) -> JSDocUnknownTag;
    createJSDocDeprecatedTag(tagName: Identifier, comment?: string) -> JSDocDeprecatedTag;
    updateJSDocDeprecatedTag(node: JSDocDeprecatedTag, tagName: Identifier, comment?: string) -> JSDocDeprecatedTag;
    createJSDocComment(comment?: string, tags?: JSDocTag[]) -> JSDoc;
    updateJSDocComment(node: JSDoc, comment: string, tags: JSDocTag[]) -> JSDoc;

    //
    // JSX
    //

    createJsxElement(openingElement: JsxOpeningElement, children: JsxChild[], closingElement: JsxClosingElement) -> JsxElement;
    updateJsxElement(node: JsxElement, openingElement: JsxOpeningElement, children: JsxChild[], closingElement: JsxClosingElement) -> JsxElement;
    createJsxSelfClosingElement(tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) -> JsxSelfClosingElement;
    updateJsxSelfClosingElement(node: JsxSelfClosingElement, tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) -> JsxSelfClosingElement;
    createJsxOpeningElement(tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) -> JsxOpeningElement;
    updateJsxOpeningElement(node: JsxOpeningElement, tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) -> JsxOpeningElement;
    createJsxClosingElement(tagName: JsxTagNameExpression) -> JsxClosingElement;
    updateJsxClosingElement(node: JsxClosingElement, tagName: JsxTagNameExpression) -> JsxClosingElement;
    createJsxFragment(openingFragment: JsxOpeningFragment, children: JsxChild[], closingFragment: JsxClosingFragment) -> JsxFragment;
    createJsxText(text: string, boolean containsOnlyTriviaWhiteSpaces = false) -> JsxText;
    updateJsxText(node: JsxText, text: string, boolean containsOnlyTriviaWhiteSpaces = false) -> JsxText;
    createJsxOpeningFragment() -> JsxOpeningFragment;
    createJsxJsxClosingFragment() -> JsxClosingFragment;
    updateJsxFragment(node: JsxFragment, openingFragment: JsxOpeningFragment, children: JsxChild[], closingFragment: JsxClosingFragment) -> JsxFragment;
    createJsxAttribute(name: Identifier, initializer: StringLiteral | JsxExpression) -> JsxAttribute;
    updateJsxAttribute(node: JsxAttribute, name: Identifier, initializer: StringLiteral | JsxExpression) -> JsxAttribute;
    createJsxAttributes(properties: JsxAttributeLike[]) -> JsxAttributes;
    updateJsxAttributes(node: JsxAttributes, properties: JsxAttributeLike[]) -> JsxAttributes;
    createJsxSpreadAttribute(expression: Expression) -> JsxSpreadAttribute;
    updateJsxSpreadAttribute(node: JsxSpreadAttribute, expression: Expression) -> JsxSpreadAttribute;
    createJsxExpression(dotDotDotToken: DotDotDotToken, expression: Expression) -> JsxExpression;
    updateJsxExpression(node: JsxExpression, expression: Expression) -> JsxExpression;

    //
    // Clauses
    //

    createCaseClause(expression: Expression, statements: Statement[]) -> CaseClause;
    updateCaseClause(node: CaseClause, expression: Expression, statements: Statement[]) -> CaseClause;
    createDefaultClause(statements: Statement[]) -> DefaultClause;
    updateDefaultClause(node: DefaultClause, statements: Statement[]) -> DefaultClause;
    createHeritageClause(token: HeritageClause["token"], types: ExpressionWithTypeArguments[]) -> HeritageClause;
    updateHeritageClause(node: HeritageClause, types: ExpressionWithTypeArguments[]) -> HeritageClause;
    createCatchClause(variableDeclaration: string | VariableDeclaration, block: Block) -> CatchClause;
    updateCatchClause(node: CatchClause, variableDeclaration: VariableDeclaration, block: Block) -> CatchClause;

    //
    // Property assignments
    //

    createPropertyAssignment(name: string | PropertyName, initializer: Expression) -> PropertyAssignment;
    updatePropertyAssignment(node: PropertyAssignment, name: PropertyName, initializer: Expression) -> PropertyAssignment;
    createShorthandPropertyAssignment(name: string | Identifier, objectAssignmentInitializer?: Expression) -> ShorthandPropertyAssignment;
    updateShorthandPropertyAssignment(node: ShorthandPropertyAssignment, name: Identifier, objectAssignmentInitializer: Expression) -> ShorthandPropertyAssignment;
    createSpreadAssignment(expression: Expression) -> SpreadAssignment;
    updateSpreadAssignment(node: SpreadAssignment, expression: Expression) -> SpreadAssignment;

    //
    // Enum
    //

    createEnumMember(name: string | PropertyName, initializer?: Expression) -> EnumMember;
    updateEnumMember(node: EnumMember, name: PropertyName, initializer: Expression) -> EnumMember;

    //
    // Top-level nodes
    //

    createSourceFile(statements: Statement[], endOfFileToken: EndOfFileToken, flags: NodeFlags) -> SourceFile;
    updateSourceFile(node: SourceFile, statements: Statement[], boolean isDeclarationFile = false, referencedFiles?: FileReference[], typeReferences?: FileReference[], boolean hasNoDefaultLib = false, libReferences?: FileReference[]) -> SourceFile;

    /* @internal */ createUnparsedSource(prologues: UnparsedPrologue[], syntheticReferences: UnparsedSyntheticReference[], texts: UnparsedSourceText[]) -> UnparsedSource;
    /* @internal */ createUnparsedPrologue(data?: string) -> UnparsedPrologue;
    /* @internal */ createUnparsedPrepend(data: string, texts: UnparsedSourceText[]) -> UnparsedPrepend;
    /* @internal */ createUnparsedTextLike(data: string, internal: boolean) -> UnparsedTextLike;
    /* @internal */ createUnparsedSyntheticReference(section: BundleFileHasNoDefaultLib | BundleFileReference) -> UnparsedSyntheticReference;
    /* @internal */ createInputFiles() -> InputFiles;

    //
    // Synthetic Nodes
    //
    /* @internal */ createSyntheticExpression(type: Type, boolean isSpread = false, tupleNameSource?: ParameterDeclaration | NamedTupleMember) -> SyntheticExpression;
    /* @internal */ createSyntaxList(children: Node[]) -> SyntaxList;

    //
    // Transformation nodes
    //

    createNotEmittedStatement(original: Node) -> NotEmittedStatement;
    /* @internal */ createEndOfDeclarationMarker(original: Node) -> EndOfDeclarationMarker;
    /* @internal */ createMergeDeclarationMarker(original: Node) -> MergeDeclarationMarker;
    createPartiallyEmittedExpression(expression: Expression, original?: Node) -> PartiallyEmittedExpression;
    updatePartiallyEmittedExpression(node: PartiallyEmittedExpression, expression: Expression) -> PartiallyEmittedExpression;
    /* @internal */ createSyntheticReferenceExpression(expression: Expression, thisArg: Expression) -> SyntheticReferenceExpression;
    /* @internal */ updateSyntheticReferenceExpression(node: SyntheticReferenceExpression, expression: Expression, thisArg: Expression) -> SyntheticReferenceExpression;
    createCommaListExpression(elements: Expression[]) -> CommaListExpression;
    updateCommaListExpression(node: CommaListExpression, elements: Expression[]) -> CommaListExpression;
    createBundle(sourceFiles: SourceFile[], prepends?: (UnparsedSource | InputFiles)[]) -> Bundle;
    updateBundle(node: Bundle, sourceFiles: SourceFile[], prepends?: (UnparsedSource | InputFiles)[]) -> Bundle;

    //
    // Common operators
    //

    createComma(left: Expression, right: Expression) -> BinaryExpression;
    createAssignment(left: ObjectLiteralExpression | ArrayLiteralExpression, right: Expression) -> DestructuringAssignment;
    createAssignment(left: Expression, right: Expression) -> AssignmentExpression<EqualsToken>;
    createLogicalOr(left: Expression, right: Expression) -> BinaryExpression;
    createLogicalAnd(left: Expression, right: Expression) -> BinaryExpression;
    createBitwiseOr(left: Expression, right: Expression) -> BinaryExpression;
    createBitwiseXor(left: Expression, right: Expression) -> BinaryExpression;
    createBitwiseAnd(left: Expression, right: Expression) -> BinaryExpression;
    createStrictEquality(left: Expression, right: Expression) -> BinaryExpression;
    createStrictInequality(left: Expression, right: Expression) -> BinaryExpression;
    createEquality(left: Expression, right: Expression) -> BinaryExpression;
    createInequality(left: Expression, right: Expression) -> BinaryExpression;
    createLessThan(left: Expression, right: Expression) -> BinaryExpression;
    createLessThanEquals(left: Expression, right: Expression) -> BinaryExpression;
    createGreaterThan(left: Expression, right: Expression) -> BinaryExpression;
    createGreaterThanEquals(left: Expression, right: Expression) -> BinaryExpression;
    createLeftShift(left: Expression, right: Expression) -> BinaryExpression;
    createRightShift(left: Expression, right: Expression) -> BinaryExpression;
    createUnsignedRightShift(left: Expression, right: Expression) -> BinaryExpression;
    createAdd(left: Expression, right: Expression) -> BinaryExpression;
    createSubtract(left: Expression, right: Expression) -> BinaryExpression;
    createMultiply(left: Expression, right: Expression) -> BinaryExpression;
    createDivide(left: Expression, right: Expression) -> BinaryExpression;
    createModulo(left: Expression, right: Expression) -> BinaryExpression;
    createExponent(left: Expression, right: Expression) -> BinaryExpression;
    createPrefixPlus(operand: Expression) -> PrefixUnaryExpression;
    createPrefixMinus(operand: Expression) -> PrefixUnaryExpression;
    createPrefixIncrement(operand: Expression) -> PrefixUnaryExpression;
    createPrefixDecrement(operand: Expression) -> PrefixUnaryExpression;
    createBitwiseNot(operand: Expression) -> PrefixUnaryExpression;
    createLogicalNot(operand: Expression) -> PrefixUnaryExpression;
    createPostfixIncrement(operand: Expression) -> PostfixUnaryExpression;
    createPostfixDecrement(operand: Expression) -> PostfixUnaryExpression;

    //
    // Compound Nodes
    //

    createImmediatelyInvokedFunctionExpression(statements: Statement[]) -> CallExpression;
    createImmediatelyInvokedFunctionExpression(statements: Statement[], param: ParameterDeclaration, paramValue: Expression) -> CallExpression;
    createImmediatelyInvokedArrowFunction(statements: Statement[]) -> CallExpression;
    createImmediatelyInvokedArrowFunction(statements: Statement[], param: ParameterDeclaration, paramValue: Expression) -> CallExpression;


    createVoidZero() -> VoidExpression;
    createExportDefault(expression: Expression) -> ExportAssignment;
    createExternalModuleExport(exportName: Identifier) -> ExportDeclaration;

    /* @internal */ createTypeCheck(value: Expression, tag: TypeOfTag) -> Expression;
    /* @internal */ createMethodCall(object: Expression, methodName: string | Identifier, argumentsList: Expression[]) -> CallExpression;
    /* @internal */ createGlobalMethodCall(globalObjectName: string, globalMethodName: string, argumentsList: Expression[]) -> CallExpression;
    /* @internal */ createFunctionBindCall(target: Expression, thisArg: Expression, argumentsList: Expression[]) -> CallExpression;
    /* @internal */ createFunctionCallCall(target: Expression, thisArg: Expression, argumentsList: Expression[]) -> CallExpression;
    /* @internal */ createFunctionApplyCall(target: Expression, thisArg: Expression, argumentsExpression: Expression) -> CallExpression;
    /* @internal */ createObjectDefinePropertyCall(target: Expression, propertyName: string | Expression, attributes: Expression) -> CallExpression;
    /* @internal */ createPropertyDescriptor(attributes: PropertyDescriptorAttributes, boolean singleLine = false) -> ObjectLiteralExpression;
    /* @internal */ createArraySliceCall(array: Expression, start?: number | Expression) -> CallExpression;
    /* @internal */ createArrayConcatCall(array: Expression, values: Expression[]) -> CallExpression;
    /* @internal */ createCallBinding(expression: Expression, recordTempVariable: (temp: Identifier) => void, languageVersion?: ScriptTarget, boolean cacheIdentifiers = false) -> CallBinding;
    /* @internal */ inlineExpressions(expressions: Expression[]) -> Expression;
    /**
     * Gets the internal name of a declaration. This is primarily used for declarations that can be
     * referred to by name in the body of an ES5 class function body. An internal name will *never*
     * be prefixed with an module or namespace export modifier like "exports." when emitted as an
     * expression. An internal name will also *never* be renamed due to a collision with a block
     * scoped variable.
     *
     * @param node The declaration.
     * @param allowComments A value indicating whether comments may be emitted for the name.
     * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
     */
    /* @internal */ getInternalName(node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    /**
     * Gets the local name of a declaration. This is primarily used for declarations that can be
     * referred to by name in the declaration's immediate scope (classes, enums, namespaces). A
     * local name will *never* be prefixed with an module or namespace export modifier like
     * "exports." when emitted as an expression.
     *
     * @param node The declaration.
     * @param allowComments A value indicating whether comments may be emitted for the name.
     * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
     */
    /* @internal */ getLocalName(node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    /**
     * Gets the export name of a declaration. This is primarily used for declarations that can be
     * referred to by name in the declaration's immediate scope (classes, enums, namespaces). An
     * export name will *always* be prefixed with a module or namespace export modifier like
     * `"exports."` when emitted as an expression if the name points to an exported symbol.
     *
     * @param node The declaration.
     * @param allowComments A value indicating whether comments may be emitted for the name.
     * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
     */
    /* @internal */ getExportName(node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    /**
     * Gets the name of a declaration for use in declarations.
     *
     * @param node The declaration.
     * @param allowComments A value indicating whether comments may be emitted for the name.
     * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
     */
    /* @internal */ getDeclarationName(node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    /**
     * Gets a namespace-qualified name for use in expressions.
     *
     * @param ns The namespace identifier.
     * @param name The name.
     * @param allowComments A value indicating whether comments may be emitted for the name.
     * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
     */
    /* @internal */ getNamespaceMemberName(ns: Identifier, name: Identifier, boolean allowComments = false, boolean allowSourceMaps = false) -> PropertyAccessExpression;
    /**
     * Gets the exported name of a declaration for use in expressions.
     *
     * An exported name will *always* be prefixed with an module or namespace export modifier like
     * "exports." if the name points to an exported symbol.
     *
     * @param ns The namespace identifier.
     * @param node The declaration.
     * @param allowComments A value indicating whether comments may be emitted for the name.
     * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
     */
    /* @internal */ getExternalModuleOrNamespaceExportName(ns: Identifier, node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier | PropertyAccessExpression;

    //
    // Utilities
    //

    restoreOuterExpressions(outerExpression: Expression, innerExpression: Expression, kinds?: OuterExpressionKinds) -> Expression;
    /* @internal */ restoreEnclosingLabel(node: Statement, outermostLabeledStatement: LabeledStatement, afterRestoreLabelCallback?: (node: LabeledStatement) => void) -> Statement;
    /* @internal */ createUseStrictPrologue() -> PrologueDirective;
    /**
     * Copies any necessary standard and custom prologue-directives into target array.
     * @param source origin statements array
     * @param target result statements array
     * @param ensureUseStrict boolean determining whether the function need to add prologue-directives
     * @param visitor Optional callback used to visit any custom prologue directives.
     */
    /* @internal */ copyPrologue(source: Statement[], target: Push<Statement>, boolean ensureUseStrict = false, visitor?: (node: Node) => VisitResult<Node>) -> number;
    /**
     * Copies only the standard (string-expression) prologue-directives into the target statement-array.
     * @param source origin statements array
     * @param target result statements array
     * @param ensureUseStrict boolean determining whether the function need to add prologue-directives
     */
    /* @internal */ copyStandardPrologue(source: Statement[], target: Push<Statement>, boolean ensureUseStrict = false) -> number;
    /**
     * Copies only the custom prologue-directives into target statement-array.
     * @param source origin statements array
     * @param target result statements array
     * @param statementOffset The offset at which to begin the copy.
     * @param visitor Optional callback used to visit any custom prologue directives.
     */
    /* @internal */ copyCustomPrologue(source: Statement[], target: Push<Statement>, statementOffset: number, visitor?: (node: Node) => VisitResult<Node>, filter?: (node: Node) => boolean) -> number;
    /* @internal */ copyCustomPrologue(source: Statement[], target: Push<Statement>, statementOffset: number, visitor?: (node: Node) => VisitResult<Node>, filter?: (node: Node) => boolean) -> number;
    /* @internal */ ensureUseStrict(statements: NodeArray<Statement>) -> NodeArray<Statement>;
    /* @internal */ liftToBlock(nodes: Node[]) -> Statement;
    /**
     * Merges generated lexical declarations into a new statement list.
     */
    /* @internal */ mergeLexicalEnvironment(statements: NodeArray<Statement>, declarations: Statement[]) -> NodeArray<Statement>;
    /**
     * Appends generated lexical declarations to an array of statements.
     */
    /* @internal */ mergeLexicalEnvironment(statements: Statement[], declarations: Statement[]) -> Statement[];
    /**
     * Creates a shallow, memberwise clone of a node.
     * - The result will have its `original` pointer set to `node`.
     * - The result will have its `pos` and `end` set to `-1`.
     * - *DO NOT USE THIS* if a more appropriate function is available.
     */
    /* @internal */ template <typename T/*extends Node*/> cloneNode(node: T) -> T;
    /* @internal */ template <typename T/*extends HasModifiers*/> updateModifiers(node: T, modifiers: Modifier[] | ModifierFlags) -> T;
};

#endif // NODEFACTORY_H