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
    NodeFactory(NodeFactoryFlags nodeFactoryFlags, BaseNodeFactory baseNodeFactory) : auto nodeFactoryFlags(nodeFactoryFlags), auto baseNodeFactory(baseNodeFactory) {}

    /* @internal */ parenthesizer: ParenthesizerRules;
    /* @internal */ converters: NodeConverters;
    template <typename T> auto createNodeArray(NodeArray<T> elements = undefined, boolean hasTrailingComma = false) -> NodeArray<T>;

    //
    // Literals
    //

    auto createNumericLiteral(value: string | number, TokenFlags numericLiteralFlags = (TokenFlags)0) -> NumericLiteral;
    auto createBigIntLiteral(value: string | PseudoBigInt) -> BigIntLiteral;
    auto createStringLiteral(text: string, boolean isSingleQuote = false) -> StringLiteral;
    /* @internal*/ auto createStringLiteral(text: string, boolean isSingleQuote = false, boolean hasExtendedUnicodeEscape = false) -> StringLiteral; // eslint-disable-line @typescript-eslint/unified-signatures
    auto createStringLiteralFromNode(sourceNode: PropertyNameLiteral, boolean isSingleQuote = false) -> StringLiteral;
    auto createRegularExpressionLiteral(text: string) -> RegularExpressionLiteral;

    //
    // Identifiers
    //

    auto createIdentifier(text: string) -> Identifier;
    /* @internal */ auto createIdentifier(text: string, NodeArray</*TypeNode | TypeParameterDeclaration*/Node> typeArguments = undefined, SyntaxKind originalKeywordKind = SyntaxKind::Unknown) -> Identifier; // eslint-disable-line @typescript-eslint/unified-signatures
    /* @internal */ auto updateIdentifier(node: Identifier, typeArguments: NodeArray</*TypeNode | TypeParameterDeclaration*/Node>) -> Identifier;

    /**
     * Create a unique temporary variable.
     * @param recordTempVariable An optional callback used to record the temporary variable name. This
     * should usually be a reference to `hoistVariableDeclaration` from a `TransformationContext`, but
     * can be `undefined` if you plan to record the temporary variable manually.
     * @param reservedInNestedScopes When `true`, reserves the temporary variable name in all nested scopes
     * during emit so that the variable can be referenced in a nested function body. This is an alternative to
     * setting `EmitFlags.ReuseTempVariableScope` on the nested function itself.
     */
    auto createTempVariable(recordTempVariable: ((node: Identifier) => void), boolean reservedInNestedScopes = false) -> Identifier;

    /**
     * Create a unique temporary variable for use in a loop.
     * @param reservedInNestedScopes When `true`, reserves the temporary variable name in all nested scopes
     * during emit so that the variable can be referenced in a nested function body. This is an alternative to
     * setting `EmitFlags.ReuseTempVariableScope` on the nested function itself.
     */
    auto createLoopVariable(boolean reservedInNestedScopes = false) -> Identifier;

    /** Create a unique name based on the supplied text. */
    auto createUniqueName(text: string, GeneratedIdentifierFlags flags = (GeneratedIdentifierFlags)0) -> Identifier;

    /** Create a unique name generated for a node. */
    auto getGeneratedNameForNode(node: Node, GeneratedIdentifierFlags flags = (GeneratedIdentifierFlags)0) -> Identifier;

    auto createPrivateIdentifier(text: string) -> PrivateIdentifier

    //
    // Punctuation
    //

    /*@internal*/ createToken(SyntaxKind token) -> Node;

    //
    // Reserved words
    //

    auto createSuper() -> SuperExpression;
    auto createThis() -> ThisExpression;
    auto createNull() -> NullLiteral;
    auto createTrue() -> TrueLiteral;
    auto createFalse() -> FalseLiteral;

    //
    // Modifiers
    //

    template <typename T/*extends ModifierSyntaxKind*/> auto createModifier(kind: T) -> ModifierToken<T>;
    auto createModifiersFromModifierFlags(flags: ModifierFlags) -> ModifiersArray;

    //
    // Names
    //

    auto createQualifiedName(left: EntityName, right: string | Identifier) -> QualifiedName;
    auto updateQualifiedName(node: QualifiedName, left: EntityName, right: Identifier) -> QualifiedName;
    auto createComputedPropertyName(expression: Expression) -> ComputedPropertyName;
    auto updateComputedPropertyName(node: ComputedPropertyName, expression: Expression) -> ComputedPropertyName;

    //
    // Signature elements
    //

    auto createTypeParameterDeclaration(name: string | Identifier, constraint?: TypeNode, defaultType?: TypeNode) -> TypeParameterDeclaration;
    auto updateTypeParameterDeclaration(node: TypeParameterDeclaration, name: Identifier, constraint: TypeNode, defaultType: TypeNode) -> TypeParameterDeclaration;
    auto createParameterDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, dotDotDotToken: DotDotDotToken, name: string | BindingName, questionToken?: QuestionToken, type?: TypeNode, initializer?: Expression) -> ParameterDeclaration;
    auto updateParameterDeclaration(node: ParameterDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, dotDotDotToken: DotDotDotToken, name: string | BindingName, questionToken: QuestionToken, type: TypeNode, initializer: Expression) -> ParameterDeclaration;
    auto createDecorator(expression: Expression) -> Decorator;
    auto updateDecorator(node: Decorator, expression: Expression) -> Decorator;

    //
    // Type Elements
    //

    auto createPropertySignature(modifiers: ModifiersArray, name: PropertyName | string, questionToken: QuestionToken, type: TypeNode) -> PropertySignature;
    auto updatePropertySignature(node: PropertySignature, modifiers: ModifiersArray, name: PropertyName, questionToken: QuestionToken, type: TypeNode) -> PropertySignature;
    auto createPropertyDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | PropertyName, questionOrExclamationToken: QuestionToken | ExclamationToken, type: TypeNode, initializer: Expression) -> PropertyDeclaration;
    auto updatePropertyDeclaration(node: PropertyDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | PropertyName, questionOrExclamationToken: QuestionToken | ExclamationToken, type: TypeNode, initializer: Expression) -> PropertyDeclaration;
    auto createMethodSignature(modifiers: ModifiersArray, name: string | PropertyName, questionToken: QuestionToken, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> MethodSignature;
    auto updateMethodSignature(node: MethodSignature, modifiers: ModifiersArray, name: PropertyName, questionToken: QuestionToken, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> MethodSignature;
    auto createMethodDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, asteriskToken: AsteriskToken, name: string | PropertyName, questionToken: QuestionToken, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> MethodDeclaration;
    auto updateMethodDeclaration(node: MethodDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, asteriskToken: AsteriskToken, name: PropertyName, questionToken: QuestionToken, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> MethodDeclaration;
    auto createConstructorDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, parameters: ParameterDeclaration[], body: Block) -> ConstructorDeclaration;
    auto updateConstructorDeclaration(node: ConstructorDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, parameters: ParameterDeclaration[], body: Block) -> ConstructorDeclaration;
    auto createGetAccessorDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | PropertyName, parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> GetAccessorDeclaration;
    auto updateGetAccessorDeclaration(node: GetAccessorDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, name: PropertyName, parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> GetAccessorDeclaration;
    auto createSetAccessorDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | PropertyName, parameters: ParameterDeclaration[], body: Block) -> SetAccessorDeclaration;
    auto updateSetAccessorDeclaration(node: SetAccessorDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, name: PropertyName, parameters: ParameterDeclaration[], body: Block) -> SetAccessorDeclaration;
    auto createCallSignature(typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> CallSignatureDeclaration;
    auto updateCallSignature(node: CallSignatureDeclaration, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> CallSignatureDeclaration;
    auto createConstructSignature(typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> ConstructSignatureDeclaration;
    auto updateConstructSignature(node: ConstructSignatureDeclaration, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> ConstructSignatureDeclaration;
    auto createIndexSignature(decorators: DecoratorsArray, modifiers: ModifiersArray, parameters: ParameterDeclaration[], type: TypeNode) -> IndexSignatureDeclaration;
    /* @internal */ auto createIndexSignature(decorators: DecoratorsArray, modifiers: ModifiersArray, parameters: ParameterDeclaration[], type: TypeNode) -> IndexSignatureDeclaration; // eslint-disable-line @typescript-eslint/unified-signatures
    auto updateIndexSignature(node: IndexSignatureDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, parameters: ParameterDeclaration[], type: TypeNode) -> IndexSignatureDeclaration;
    auto createTemplateLiteralTypeSpan(type: TypeNode, literal: TemplateMiddle | TemplateTail) -> TemplateLiteralTypeSpan;
    auto updateTemplateLiteralTypeSpan(node: TemplateLiteralTypeSpan, type: TypeNode, literal: TemplateMiddle | TemplateTail) -> TemplateLiteralTypeSpan;

    //
    // Types
    //

    auto createKeywordTypeNode(SyntaxKind kind) -> Node;
    auto createTypePredicateNode(assertsModifier: AssertsKeyword, parameterName: Identifier | ThisTypeNode | string, type: TypeNode) -> TypePredicateNode;
    auto updateTypePredicateNode(node: TypePredicateNode, assertsModifier: AssertsKeyword, parameterName: Identifier | ThisTypeNode, type: TypeNode) -> TypePredicateNode;
    auto createTypeReferenceNode(typeName: string | EntityName, NodeArray<TypeNode> typeArguments = undefined) -> TypeReferenceNode;
    auto updateTypeReferenceNode(node: TypeReferenceNode, typeName: EntityName, typeArguments: NodeArray<TypeNode>) -> TypeReferenceNode;
    auto createFunctionTypeNode(typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> FunctionTypeNode;
    auto updateFunctionTypeNode(node: FunctionTypeNode, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> FunctionTypeNode;
    auto createConstructorTypeNode(modifiers: ModifiersArray, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> ConstructorTypeNode;
    /** @deprecated */
    auto createConstructorTypeNode(typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode) -> ConstructorTypeNode;
    auto updateConstructorTypeNode(node: ConstructorTypeNode, modifiers: ModifiersArray, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> ConstructorTypeNode;
    /** @deprecated */
    auto updateConstructorTypeNode(node: ConstructorTypeNode, typeParameters: NodeArray<TypeParameterDeclaration>, parameters: NodeArray<ParameterDeclaration>, type: TypeNode) -> ConstructorTypeNode;
    auto createTypeQueryNode(exprName: EntityName) -> TypeQueryNode;
    auto updateTypeQueryNode(node: TypeQueryNode, exprName: EntityName) -> TypeQueryNode;
    auto createTypeLiteralNode(members: TypeElement[]) -> TypeLiteralNode;
    auto updateTypeLiteralNode(node: TypeLiteralNode, members: NodeArray<TypeElement>) -> TypeLiteralNode;
    auto createArrayTypeNode(elementType: TypeNode) -> ArrayTypeNode;
    auto updateArrayTypeNode(node: ArrayTypeNode, elementType: TypeNode) -> ArrayTypeNode;
    auto createTupleTypeNode(elements: /*TypeNode | NamedTupleMember*/Node[]) -> TupleTypeNode;
    auto updateTupleTypeNode(node: TupleTypeNode, elements: /*TypeNode | NamedTupleMember*/Node[]) -> TupleTypeNode;
    auto createNamedTupleMember(dotDotDotToken: DotDotDotToken, name: Identifier, questionToken: QuestionToken, type: TypeNode) -> NamedTupleMember;
    auto updateNamedTupleMember(node: NamedTupleMember, dotDotDotToken: DotDotDotToken, name: Identifier, questionToken: QuestionToken, type: TypeNode) -> NamedTupleMember;
    auto createOptionalTypeNode(type: TypeNode) -> OptionalTypeNode;
    auto updateOptionalTypeNode(node: OptionalTypeNode, type: TypeNode) -> OptionalTypeNode;
    auto createRestTypeNode(type: TypeNode) -> RestTypeNode;
    auto updateRestTypeNode(node: RestTypeNode, type: TypeNode) -> RestTypeNode;
    auto createUnionTypeNode(types: TypeNode[]) -> UnionTypeNode;
    auto updateUnionTypeNode(node: UnionTypeNode, types: NodeArray<TypeNode>) -> UnionTypeNode;
    auto createIntersectionTypeNode(types: TypeNode[]) -> IntersectionTypeNode;
    auto updateIntersectionTypeNode(node: IntersectionTypeNode, types: NodeArray<TypeNode>) -> IntersectionTypeNode;
    auto createConditionalTypeNode(checkType: TypeNode, extendsType: TypeNode, trueType: TypeNode, falseType: TypeNode) -> ConditionalTypeNode;
    auto updateConditionalTypeNode(node: ConditionalTypeNode, checkType: TypeNode, extendsType: TypeNode, trueType: TypeNode, falseType: TypeNode) -> ConditionalTypeNode;
    auto createInferTypeNode(typeParameter: TypeParameterDeclaration) -> InferTypeNode;
    auto updateInferTypeNode(node: InferTypeNode, typeParameter: TypeParameterDeclaration) -> InferTypeNode;
    auto createImportTypeNode(argument: TypeNode, qualifier?: EntityName, NodeArray<TypeNode> typeArguments = undefined, boolean isTypeOf = false) -> ImportTypeNode;
    auto updateImportTypeNode(node: ImportTypeNode, argument: TypeNode, qualifier: EntityName, typeArguments: TypeNode[], boolean isTypeOf = false) -> ImportTypeNode;
    auto createParenthesizedType(type: TypeNode) -> ParenthesizedTypeNode;
    auto updateParenthesizedType(node: ParenthesizedTypeNode, type: TypeNode) -> ParenthesizedTypeNode;
    auto createThisTypeNode() -> ThisTypeNode;
    auto createTypeOperatorNode(operator: SyntaxKind.KeyOfKeyword | SyntaxKind.UniqueKeyword | SyntaxKind.ReadonlyKeyword, type: TypeNode) -> TypeOperatorNode;
    auto updateTypeOperatorNode(node: TypeOperatorNode, type: TypeNode) -> TypeOperatorNode;
    auto createIndexedAccessTypeNode(objectType: TypeNode, indexType: TypeNode) -> IndexedAccessTypeNode;
    auto updateIndexedAccessTypeNode(node: IndexedAccessTypeNode, objectType: TypeNode, indexType: TypeNode) -> IndexedAccessTypeNode;
    auto createMappedTypeNode(readonlyToken:Keyword | PlusToken | MinusToken, typeParameter: TypeParameterDeclaration, nameType: TypeNode, questionToken: QuestionToken | PlusToken | MinusToken, type: TypeNode) -> MappedTypeNode;
    auto updateMappedTypeNode(node: MappedTypeNode,Token:Keyword | PlusToken | MinusToken, typeParameter: TypeParameterDeclaration, nameType: TypeNode, questionToken: QuestionToken | PlusToken | MinusToken, type: TypeNode) -> MappedTypeNode;
    auto createLiteralTypeNode(literal: LiteralTypeNode["literal"]) -> LiteralTypeNode;
    auto updateLiteralTypeNode(node: LiteralTypeNode, literal: LiteralTypeNode["literal"]) -> LiteralTypeNode;
    auto createTemplateLiteralType(head: TemplateHead, templateSpans: TemplateLiteralTypeSpan[]) -> TemplateLiteralTypeNode;
    auto updateTemplateLiteralType(node: TemplateLiteralTypeNode, head: TemplateHead, templateSpans: TemplateLiteralTypeSpan[]) -> TemplateLiteralTypeNode;

    //
    // Binding Patterns
    //

    auto createObjectBindingPattern(elements: BindingElement[]) -> ObjectBindingPattern;
    auto updateObjectBindingPattern(node: ObjectBindingPattern, elements: BindingElement[]) -> ObjectBindingPattern;
    auto createArrayBindingPattern(elements: ArrayBindingElement[]) -> ArrayBindingPattern;
    auto updateArrayBindingPattern(node: ArrayBindingPattern, elements: ArrayBindingElement[]) -> ArrayBindingPattern;
    auto createBindingElement(dotDotDotToken: DotDotDotToken, propertyName: string | PropertyName, name: string | BindingName, initializer?: Expression) -> BindingElement;
    auto updateBindingElement(node: BindingElement, dotDotDotToken: DotDotDotToken, propertyName: PropertyName, name: BindingName, initializer: Expression) -> BindingElement;

    //
    // Expression
    //

    auto createArrayLiteralExpression(NodeArray<Expression> elements = undefined, boolean multiLine = false) -> ArrayLiteralExpression;
    auto updateArrayLiteralExpression(node: ArrayLiteralExpression, elements: Expression[]) -> ArrayLiteralExpression;
    auto createObjectLiteralExpression(NodeArray<ObjectLiteralElementLike> properties = undefined, boolean multiLine = false) -> ObjectLiteralExpression;
    auto updateObjectLiteralExpression(node: ObjectLiteralExpression, properties: ObjectLiteralElementLike[]) -> ObjectLiteralExpression;
    auto createPropertyAccessExpression(expression: Expression, name: string | MemberName) -> PropertyAccessExpression;
    auto updatePropertyAccessExpression(node: PropertyAccessExpression, expression: Expression, name: MemberName) -> PropertyAccessExpression;
    auto createPropertyAccessChain(expression: Expression, questionDotToken: QuestionDotToken, name: string | MemberName) -> PropertyAccessChain;
    auto updatePropertyAccessChain(node: PropertyAccessChain, expression: Expression, questionDotToken: QuestionDotToken, name: MemberName) -> PropertyAccessChain;
    auto createElementAccessExpression(expression: Expression, index: number | Expression) -> ElementAccessExpression;
    auto updateElementAccessExpression(node: ElementAccessExpression, expression: Expression, argumentExpression: Expression) -> ElementAccessExpression;
    auto createElementAccessChain(expression: Expression, questionDotToken: QuestionDotToken, index: number | Expression) -> ElementAccessChain;
    auto updateElementAccessChain(node: ElementAccessChain, expression: Expression, questionDotToken: QuestionDotToken, argumentExpression: Expression) -> ElementAccessChain;
    auto createCallExpression(expression: Expression, typeArguments: TypeNode[], argumentsArray: Expression[]) -> CallExpression;
    auto updateCallExpression(node: CallExpression, expression: Expression, typeArguments: TypeNode[], argumentsArray: Expression[]) -> CallExpression;
    auto createCallChain(expression: Expression, questionDotToken: QuestionDotToken, typeArguments: TypeNode[], argumentsArray: Expression[]) -> CallChain;
    auto updateCallChain(node: CallChain, expression: Expression, questionDotToken: QuestionDotToken, typeArguments: TypeNode[], argumentsArray: Expression[]) -> CallChain;
    auto createNewExpression(expression: Expression, typeArguments: TypeNode[], argumentsArray: Expression[]) -> NewExpression;
    auto updateNewExpression(node: NewExpression, expression: Expression, typeArguments: TypeNode[], argumentsArray: Expression[]) -> NewExpression;
    auto createTaggedTemplateExpression(tag: Expression, typeArguments: TypeNode[], template: TemplateLiteral) -> TaggedTemplateExpression;
    auto updateTaggedTemplateExpression(node: TaggedTemplateExpression, tag: Expression, typeArguments: TypeNode[], template: TemplateLiteral) -> TaggedTemplateExpression;
    auto createTypeAssertion(type: TypeNode, expression: Expression) -> TypeAssertion;
    auto updateTypeAssertion(node: TypeAssertion, type: TypeNode, expression: Expression) -> TypeAssertion;
    auto createParenthesizedExpression(expression: Expression) -> ParenthesizedExpression;
    auto updateParenthesizedExpression(node: ParenthesizedExpression, expression: Expression) -> ParenthesizedExpression;
    auto createFunctionExpression(modifiers: ModifiersArray, asteriskToken: AsteriskToken, name: string | Identifier, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> FunctionExpression;
    auto updateFunctionExpression(node: FunctionExpression, modifiers: ModifiersArray, asteriskToken: AsteriskToken, name: Identifier, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> FunctionExpression;
    auto createArrowFunction(modifiers: ModifiersArray, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, equalsGreaterThanToken: EqualsGreaterThanToken, body: ConciseBody) -> ArrowFunction;
    auto updateArrowFunction(node: ArrowFunction, modifiers: ModifiersArray, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, equalsGreaterThanToken: EqualsGreaterThanToken, body: ConciseBody) -> ArrowFunction;
    auto createDeleteExpression(expression: Expression) -> DeleteExpression;
    auto updateDeleteExpression(node: DeleteExpression, expression: Expression) -> DeleteExpression;
    auto createTypeOfExpression(expression: Expression) -> TypeOfExpression;
    auto updateTypeOfExpression(node: TypeOfExpression, expression: Expression) -> TypeOfExpression;
    auto createVoidExpression(expression: Expression) -> VoidExpression;
    auto updateVoidExpression(node: VoidExpression, expression: Expression) -> VoidExpression;
    auto createAwaitExpression(expression: Expression) -> AwaitExpression;
    auto updateAwaitExpression(node: AwaitExpression, expression: Expression) -> AwaitExpression;
    auto createPrefixUnaryExpression(operator: PrefixUnaryOperator, operand: Expression) -> PrefixUnaryExpression;
    auto updatePrefixUnaryExpression(node: PrefixUnaryExpression, operand: Expression) -> PrefixUnaryExpression;
    auto createPostfixUnaryExpression(operand: Expression, operator: PostfixUnaryOperator) -> PostfixUnaryExpression;
    auto updatePostfixUnaryExpression(node: PostfixUnaryExpression, operand: Expression) -> PostfixUnaryExpression;
    auto createBinaryExpression(left: Expression, operator: BinaryOperator | BinaryOperatorToken, right: Expression) -> BinaryExpression;
    auto updateBinaryExpression(node: BinaryExpression, left: Expression, operator: BinaryOperator | BinaryOperatorToken, right: Expression) -> BinaryExpression;
    auto createConditionalExpression(condition: Expression, questionToken: QuestionToken, whenTrue: Expression, colonToken: ColonToken, whenFalse: Expression) -> ConditionalExpression;
    auto updateConditionalExpression(node: ConditionalExpression, condition: Expression, questionToken: QuestionToken, whenTrue: Expression, colonToken: ColonToken, whenFalse: Expression) -> ConditionalExpression;
    auto createTemplateExpression(head: TemplateHead, templateSpans: TemplateSpan[]) -> TemplateExpression;
    auto updateTemplateExpression(node: TemplateExpression, head: TemplateHead, templateSpans: TemplateSpan[]) -> TemplateExpression;
    auto createTemplateHead(text: string, rawText?: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateHead;
    auto createTemplateHead(text: string, rawText: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateHead;
    auto createTemplateMiddle(text: string, rawText?: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateMiddle;
    auto createTemplateMiddle(text: string, rawText: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateMiddle;
    auto createTemplateTail(text: string, rawText?: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateTail;
    auto createTemplateTail(text: string, rawText: string, TokenFlags templateFlags = (TokenFlags)0) -> TemplateTail;
    auto createNoSubstitutionTemplateLiteral(text: string, rawText?: string) -> NoSubstitutionTemplateLiteral;
    auto createNoSubstitutionTemplateLiteral(text: string, rawText: string) -> NoSubstitutionTemplateLiteral;
    /* @internal */ auto createLiteralLikeNode(kind: LiteralToken["kind"] | SyntaxKind.JsxTextAllWhiteSpaces, text: string) -> LiteralToken;
    /* @internal */ auto createTemplateLiteralLikeNode(kind: TemplateLiteralToken["kind"], text: string, rawText: string, templateFlags: TokenFlags) -> TemplateLiteralLikeNode;
    auto createYieldExpression(asteriskToken: AsteriskToken, expression: Expression) -> YieldExpression;
    auto createYieldExpression(asteriskToken: undefined, expression: Expression) -> YieldExpression;
    /* @internal */ auto createYieldExpression(asteriskToken: AsteriskToken, expression: Expression) -> YieldExpression; // eslint-disable-line @typescript-eslint/unified-signatures
    auto updateYieldExpression(node: YieldExpression, asteriskToken: AsteriskToken, expression: Expression) -> YieldExpression;
    auto createSpreadElement(expression: Expression) -> SpreadElement;
    auto updateSpreadElement(node: SpreadElement, expression: Expression) -> SpreadElement;
    auto createClassExpression(decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: ClassElement[]) -> ClassExpression;
    auto updateClassExpression(node: ClassExpression, decorators: DecoratorsArray, modifiers: ModifiersArray, name: Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: ClassElement[]) -> ClassExpression;
    auto createOmittedExpression() -> OmittedExpression;
    auto createExpressionWithTypeArguments(expression: Expression, typeArguments: TypeNode[]) -> ExpressionWithTypeArguments;
    auto updateExpressionWithTypeArguments(node: ExpressionWithTypeArguments, expression: Expression, typeArguments: TypeNode[]) -> ExpressionWithTypeArguments;
    auto createAsExpression(expression: Expression, type: TypeNode) -> AsExpression;
    auto updateAsExpression(node: AsExpression, expression: Expression, type: TypeNode) -> AsExpression;
    auto createNonNullExpression(expression: Expression) -> NonNullExpression;
    auto updateNonNullExpression(node: NonNullExpression, expression: Expression) -> NonNullExpression;
    auto createNonNullChain(expression: Expression) -> NonNullChain;
    auto updateNonNullChain(node: NonNullChain, expression: Expression) -> NonNullChain;
    auto createMetaProperty(keywordToken: MetaProperty["keywordToken"], name: Identifier) -> MetaProperty;
    auto updateMetaProperty(node: MetaProperty, name: Identifier) -> MetaProperty;

    //
    // Misc
    //

    auto createTemplateSpan(expression: Expression, literal: TemplateMiddle | TemplateTail) -> TemplateSpan;
    auto updateTemplateSpan(node: TemplateSpan, expression: Expression, literal: TemplateMiddle | TemplateTail) -> TemplateSpan;
    auto createSemicolonClassElement() -> SemicolonClassElement;

    //
    // Element
    //

    auto createBlock(statements: Statement[], boolean multiLine = false) -> Block;
    auto updateBlock(node: Block, statements: Statement[]) -> Block;
    auto createVariableStatement(modifiers: ModifiersArray, declarationList: VariableDeclarationList | VariableDeclaration[]) -> VariableStatement;
    auto updateVariableStatement(node: VariableStatement, modifiers: ModifiersArray, declarationList: VariableDeclarationList) -> VariableStatement;
    auto createEmptyStatement() -> EmptyStatement;
    auto createExpressionStatement(expression: Expression) -> ExpressionStatement;
    auto updateExpressionStatement(node: ExpressionStatement, expression: Expression) -> ExpressionStatement;
    auto createIfStatement(expression: Expression, thenStatement: Statement, elseStatement?: Statement) -> IfStatement;
    auto updateIfStatement(node: IfStatement, expression: Expression, thenStatement: Statement, elseStatement: Statement) -> IfStatement;
    auto createDoStatement(statement: Statement, expression: Expression) -> DoStatement;
    auto updateDoStatement(node: DoStatement, statement: Statement, expression: Expression) -> DoStatement;
    auto createWhileStatement(expression: Expression, statement: Statement) -> WhileStatement;
    auto updateWhileStatement(node: WhileStatement, expression: Expression, statement: Statement) -> WhileStatement;
    auto createForStatement(initializer: ForInitializer, condition: Expression, incrementor: Expression, statement: Statement) -> ForStatement;
    auto updateForStatement(node: ForStatement, initializer: ForInitializer, condition: Expression, incrementor: Expression, statement: Statement) -> ForStatement;
    auto createForInStatement(initializer: ForInitializer, expression: Expression, statement: Statement) -> ForInStatement;
    auto updateForInStatement(node: ForInStatement, initializer: ForInitializer, expression: Expression, statement: Statement) -> ForInStatement;
    auto createForOfStatement(awaitModifier: AwaitKeyword, initializer: ForInitializer, expression: Expression, statement: Statement) -> ForOfStatement;
    auto updateForOfStatement(node: ForOfStatement, awaitModifier: AwaitKeyword, initializer: ForInitializer, expression: Expression, statement: Statement) -> ForOfStatement;
    auto createContinueStatement(label?: string | Identifier) -> ContinueStatement;
    auto updateContinueStatement(node: ContinueStatement, label: Identifier) -> ContinueStatement;
    auto createBreakStatement(label?: string | Identifier) -> BreakStatement;
    auto updateBreakStatement(node: BreakStatement, label: Identifier) -> BreakStatement;
    auto createReturnStatement(expression?: Expression) -> ReturnStatement;
    auto updateReturnStatement(node: ReturnStatement, expression: Expression) -> ReturnStatement;
    auto createWithStatement(expression: Expression, statement: Statement) -> WithStatement;
    auto updateWithStatement(node: WithStatement, expression: Expression, statement: Statement) -> WithStatement;
    auto createSwitchStatement(expression: Expression, caseBlock: CaseBlock) -> SwitchStatement;
    auto updateSwitchStatement(node: SwitchStatement, expression: Expression, caseBlock: CaseBlock) -> SwitchStatement;
    auto createLabeledStatement(label: string | Identifier, statement: Statement) -> LabeledStatement;
    auto updateLabeledStatement(node: LabeledStatement, label: Identifier, statement: Statement) -> LabeledStatement;
    auto createThrowStatement(expression: Expression) -> ThrowStatement;
    auto updateThrowStatement(node: ThrowStatement, expression: Expression) -> ThrowStatement;
    auto createTryStatement(tryBlock: Block, catchClause: CatchClause, finallyBlock: Block) -> TryStatement;
    auto updateTryStatement(node: TryStatement, tryBlock: Block, catchClause: CatchClause, finallyBlock: Block) -> TryStatement;
    auto createDebuggerStatement() -> DebuggerStatement;
    auto createVariableDeclaration(name: string | BindingName, exclamationToken?: ExclamationToken, type?: TypeNode, initializer?: Expression) -> VariableDeclaration;
    auto updateVariableDeclaration(node: VariableDeclaration, name: BindingName, exclamationToken: ExclamationToken, type: TypeNode, initializer: Expression) -> VariableDeclaration;
    auto createVariableDeclarationList(declarations: VariableDeclaration[], NodeFlags flags = (NodeFlags)0) -> VariableDeclarationList;
    auto updateVariableDeclarationList(node: VariableDeclarationList, declarations: VariableDeclaration[]) -> VariableDeclarationList;
    auto createFunctionDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, asteriskToken: AsteriskToken, name: string | Identifier, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> FunctionDeclaration;
    auto updateFunctionDeclaration(node: FunctionDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, asteriskToken: AsteriskToken, name: Identifier, typeParameters: TypeParameterDeclaration[], parameters: ParameterDeclaration[], type: TypeNode, body: Block) -> FunctionDeclaration;
    auto createClassDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: ClassElement[]) -> ClassDeclaration;
    auto updateClassDeclaration(node: ClassDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, name: Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: ClassElement[]) -> ClassDeclaration;
    auto createInterfaceDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: TypeElement[]) -> InterfaceDeclaration;
    auto updateInterfaceDeclaration(node: InterfaceDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, name: Identifier, typeParameters: TypeParameterDeclaration[], heritageClauses: HeritageClause[], members: TypeElement[]) -> InterfaceDeclaration;
    auto createTypeAliasDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | Identifier, typeParameters: TypeParameterDeclaration[], type: TypeNode) -> TypeAliasDeclaration;
    auto updateTypeAliasDeclaration(node: TypeAliasDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, name: Identifier, typeParameters: TypeParameterDeclaration[], type: TypeNode) -> TypeAliasDeclaration;
    auto createEnumDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, name: string | Identifier, members: EnumMember[]) -> EnumDeclaration;
    auto updateEnumDeclaration(node: EnumDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, name: Identifier, members: EnumMember[]) -> EnumDeclaration;
    auto createModuleDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, name: ModuleName, body: ModuleBody, NodeFlags flags = (NodeFlags)0) -> ModuleDeclaration;
    auto updateModuleDeclaration(node: ModuleDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, name: ModuleName, body: ModuleBody) -> ModuleDeclaration;
    auto createModuleBlock(statements: Statement[]) -> ModuleBlock;
    auto updateModuleBlock(node: ModuleBlock, statements: Statement[]) -> ModuleBlock;
    auto createCaseBlock(clauses: CaseOrDefaultClause[]) -> CaseBlock;
    auto updateCaseBlock(node: CaseBlock, clauses: CaseOrDefaultClause[]) -> CaseBlock;
    auto createNamespaceExportDeclaration(name: string | Identifier) -> NamespaceExportDeclaration;
    auto updateNamespaceExportDeclaration(node: NamespaceExportDeclaration, name: Identifier) -> NamespaceExportDeclaration;
    auto createImportEqualsDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, isTypeOnly: boolean, name: string | Identifier, moduleReference: ModuleReference) -> ImportEqualsDeclaration;
    auto updateImportEqualsDeclaration(node: ImportEqualsDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, isTypeOnly: boolean, name: Identifier, moduleReference: ModuleReference) -> ImportEqualsDeclaration;
    auto createImportDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, importClause: ImportClause, moduleSpecifier: Expression) -> ImportDeclaration;
    auto updateImportDeclaration(node: ImportDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, importClause: ImportClause, moduleSpecifier: Expression) -> ImportDeclaration;
    auto createImportClause(isTypeOnly: boolean, name: Identifier, namedBindings: NamedImportBindings) -> ImportClause;
    auto updateImportClause(node: ImportClause, isTypeOnly: boolean, name: Identifier, namedBindings: NamedImportBindings) -> ImportClause;
    auto createNamespaceImport(name: Identifier) -> NamespaceImport;
    auto updateNamespaceImport(node: NamespaceImport, name: Identifier) -> NamespaceImport;
    auto createNamespaceExport(name: Identifier) -> NamespaceExport;
    auto updateNamespaceExport(node: NamespaceExport, name: Identifier) -> NamespaceExport;
    auto createNamedImports(elements: ImportSpecifier[]) -> NamedImports;
    auto updateNamedImports(node: NamedImports, elements: ImportSpecifier[]) -> NamedImports;
    auto createImportSpecifier(propertyName: Identifier, name: Identifier) -> ImportSpecifier;
    auto updateImportSpecifier(node: ImportSpecifier, propertyName: Identifier, name: Identifier) -> ImportSpecifier;
    auto createExportAssignment(decorators: DecoratorsArray, modifiers: ModifiersArray, isExportEquals: boolean, expression: Expression) -> ExportAssignment;
    auto updateExportAssignment(node: ExportAssignment, decorators: DecoratorsArray, modifiers: ModifiersArray, expression: Expression) -> ExportAssignment;
    auto createExportDeclaration(decorators: DecoratorsArray, modifiers: ModifiersArray, isTypeOnly: boolean, exportClause: NamedExportBindings, moduleSpecifier?: Expression) -> ExportDeclaration;
    auto updateExportDeclaration(node: ExportDeclaration, decorators: DecoratorsArray, modifiers: ModifiersArray, isTypeOnly: boolean, exportClause: NamedExportBindings, moduleSpecifier: Expression) -> ExportDeclaration;
    auto createNamedExports(elements: ExportSpecifier[]) -> NamedExports;
    auto updateNamedExports(node: NamedExports, elements: ExportSpecifier[]) -> NamedExports;
    auto createExportSpecifier(propertyName: string | Identifier, name: string | Identifier) -> ExportSpecifier;
    auto updateExportSpecifier(node: ExportSpecifier, propertyName: Identifier, name: Identifier) -> ExportSpecifier;
    /* @internal*/ auto createMissingDeclaration() -> MissingDeclaration;

    //
    // Module references
    //

    auto createExternalModuleReference(expression: Expression) -> ExternalModuleReference;
    auto updateExternalModuleReference(node: ExternalModuleReference, expression: Expression) -> ExternalModuleReference;

    //
    // JSDoc
    //

    auto createJSDocAllType() -> JSDocAllType;
    auto createJSDocUnknownType() -> JSDocUnknownType;
    auto createJSDocNonNullableType(type: TypeNode) -> JSDocNonNullableType;
    auto updateJSDocNonNullableType(node: JSDocNonNullableType, type: TypeNode) -> JSDocNonNullableType;
    auto createJSDocNullableType(type: TypeNode) -> JSDocNullableType;
    auto updateJSDocNullableType(node: JSDocNullableType, type: TypeNode) -> JSDocNullableType;
    auto createJSDocOptionalType(type: TypeNode) -> JSDocOptionalType;
    auto updateJSDocOptionalType(node: JSDocOptionalType, type: TypeNode) -> JSDocOptionalType;
    auto createJSDocFunctionType(parameters: ParameterDeclaration[], type: TypeNode) -> JSDocFunctionType;
    auto updateJSDocFunctionType(node: JSDocFunctionType, parameters: ParameterDeclaration[], type: TypeNode) -> JSDocFunctionType;
    auto createJSDocVariadicType(type: TypeNode) -> JSDocVariadicType;
    auto updateJSDocVariadicType(node: JSDocVariadicType, type: TypeNode) -> JSDocVariadicType;
    auto createJSDocNamepathType(type: TypeNode) -> JSDocNamepathType;
    auto updateJSDocNamepathType(node: JSDocNamepathType, type: TypeNode) -> JSDocNamepathType;
    auto createJSDocTypeExpression(type: TypeNode) -> JSDocTypeExpression;
    auto updateJSDocTypeExpression(node: JSDocTypeExpression, type: TypeNode) -> JSDocTypeExpression;
    auto createJSDocNameReference(name: EntityName) -> JSDocNameReference;
    auto updateJSDocNameReference(node: JSDocNameReference, name: EntityName) -> JSDocNameReference;
    auto createJSDocTypeLiteral(NodeArray<JSDocPropertyLikeTag> jsDocPropertyTags = undefined, boolean isArrayType = false) -> JSDocTypeLiteral;
    auto updateJSDocTypeLiteral(node: JSDocTypeLiteral, jsDocPropertyTags: JSDocPropertyLikeTag[], isArrayType: boolean) -> JSDocTypeLiteral;
    auto createJSDocSignature(typeParameters: JSDocTemplateTag[], parameters: JSDocParameterTag[], type?: JSDocReturnTag) -> JSDocSignature;
    auto updateJSDocSignature(node: JSDocSignature, typeParameters: JSDocTemplateTag[], parameters: JSDocParameterTag[], type: JSDocReturnTag) -> JSDocSignature;
    auto createJSDocTemplateTag(tagName: Identifier, constraint: JSDocTypeExpression, typeParameters: TypeParameterDeclaration[], comment?: string) -> JSDocTemplateTag;
    auto updateJSDocTemplateTag(node: JSDocTemplateTag, tagName: Identifier, constraint: JSDocTypeExpression, typeParameters: TypeParameterDeclaration[], comment: string) -> JSDocTemplateTag;
    auto createJSDocTypedefTag(tagName: Identifier, typeExpression?: JSDocTypeExpression | JSDocTypeLiteral, fullName?: Identifier | JSDocNamespaceDeclaration, comment?: string) -> JSDocTypedefTag;
    auto updateJSDocTypedefTag(node: JSDocTypedefTag, tagName: Identifier, typeExpression: JSDocTypeExpression | JSDocTypeLiteral, fullName: Identifier | JSDocNamespaceDeclaration, comment: string) -> JSDocTypedefTag;
    auto createJSDocParameterTag(tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression?: JSDocTypeExpression, boolean isNameFirst = false, comment?: string) -> JSDocParameterTag;
    auto updateJSDocParameterTag(node: JSDocParameterTag, tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression: JSDocTypeExpression, isNameFirst: boolean, comment: string) -> JSDocParameterTag;
    auto createJSDocPropertyTag(tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression?: JSDocTypeExpression, boolean isNameFirst = false, comment?: string) -> JSDocPropertyTag;
    auto updateJSDocPropertyTag(node: JSDocPropertyTag, tagName: Identifier, name: EntityName, isBracketed: boolean, typeExpression: JSDocTypeExpression, isNameFirst: boolean, comment: string) -> JSDocPropertyTag;
    auto createJSDocTypeTag(tagName: Identifier, typeExpression: JSDocTypeExpression, comment?: string) -> JSDocTypeTag;
    auto updateJSDocTypeTag(node: JSDocTypeTag, tagName: Identifier, typeExpression: JSDocTypeExpression, comment: string) -> JSDocTypeTag;
    auto createJSDocSeeTag(tagName: Identifier, nameExpression: JSDocNameReference, comment?: string) -> JSDocSeeTag;
    auto updateJSDocSeeTag(node: JSDocSeeTag, tagName: Identifier, nameExpression: JSDocNameReference, comment?: string) -> JSDocSeeTag;
    auto createJSDocReturnTag(tagName: Identifier, typeExpression?: JSDocTypeExpression, comment?: string) -> JSDocReturnTag;
    auto updateJSDocReturnTag(node: JSDocReturnTag, tagName: Identifier, typeExpression: JSDocTypeExpression, comment: string) -> JSDocReturnTag;
    auto createJSDocThisTag(tagName: Identifier, typeExpression: JSDocTypeExpression, comment?: string) -> JSDocThisTag;
    auto updateJSDocThisTag(node: JSDocThisTag, tagName: Identifier, typeExpression: JSDocTypeExpression, comment: string) -> JSDocThisTag;
    auto createJSDocEnumTag(tagName: Identifier, typeExpression: JSDocTypeExpression, comment?: string) -> JSDocEnumTag;
    auto updateJSDocEnumTag(node: JSDocEnumTag, tagName: Identifier, typeExpression: JSDocTypeExpression, comment: string) -> JSDocEnumTag;
    auto createJSDocCallbackTag(tagName: Identifier, typeExpression: JSDocSignature, fullName?: Identifier | JSDocNamespaceDeclaration, comment?: string) -> JSDocCallbackTag;
    auto updateJSDocCallbackTag(node: JSDocCallbackTag, tagName: Identifier, typeExpression: JSDocSignature, fullName: Identifier | JSDocNamespaceDeclaration, comment: string) -> JSDocCallbackTag;
    auto createJSDocAugmentsTag(tagName: Identifier, className: JSDocAugmentsTag["class"], comment?: string) -> JSDocAugmentsTag;
    auto updateJSDocAugmentsTag(node: JSDocAugmentsTag, tagName: Identifier, className: JSDocAugmentsTag["class"], comment: string) -> JSDocAugmentsTag;
    auto createJSDocImplementsTag(tagName: Identifier, className: JSDocImplementsTag["class"], comment?: string) -> JSDocImplementsTag;
    auto updateJSDocImplementsTag(node: JSDocImplementsTag, tagName: Identifier, className: JSDocImplementsTag["class"], comment: string) -> JSDocImplementsTag;
    auto createJSDocAuthorTag(tagName: Identifier, comment?: string) -> JSDocAuthorTag;
    auto updateJSDocAuthorTag(node: JSDocAuthorTag, tagName: Identifier, comment: string) -> JSDocAuthorTag;
    auto createJSDocClassTag(tagName: Identifier, comment?: string) -> JSDocClassTag;
    auto updateJSDocClassTag(node: JSDocClassTag, tagName: Identifier, comment: string) -> JSDocClassTag;
    auto createJSDocPublicTag(tagName: Identifier, comment?: string) -> JSDocPublicTag;
    auto updateJSDocPublicTag(node: JSDocPublicTag, tagName: Identifier, comment: string) -> JSDocPublicTag;
    auto createJSDocPrivateTag(tagName: Identifier, comment?: string) -> JSDocPrivateTag;
    auto updateJSDocPrivateTag(node: JSDocPrivateTag, tagName: Identifier, comment: string) -> JSDocPrivateTag;
    auto createJSDocProtectedTag(tagName: Identifier, comment?: string) -> JSDocProtectedTag;
    auto updateJSDocProtectedTag(node: JSDocProtectedTag, tagName: Identifier, comment: string) -> JSDocProtectedTag;
    auto createJSDocReadonlyTag(tagName: Identifier, comment?: string) -> JSDocReadonlyTag;
    auto updateJSDocReadonlyTag(node: JSDocReadonlyTag, tagName: Identifier, comment: string) -> JSDocReadonlyTag;
    auto createJSDocUnknownTag(tagName: Identifier, comment?: string) -> JSDocUnknownTag;
    auto updateJSDocUnknownTag(node: JSDocUnknownTag, tagName: Identifier, comment: string) -> JSDocUnknownTag;
    auto createJSDocDeprecatedTag(tagName: Identifier, comment?: string) -> JSDocDeprecatedTag;
    auto updateJSDocDeprecatedTag(node: JSDocDeprecatedTag, tagName: Identifier, comment?: string) -> JSDocDeprecatedTag;
    auto createJSDocComment(comment?: string, NodeArray<JSDocTag> tags = undefined) -> JSDoc;
    auto updateJSDocComment(node: JSDoc, comment: string, tags: JSDocTag[]) -> JSDoc;

    //
    // JSX
    //

    auto createJsxElement(openingElement: JsxOpeningElement, children: JsxChild[], closingElement: JsxClosingElement) -> JsxElement;
    auto updateJsxElement(node: JsxElement, openingElement: JsxOpeningElement, children: JsxChild[], closingElement: JsxClosingElement) -> JsxElement;
    auto createJsxSelfClosingElement(tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) -> JsxSelfClosingElement;
    auto updateJsxSelfClosingElement(node: JsxSelfClosingElement, tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) -> JsxSelfClosingElement;
    auto createJsxOpeningElement(tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) -> JsxOpeningElement;
    auto updateJsxOpeningElement(node: JsxOpeningElement, tagName: JsxTagNameExpression, typeArguments: TypeNode[], attributes: JsxAttributes) -> JsxOpeningElement;
    auto createJsxClosingElement(tagName: JsxTagNameExpression) -> JsxClosingElement;
    auto updateJsxClosingElement(node: JsxClosingElement, tagName: JsxTagNameExpression) -> JsxClosingElement;
    auto createJsxFragment(openingFragment: JsxOpeningFragment, children: JsxChild[], closingFragment: JsxClosingFragment) -> JsxFragment;
    auto createJsxText(text: string, boolean containsOnlyTriviaWhiteSpaces = false) -> JsxText;
    auto updateJsxText(node: JsxText, text: string, boolean containsOnlyTriviaWhiteSpaces = false) -> JsxText;
    auto createJsxOpeningFragment() -> JsxOpeningFragment;
    auto createJsxJsxClosingFragment() -> JsxClosingFragment;
    auto updateJsxFragment(node: JsxFragment, openingFragment: JsxOpeningFragment, children: JsxChild[], closingFragment: JsxClosingFragment) -> JsxFragment;
    auto createJsxAttribute(name: Identifier, initializer: StringLiteral | JsxExpression) -> JsxAttribute;
    auto updateJsxAttribute(node: JsxAttribute, name: Identifier, initializer: StringLiteral | JsxExpression) -> JsxAttribute;
    auto createJsxAttributes(properties: JsxAttributeLike[]) -> JsxAttributes;
    auto updateJsxAttributes(node: JsxAttributes, properties: JsxAttributeLike[]) -> JsxAttributes;
    auto createJsxSpreadAttribute(expression: Expression) -> JsxSpreadAttribute;
    auto updateJsxSpreadAttribute(node: JsxSpreadAttribute, expression: Expression) -> JsxSpreadAttribute;
    auto createJsxExpression(dotDotDotToken: DotDotDotToken, expression: Expression) -> JsxExpression;
    auto updateJsxExpression(node: JsxExpression, expression: Expression) -> JsxExpression;

    //
    // Clauses
    //

    auto createCaseClause(expression: Expression, statements: Statement[]) -> CaseClause;
    auto updateCaseClause(node: CaseClause, expression: Expression, statements: Statement[]) -> CaseClause;
    auto createDefaultClause(statements: Statement[]) -> DefaultClause;
    auto updateDefaultClause(node: DefaultClause, statements: Statement[]) -> DefaultClause;
    auto createHeritageClause(token: HeritageClause["token"], types: ExpressionWithTypeArguments[]) -> HeritageClause;
    auto updateHeritageClause(node: HeritageClause, types: ExpressionWithTypeArguments[]) -> HeritageClause;
    auto createCatchClause(variableDeclaration: string | VariableDeclaration, block: Block) -> CatchClause;
    auto updateCatchClause(node: CatchClause, variableDeclaration: VariableDeclaration, block: Block) -> CatchClause;

    //
    // Property assignments
    //

    auto createPropertyAssignment(name: string | PropertyName, initializer: Expression) -> PropertyAssignment;
    auto updatePropertyAssignment(node: PropertyAssignment, name: PropertyName, initializer: Expression) -> PropertyAssignment;
    auto createShorthandPropertyAssignment(name: string | Identifier, objectAssignmentInitializer?: Expression) -> ShorthandPropertyAssignment;
    auto updateShorthandPropertyAssignment(node: ShorthandPropertyAssignment, name: Identifier, objectAssignmentInitializer: Expression) -> ShorthandPropertyAssignment;
    auto createSpreadAssignment(expression: Expression) -> SpreadAssignment;
    auto updateSpreadAssignment(node: SpreadAssignment, expression: Expression) -> SpreadAssignment;

    //
    // Enum
    //

    auto createEnumMember(name: string | PropertyName, initializer?: Expression) -> EnumMember;
    auto updateEnumMember(node: EnumMember, name: PropertyName, initializer: Expression) -> EnumMember;

    //
    // Top-level nodes
    //

    auto createSourceFile(statements: Statement[], endOfFileToken: EndOfFileToken, flags: NodeFlags) -> SourceFile;
    auto updateSourceFile(node: SourceFile, statements: Statement[], boolean isDeclarationFile = false, NodeArray<FileReference> referencedFiles = undefined, NodeArray<FileReference> typeReferences = undefined, boolean hasNoDefaultLib = false, NodeArray<FileReference> libReferences = undefined) -> SourceFile;

    /* @internal */ auto createUnparsedSource(prologues: UnparsedPrologue[], syntheticReferences: UnparsedSyntheticReference[], texts: UnparsedSourceText[]) -> UnparsedSource;
    /* @internal */ auto createUnparsedPrologue(data?: string) -> UnparsedPrologue;
    /* @internal */ auto createUnparsedPrepend(data: string, texts: UnparsedSourceText[]) -> UnparsedPrepend;
    /* @internal */ auto createUnparsedTextLike(data: string, internal: boolean) -> UnparsedTextLike;
    /* @internal */ auto createUnparsedSyntheticReference(section: BundleFileHasNoDefaultLib | BundleFileReference) -> UnparsedSyntheticReference;
    /* @internal */ auto createInputFiles() -> InputFiles;

    //
    // Synthetic Nodes
    //
    /* @internal */ auto createSyntheticExpression(type: Type, boolean isSpread = false, tupleNameSource?: ParameterDeclaration | NamedTupleMember) -> SyntheticExpression;
    /* @internal */ auto createSyntaxList(children: Node[]) -> SyntaxList;

    //
    // Transformation nodes
    //

    auto createNotEmittedStatement(original: Node) -> NotEmittedStatement;
    /* @internal */ auto createEndOfDeclarationMarker(original: Node) -> EndOfDeclarationMarker;
    /* @internal */ auto createMergeDeclarationMarker(original: Node) -> MergeDeclarationMarker;
    auto createPartiallyEmittedExpression(expression: Expression, original?: Node) -> PartiallyEmittedExpression;
    auto updatePartiallyEmittedExpression(node: PartiallyEmittedExpression, expression: Expression) -> PartiallyEmittedExpression;
    /* @internal */ auto createSyntheticReferenceExpression(expression: Expression, thisArg: Expression) -> SyntheticReferenceExpression;
    /* @internal */ auto updateSyntheticReferenceExpression(node: SyntheticReferenceExpression, expression: Expression, thisArg: Expression) -> SyntheticReferenceExpression;
    auto createCommaListExpression(elements: Expression[]) -> CommaListExpression;
    auto updateCommaListExpression(node: CommaListExpression, elements: Expression[]) -> CommaListExpression;
    auto createBundle(sourceFiles: SourceFile[], NodeArray</*UnparsedSource | InputFiles*/Node> prepends = undefined) -> Bundle;
    auto updateBundle(node: Bundle, sourceFiles: SourceFile[], NodeArray</*UnparsedSource | InputFiles*/Node> prepends = undefined) -> Bundle;

    //
    // Common operators
    //

    auto createComma(left: Expression, right: Expression) -> BinaryExpression;
    auto createAssignment(left: ObjectLiteralExpression | ArrayLiteralExpression, right: Expression) -> DestructuringAssignment;
    auto createAssignment(left: Expression, right: Expression) -> AssignmentExpression<EqualsToken>;
    auto createLogicalOr(left: Expression, right: Expression) -> BinaryExpression;
    auto createLogicalAnd(left: Expression, right: Expression) -> BinaryExpression;
    auto createBitwiseOr(left: Expression, right: Expression) -> BinaryExpression;
    auto createBitwiseXor(left: Expression, right: Expression) -> BinaryExpression;
    auto createBitwiseAnd(left: Expression, right: Expression) -> BinaryExpression;
    auto createStrictEquality(left: Expression, right: Expression) -> BinaryExpression;
    auto createStrictInequality(left: Expression, right: Expression) -> BinaryExpression;
    auto createEquality(left: Expression, right: Expression) -> BinaryExpression;
    auto createInequality(left: Expression, right: Expression) -> BinaryExpression;
    auto createLessThan(left: Expression, right: Expression) -> BinaryExpression;
    auto createLessThanEquals(left: Expression, right: Expression) -> BinaryExpression;
    auto createGreaterThan(left: Expression, right: Expression) -> BinaryExpression;
    auto createGreaterThanEquals(left: Expression, right: Expression) -> BinaryExpression;
    auto createLeftShift(left: Expression, right: Expression) -> BinaryExpression;
    auto createRightShift(left: Expression, right: Expression) -> BinaryExpression;
    auto createUnsignedRightShift(left: Expression, right: Expression) -> BinaryExpression;
    auto createAdd(left: Expression, right: Expression) -> BinaryExpression;
    auto createSubtract(left: Expression, right: Expression) -> BinaryExpression;
    auto createMultiply(left: Expression, right: Expression) -> BinaryExpression;
    auto createDivide(left: Expression, right: Expression) -> BinaryExpression;
    auto createModulo(left: Expression, right: Expression) -> BinaryExpression;
    auto createExponent(left: Expression, right: Expression) -> BinaryExpression;
    auto createPrefixPlus(operand: Expression) -> PrefixUnaryExpression;
    auto createPrefixMinus(operand: Expression) -> PrefixUnaryExpression;
    auto createPrefixIncrement(operand: Expression) -> PrefixUnaryExpression;
    auto createPrefixDecrement(operand: Expression) -> PrefixUnaryExpression;
    auto createBitwiseNot(operand: Expression) -> PrefixUnaryExpression;
    auto createLogicalNot(operand: Expression) -> PrefixUnaryExpression;
    auto createPostfixIncrement(operand: Expression) -> PostfixUnaryExpression;
    auto createPostfixDecrement(operand: Expression) -> PostfixUnaryExpression;

    //
    // Compound Nodes
    //

    auto createImmediatelyInvokedFunctionExpression(statements: Statement[]) -> CallExpression;
    auto createImmediatelyInvokedFunctionExpression(statements: Statement[], param: ParameterDeclaration, paramValue: Expression) -> CallExpression;
    auto createImmediatelyInvokedArrowFunction(statements: Statement[]) -> CallExpression;
    auto createImmediatelyInvokedArrowFunction(statements: Statement[], param: ParameterDeclaration, paramValue: Expression) -> CallExpression;


    auto createVoidZero() -> VoidExpression;
    auto createExportDefault(expression: Expression) -> ExportAssignment;
    auto createExternalModuleExport(exportName: Identifier) -> ExportDeclaration;

    /* @internal */ auto createTypeCheck(value: Expression, tag: TypeOfTag) -> Expression;
    /* @internal */ auto createMethodCall(object: Expression, methodName: string | Identifier, argumentsList: Expression[]) -> CallExpression;
    /* @internal */ auto createGlobalMethodCall(globalObjectName: string, globalMethodName: string, argumentsList: Expression[]) -> CallExpression;
    /* @internal */ auto createFunctionBindCall(target: Expression, thisArg: Expression, argumentsList: Expression[]) -> CallExpression;
    /* @internal */ auto createFunctionCallCall(target: Expression, thisArg: Expression, argumentsList: Expression[]) -> CallExpression;
    /* @internal */ auto createFunctionApplyCall(target: Expression, thisArg: Expression, argumentsExpression: Expression) -> CallExpression;
    /* @internal */ auto createObjectDefinePropertyCall(target: Expression, propertyName: string | Expression, attributes: Expression) -> CallExpression;
    /* @internal */ auto createPropertyDescriptor(attributes: PropertyDescriptorAttributes, boolean singleLine = false) -> ObjectLiteralExpression;
    /* @internal */ auto createArraySliceCall(array: Expression, start?: number | Expression) -> CallExpression;
    /* @internal */ auto createArrayConcatCall(array: Expression, values: Expression[]) -> CallExpression;
    /* @internal */ auto createCallBinding(expression: Expression, recordTempVariable: (temp: Identifier) => void, languageVersion?: ScriptTarget, boolean cacheIdentifiers = false) -> CallBinding;
    /* @internal */ auto inlineExpressions(expressions: Expression[]) -> Expression;
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
    /* @internal */ auto getInternalName(node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
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
    /* @internal */ auto getLocalName(node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
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
    /* @internal */ auto getExportName(node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    /**
     * Gets the name of a declaration for use in declarations.
     *
     * @param node The declaration.
     * @param allowComments A value indicating whether comments may be emitted for the name.
     * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
     */
    /* @internal */ auto getDeclarationName(node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier;
    /**
     * Gets a namespace-qualified name for use in expressions.
     *
     * @param ns The namespace identifier.
     * @param name The name.
     * @param allowComments A value indicating whether comments may be emitted for the name.
     * @param allowSourceMaps A value indicating whether source maps may be emitted for the name.
     */
    /* @internal */ auto getNamespaceMemberName(ns: Identifier, name: Identifier, boolean allowComments = false, boolean allowSourceMaps = false) -> PropertyAccessExpression;
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
    /* @internal */ auto getExternalModuleOrNamespaceExportName(ns: Identifier, node: Declaration, boolean allowComments = false, boolean allowSourceMaps = false) -> Identifier | PropertyAccessExpression;

    //
    // Utilities
    //

    auto restoreOuterExpressions(outerExpression: Expression, innerExpression: Expression, kinds?: OuterExpressionKinds) -> Expression;
    /* @internal */ auto restoreEnclosingLabel(node: Statement, outermostLabeledStatement: LabeledStatement, afterRestoreLabelCallback?: (node: LabeledStatement) => void) -> Statement;
    /* @internal */ auto createUseStrictPrologue() -> PrologueDirective;
    /**
     * Copies any necessary standard and custom prologue-directives into target array.
     * @param source origin statements array
     * @param target result statements array
     * @param ensureUseStrict boolean determining whether the function need to add prologue-directives
     * @param visitor Optional callback used to visit any custom prologue directives.
     */
    /* @internal */ auto copyPrologue(source: Statement[], target: Push<Statement>, boolean ensureUseStrict = false, visitor?: (node: Node) => VisitResult<Node>) -> number;
    /**
     * Copies only the standard (string-expression) prologue-directives into the target statement-array.
     * @param source origin statements array
     * @param target result statements array
     * @param ensureUseStrict boolean determining whether the function need to add prologue-directives
     */
    /* @internal */ auto copyStandardPrologue(source: Statement[], target: Push<Statement>, boolean ensureUseStrict = false) -> number;
    /**
     * Copies only the custom prologue-directives into target statement-array.
     * @param source origin statements array
     * @param target result statements array
     * @param statementOffset The offset at which to begin the copy.
     * @param visitor Optional callback used to visit any custom prologue directives.
     */
    /* @internal */ auto copyCustomPrologue(source: Statement[], target: Push<Statement>, statementOffset: number, visitor?: (node: Node) => VisitResult<Node>, filter?: (node: Node) => boolean) -> number;
    /* @internal */ auto copyCustomPrologue(source: Statement[], target: Push<Statement>, statementOffset: number, visitor?: (node: Node) => VisitResult<Node>, filter?: (node: Node) => boolean) -> number;
    /* @internal */ auto ensureUseStrict(statements: NodeArray<Statement>) -> NodeArray<Statement>;
    /* @internal */ auto liftToBlock(nodes: Node[]) -> Statement;
    /**
     * Merges generated lexical declarations into a new statement list.
     */
    /* @internal */ auto mergeLexicalEnvironment(statements: NodeArray<Statement>, declarations: Statement[]) -> NodeArray<Statement>;
    /**
     * Appends generated lexical declarations to an array of statements.
     */
    /* @internal */ auto mergeLexicalEnvironment(statements: Statement[], declarations: Statement[]) -> Statement[];
    /**
     * Creates a shallow, memberwise clone of a node.
     * - The result will have its `original` pointer set to `node`.
     * - The result will have its `pos` and `end` set to `-1`.
     * - *DO NOT USE THIS* if a more appropriate function is available.
     */
    /* @internal */ template <typename T/*extends Node*/> auto cloneNode(node: T) -> T;
    /* @internal */ template <typename T/*extends HasModifiers*/> auto updateModifiers(node: T, modifiers: ModifiersArray | ModifierFlags) -> T;
};

#endif // NODEFACTORY_H