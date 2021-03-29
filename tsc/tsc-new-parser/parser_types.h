#ifndef NEW_PARSER_TYPES_H
#define NEW_PARSER_TYPES_H

#include "parser_fwd_types.h"

#include <vector>
#include <map>
#include <set>

namespace data
{
    using NodeId = number;

    using SymbolId = number;
    /* @internal */
    using TypeId = number;

    using any = char *;
    struct never
    {
    };

    using SymbolTable = std::map<string, Symbol>;

    ///////////////////////////////////////////////////////////////////////

    struct TextRange
    {
        TextRange() = default;

        number pos;
        number end;
    };

    struct ReadonlyTextRange : TextRange
    {
    };

    template <typename T /*extends Node*/>
    struct ReadonlyArray : std::vector<T>
    {
        using vector::vector;
    };

    template <typename T /*extends Node*/>
    struct NodeArray : ReadonlyArray<T>
    {
        using ReadonlyArray::ReadonlyArray;
        
        NodeArray() {}
        NodeArray(undefined_t) {}

        auto pop() -> T
        {
            auto v = back();
            pop_back();
            return v;
        }

        inline operator TextRange&()
        {
            return range;
        }

        inline operator bool()
        {
            return !empty();
        }

        inline auto operator->()
        {
            return this;
        }        

        inline auto operator==(undefined_t)
        {
            // TODO: review it
            return size() == 0;
        }

        inline auto operator!=(undefined_t)
        {
            return size() != 0;
        }        

        ReadonlyTextRange range;
        boolean hasTrailingComma;
        /* @internal */ TransformFlags transformFlags; // Flags for transforms, possibly undefined
        // to support MissingList
        boolean isMissingList;
    };

    using ModifiersArray = NodeArray<PTR(Modifier)>;

    struct Symbol
    {
        SymbolFlags flags;                                                          // Symbol flags
        string escapedName;                                                         // Name of symbol
        std::vector<Declaration> declarations;                                      // Declarations associated with this symbol
        PTR(Declaration) valueDeclaration;                                          // First value declaration of the symbol
        SymbolTable members;                                                        // Class, interface or object literal instance members
        SymbolTable exports;                                                        // Module exports
        SymbolTable globalExports;                                                  // Conditional global UMD exports
        /* @internal */ SymbolId id;                                                // Unique id (used to look up SymbolLinks)
        /* @internal */ number mergeId;                                             // Merge id (used to look up merged symbol)
        /* @internal */ PTR(Symbol) parent;                                         // Parent symbol
        /* @internal */ PTR(Symbol) exportSymbol;                                   // Exported symbol associated with this symbol
        /* @internal */ boolean constEnumOnlyModule;                                // True if module contains only const enums or other modules with only const enums
        /* @internal */ SymbolFlags isReferenced;                                   // True if the symbol is referenced elsewhere. Keeps track of the meaning of a reference in case a symbol is both a type parameter and parameter.
        /* @internal */ boolean isReplaceableByMethod;                              // Can this Javascript class property be replaced by a method symbol?
        /* @internal */ boolean isAssigned;                                         // True if the symbol is a parameter with assignments
        /* @internal */ std::map<number, Declaration> assignmentDeclarationMembers; // detected late-bound assignment declarations associated with the symbol
    };

    // Properties common to all types
    struct Type
    {
        TypeFlags flags;           // Flags
        /* @internal */ TypeId id; // Unique ID
        ///* @internal */ PTR(TypeChecker) checker;
        PTR(Symbol) symbol;                                            // Symbol associated with type (if any)
        PTR(DestructuringPattern) pattern;                        // Destructuring pattern represented by type (if any)
        PTR(Symbol) aliasSymbol;                                       // Alias associated with type
        std::vector<Type> aliasTypeArguments;                     // Alias type arguments (if any)
        /* @internal */ boolean aliasTypeArgumentsContainsMarker; // Alias type arguments (if any)
        /* @internal */
        PTR(Type) permissiveInstantiation; // Instantiation with type parameters mapped to wildcard type
        /* @internal */
        PTR(Type) restrictiveInstantiation; // Instantiation with type parameters mapped to unconstrained form
        /* @internal */
        PTR(Type) immediateBaseConstraint; // Immediate base constraint cache
        /* @internal */
        PTR(Type) widened; // Cached widened form of the type
    };

    struct Node : TextRange
    {
        virtual ~Node() {}

        Node() = default;
        Node(SyntaxKind kind, number pos, number end) : kind(kind), TextRange{pos, end} {}

        SyntaxKind kind;
        NodeFlags flags;
        /* @internal */ ModifierFlags modifierFlagsCache;
        /* @internal */ TransformFlags transformFlags; // Flags for transforms
        DecoratorsArray decorators;               // Array of decorators (in document order)
        ModifiersArray modifiers;                      // Array of modifiers
        /* @internal */ NodeId id;                     // Unique id (used to look up NodeLinks)
        PTR(Node) parent;                                   // Parent node (initialized by binding)
        /* @internal */ PTR(Node) original;                 // The original node if this is an updated node.
        /* @internal */ PTR(Symbol) symbol;                 // Symbol declared by node (initialized by binding)
        /* @internal */ SymbolTable locals;            // Locals associated with node (initialized by binding)
        /* @internal */ PTR(Node) nextContainer;            // Next container in declaration order (initialized by binding)
        /* @internal */ PTR(Symbol) localSymbol;            // Local symbol declared by node (initialized by binding only for exported nodes)
        ///* @internal */ PTR(FlowNode) flowNode;                  // Associated FlowNode (initialized by binding)
        ///* @internal */ PTR(EmitNode) emitNode;                  // Associated EmitNode (initialized by transforms)
        ///* @internal */ PTR(Type) contextualType;                // Used to temporarily assign a contextual type during overload resolution
        ///* @internal */ PTR(InferenceContext) inferenceContext;  // Inference context for contextual type
    };

    // TODO(rbuckton): Constraint 'TKind' to 'TokenSyntaxKind'
    template <SyntaxKind... TKind>
    struct Token : virtual Node
    {
    };

    struct JSDocContainer : virtual Node
    {
        /* @internal */ NodeArray<PTR(JSDoc)> jsDoc;         // JSDoc that directly precedes this node
        /* @internal */ NodeArray<PTR(JSDocTag)> jsDocCache; // Cache for getJSDocTags
    };

    struct EndOfFileToken : Token<SyntaxKind::EndOfFileToken>, JSDocContainer
    {
    };

    // Punctuation
    template <SyntaxKind TKind>
    struct PunctuationToken : Token<TKind>
    {
    };

    // Keywords
    template <SyntaxKind TKind>
    struct KeywordToken : Token<TKind>
    {
    };

    template <SyntaxKind TKind>
    struct ModifierToken : KeywordToken<TKind>
    {
    };

    struct QualifiedName : virtual Node
    {
        // kind: SyntaxKind::QualifiedName;
        PTR(EntityName) left;
        PTR(Identifier) right;
        /*@internal*/ number jsdocDotPos; // QualifiedName occurs in JSDoc-style generic: Id1.Id2.<T>
    };

    struct Declaration : virtual Node
    {
        any _declarationBrand;
    };

    struct DeclarationNoNode
    {
        any _declarationBrand;
    };

    struct NamedDeclaration : Declaration
    {
        PTR(DeclarationName) name;
    };

    struct NamedDeclarationNoNode : DeclarationNoNode
    {
        PTR(DeclarationName) name;
    };

    struct TypeElement : NamedDeclaration
    {
        any _typeElementBrand;
        PTR(PropertyName) name;
        PTR(QuestionToken) questionToken;
    };

    /* @internal */
    struct DynamicNamedDeclaration : NamedDeclaration
    {
        PTR(ComputedPropertyName) name;
    };

    /* @internal */
    // A declaration that supports late-binding (used in checker)
    struct LateBoundDeclaration : DynamicNamedDeclaration
    {
        PTR(LateBoundName) name;
    };

    struct Statement : virtual Node
    {
        any _statementBrand;
    };

    struct DeclarationStatement : NamedDeclaration, Statement
    {
        PTR(Node) /**Identifier | StringLiteral | NumericLiteral*/ name;
    };

    struct ComputedPropertyName : virtual Node
    {
        // kind: SyntaxKind::ComputedPropertyName;
        PTR(Declaration) parent;
        PTR(Expression) expression;
    };

    struct PrivateIdentifier : virtual Node
    {
        PrivateIdentifier() = default;
        PrivateIdentifier(SyntaxKind kind, number pos, number end) : Node{kind, pos, end} {}

        // kind: SyntaxKind::PrivateIdentifier;
        // escaping not strictly necessary
        // avoids gotchas in transforms and utils
        string escapedText;
    };

    /* @internal */
    // A name that supports late-binding (used in checker)
    struct LateBoundName : ComputedPropertyName
    {
        PTR(EntityNameExpression) expression;
    };

    struct Decorator : virtual Node
    {
        // kind: SyntaxKind::Decorator;
        PTR(NamedDeclaration) parent;
        PTR(LeftHandSideExpression) expression;
    };

    struct TypeParameterDeclaration : NamedDeclaration
    {
        // kind: SyntaxKind::TypeParameter;
        PTR(Node) /**DeclarationWithTypeParameterChildren | InferTypeNode*/ parent;
        PTR(Identifier) name;
        /** Note: Consider calling `getEffectiveConstraintOfTypeParameter` */
        PTR(TypeNode) constraint;
        PTR(TypeNode) _default;

        // For error recovery purposes.
        PTR(Expression) expression;
    };

    struct SignatureDeclarationBase : NamedDeclaration, JSDocContainer
    {
        PTR(PropertyName) name;
        NodeArray<PTR(TypeParameterDeclaration)> typeParameters;
        NodeArray<PTR(ParameterDeclaration)> parameters;
        PTR(TypeNode) type;
        /* @internal */ NodeArray<PTR(TypeNode)> typeArguments; // Used for quick info, replaces typeParameters for instantiated signatures
    };

    struct SignatureDeclarationBaseNoNode : NamedDeclarationNoNode, JSDocContainer
    {
        PTR(PropertyName) name;
        NodeArray<PTR(TypeParameterDeclaration)> typeParameters;
        NodeArray<PTR(ParameterDeclaration)> parameters;
        PTR(TypeNode) type;
        /* @internal */ NodeArray<PTR(TypeNode)> typeArguments; // Used for quick info, replaces typeParameters for instantiated signatures
    };

    struct CallSignatureDeclaration : SignatureDeclarationBase, TypeElement
    {
        // kind: SyntaxKind::CallSignature;
    };

    struct ConstructSignatureDeclaration : SignatureDeclarationBase, TypeElement
    {
        // kind: SyntaxKind::ConstructSignature;
    };

    struct VariableDeclaration : NamedDeclaration, JSDocContainer
    {
        // kind: SyntaxKind::VariableDeclaration;
        PTR(Node) /**VariableDeclarationList | CatchClause*/ parent;
        PTR(BindingName) name;                  // Declared variable name
        PTR(ExclamationToken) exclamationToken; // Optional definite assignment assertion
        PTR(TypeNode) type;                     // Optional type annotation
        PTR(Expression) initializer;            // Optional initializer
    };

    /* @internal */
    struct InitializedVariableDeclaration : VariableDeclaration
    {
        PTR(Expression) initializer;
    };

    struct VariableDeclarationList : virtual Node
    {
        // kind: SyntaxKind::VariableDeclarationList;
        PTR(Node) /**VariableStatement | ForStatement | ForOfStatement | ForInStatement*/ parent;
        NodeArray<PTR(VariableDeclaration)> declarations;
    };

    struct ParameterDeclaration : NamedDeclaration, JSDocContainer
    {
        // kind: SyntaxKind::Parameter;
        PTR(SignatureDeclaration) parent;
        PTR(DotDotDotToken) dotDotDotToken; // Present on rest parameter
        PTR(BindingName) name;              // Declared parameter name.
        PTR(QuestionToken) questionToken;   // Present on optional parameter
        PTR(TypeNode) type;                 // Optional type annotation
        PTR(Expression) initializer;        // Optional initializer
    };

    struct BindingElement : NamedDeclaration
    {
        // kind: SyntaxKind::BindingElement;
        PTR(BindingPattern) parent;
        PTR(PropertyName) propertyName;     // Binding property name (in object binding pattern)
        PTR(DotDotDotToken) dotDotDotToken; // Present on rest element (in object binding pattern)
        PTR(BindingName) name;              // Declared binding element name
        PTR(Expression) initializer;        // Optional initializer
    };

    struct PropertySignature : TypeElement, JSDocContainer
    {
        // kind: SyntaxKind::PropertySignature;
        PTR(PropertyName) name;           // Declared property name
        PTR(QuestionToken) questionToken; // Present on optional property
        PTR(TypeNode) type;               // Optional type annotation
        PTR(Expression) initializer;      // Present for use with reporting a grammar error
    };

    struct ClassElement : NamedDeclaration
    {
        any _classElementBrand;
        PTR(PropertyName) name;
    };

    struct PropertyDeclaration : ClassElement, JSDocContainer
    {
        // kind: SyntaxKind::PropertyDeclaration;
        PTR(ClassLikeDeclaration) parent;
        PTR(PropertyName) name;
        PTR(QuestionToken) questionToken; // Present for use with reporting a grammar error
        PTR(ExclamationToken) exclamationToken;
        PTR(TypeNode) type;
        PTR(Expression) initializer; // Optional initializer
    };

    /*@internal*/
    struct PrivateIdentifierPropertyDeclaration : PropertyDeclaration
    {
        PTR(PrivateIdentifier) name;
    };

    /* @internal */
    struct InitializedPropertyDeclaration : PropertyDeclaration
    {
        PTR(Expression) initializer;
    };

    struct ObjectLiteralElement : NamedDeclaration
    {
        any _objectLiteralBrand;
        PTR(PropertyName) name;
    };

    struct PropertyAssignment : ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::PropertyAssignment;
        PTR(ObjectLiteralExpression) parent;
        PTR(PropertyName) name;
        PTR(QuestionToken) questionToken;       // Present for use with reporting a grammar error
        PTR(ExclamationToken) exclamationToken; // Present for use with reporting a grammar error
        PTR(Expression) initializer;
    };

    struct ShorthandPropertyAssignment : ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::ShorthandPropertyAssignment;
        PTR(ObjectLiteralExpression) parent;
        PTR(Identifier) name;
        PTR(QuestionToken) questionToken;
        PTR(ExclamationToken) exclamationToken;
        // used when ObjectLiteralExpression is used in ObjectAssignmentPattern
        // it is a grammar error to appear in actual object initializer:
        PTR(EqualsToken) equalsToken;
        PTR(Expression) objectAssignmentInitializer;
    };

    struct SpreadAssignment : ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::SpreadAssignment;
        PTR(ObjectLiteralExpression) parent;
        PTR(Expression) expression;
    };

    struct PropertyLikeDeclaration : NamedDeclaration
    {
        PTR(PropertyName) name;
    };

    struct ObjectBindingPattern : virtual Node
    {
        // kind: SyntaxKind::ObjectBindingPattern;
        PTR(Node) /**VariableDeclaration | ParameterDeclaration | BindingElement*/ parent;
        NodeArray<PTR(BindingElement)> elements;
    };

    struct ArrayBindingPattern : virtual Node
    {
        // kind: SyntaxKind::ArrayBindingPattern;
        PTR(Node) /**VariableDeclaration | ParameterDeclaration | BindingElement*/ parent;
        NodeArray<PTR(ArrayBindingElement)> elements;
    };

    /**
 * Several node kinds share function-like features such as a signature,
 * a name, and a body. These nodes should extend FunctionLikeDeclarationBase.
 * Examples:
 * - FunctionDeclaration
 * - MethodDeclaration
 * - AccessorDeclaration
 */
    struct FunctionLikeDeclarationBase : SignatureDeclarationBase
    {
        any _functionLikeDeclarationBrand;

        PTR(AsteriskToken) asteriskToken;
        PTR(QuestionToken) questionToken;
        PTR(ExclamationToken) exclamationToken;
        PTR(Node) /**Block | Expression*/ body;
        ///* @internal */ PTR(FlowNode) endFlowNode;
        ///* @internal */ PTR(FlowNode) returnFlowNode;
    };

    struct FunctionDeclaration : FunctionLikeDeclarationBase, DeclarationStatement
    {
        // kind: SyntaxKind::FunctionDeclaration;
        PTR(Identifier) name;
        PTR(FunctionBody) body;
    };

    struct MethodSignature : SignatureDeclarationBase, TypeElement
    {
        // kind: SyntaxKind::MethodSignature;
        PTR(ObjectTypeDeclaration) parent;
        PTR(PropertyName) name;
    };

    // Note that a MethodDeclaration is considered both a ClassElement and an ObjectLiteralElement.
    // Both the grammars for ClassDeclaration and ObjectLiteralExpression allow for MethodDeclarations
    // as child elements, and so a MethodDeclaration satisfies both interfaces.  This avoids the
    // alternative where we would need separate kinds/types for ClassMethodDeclaration and
    // ObjectLiteralMethodDeclaration, which would look identical.
    //
    // Because of this, it may be necessary to determine what sort of MethodDeclaration you have
    // at later stages of the compiler pipeline.  In that case, you can either check the parent kind
    // of the method, or use helpers like isObjectLiteralMethodDeclaration
    struct MethodDeclaration : FunctionLikeDeclarationBase, ClassElement, ObjectLiteralElement
    {
        // kind: SyntaxKind::MethodDeclaration;
        PTR(Node) /**ClassLikeDeclaration | ObjectLiteralExpression*/ parent;
        PTR(PropertyName) name;
        PTR(FunctionBody) body;
        /* @internal*/ PTR(ExclamationToken) exclamationToken; // Present for use with reporting a grammar error
    };

    struct ConstructorDeclaration : FunctionLikeDeclarationBase, ClassElement
    {
        // kind: SyntaxKind::Constructor;
        PTR(ClassLikeDeclaration) parent;
        PTR(FunctionBody) body;
        /* @internal */ NodeArray<PTR(TypeParameterDeclaration)> typeParameters; // Present for use with reporting a grammar error
        /* @internal */ PTR(TypeNode) type;                                      // Present for use with reporting a grammar error
    };

    /** For when we encounter a semicolon in a class declaration. ES6 allows these as class elements. */
    struct SemicolonClassElement : ClassElement
    {
        // kind: SyntaxKind::SemicolonClassElement;
        PTR(ClassLikeDeclaration) parent;
    };

    // See the comment on MethodDeclaration for the intuition behind GetAccessorDeclaration being a
    // ClassElement and an ObjectLiteralElement.
    struct GetAccessorDeclaration : FunctionLikeDeclarationBase/*, ClassElement, ObjectLiteralElement*/ // ClassElement and ObjectLiteralElement contains all fields in FunctionLikeDeclarationBase
    {
        // kind: SyntaxKind::GetAccessor;
        PTR(Node) /**ClassLikeDeclaration | ObjectLiteralExpression*/ parent;
        PTR(PropertyName) name;
        PTR(FunctionBody) body;
        /* @internal */ NodeArray<PTR(TypeParameterDeclaration)> typeParameters; // Present for use with reporting a grammar error
    };

    // See the comment on MethodDeclaration for the intuition behind SetAccessorDeclaration being a
    // ClassElement and an ObjectLiteralElement.
    struct SetAccessorDeclaration : FunctionLikeDeclarationBase/*, ClassElement, ObjectLiteralElement*/ // ClassElement and ObjectLiteralElement contains all fields in FunctionLikeDeclarationBase
    {
        // kind: SyntaxKind::SetAccessor;
        PTR(Node) /**ClassLikeDeclaration | ObjectLiteralExpression*/ parent;
        PTR(PropertyName) name;
        PTR(FunctionBody) body;
        /* @internal */ NodeArray<PTR(TypeParameterDeclaration)> typeParameters; // Present for use with reporting a grammar error
        /* @internal */ PTR(TypeNode) type;                                      // Present for use with reporting a grammar error
    };

    struct IndexSignatureDeclaration : SignatureDeclarationBase, ClassElement, TypeElement
    {
        // kind: SyntaxKind::IndexSignature;
        PTR(ObjectTypeDeclaration) parent;
        PTR(TypeNode) type;
    };

    struct TypeNode : virtual Node
    {
        // kind: TypeNodeSyntaxKind;
        any _typeNodeBrand;
    };

    template <SyntaxKind TKind>
    struct KeywordTypeNode : KeywordToken<TKind>, TypeNode
    {
    };

    struct NodeWithTypeArguments : TypeNode
    {
        NodeArray<PTR(TypeNode)> typeArguments;
    };

    struct ImportTypeNode : NodeWithTypeArguments
    {
        // kind: SyntaxKind::ImportType;
        boolean isTypeOf;
        PTR(TypeNode) argument;
        PTR(EntityName) qualifier;
    };

    struct LiteralTypeNode : TypeNode
    {
        // kind: SyntaxKind::LiteralType;
        PTR(Node) /**NullLiteral | BooleanLiteral | LiteralExpression | PrefixUnaryExpression*/ literal;
    };

    /* @internal */
    struct argumentType_ : LiteralTypeNode
    {
        PTR(StringLiteral) iteral;
    };

    struct LiteralImportTypeNode : ImportTypeNode
    {
        argumentType_ argument;
    };

    struct ThisTypeNode : TypeNode
    {
        // kind: SyntaxKind::ThisType;
    };

    struct FunctionOrConstructorTypeNodeBase : TypeNode, SignatureDeclarationBase
    {
        using TypeNode::Node::modifiers;
        // kind: SyntaxKind::FunctionType | SyntaxKind::ConstructorType;
        PTR(TypeNode) type;
    };

    struct FunctionTypeNode : FunctionOrConstructorTypeNodeBase
    {
        // kind: SyntaxKind::FunctionType;
    };

    struct ConstructorTypeNode : FunctionOrConstructorTypeNodeBase
    {
        // kind: SyntaxKind::ConstructorType;
    };

    struct TypeReferenceNode : NodeWithTypeArguments
    {
        // kind: SyntaxKind::TypeReference;
        PTR(EntityName) typeName;
    };

    struct TypePredicateNode : TypeNode
    {
        // kind: SyntaxKind::TypePredicate;
        PTR(Node) /**SignatureDeclaration | JSDocTypeExpression*/ parent;
        PTR(AssertsToken) assertsModifier;
        PTR(Node) /**Identifier | ThisTypeNode*/ parameterName;
        PTR(TypeNode) type;
    };

    struct TypeQueryNode : TypeNode
    {
        // kind: SyntaxKind::TypeQuery;
        PTR(EntityName) exprName;
    };

    // A TypeLiteral is the declaration node for an anonymous symbol.
    struct TypeLiteralNode : TypeNode, Declaration
    {
        // kind: SyntaxKind::TypeLiteral;
        NodeArray<PTR(TypeElement)> members;
    };

    struct ArrayTypeNode : TypeNode
    {
        // kind: SyntaxKind::ArrayType;
        PTR(TypeNode) elementType;
    };

    struct TupleTypeNode : TypeNode
    {
        // kind: SyntaxKind::TupleType;
        NodeArray<PTR(Node /*TypeNode | NamedTupleMember*/)> elements;
    };

    struct NamedTupleMember : TypeNode, JSDocContainer, Declaration
    {
        // kind: SyntaxKind::NamedTupleMember;
        Token<SyntaxKind::DotDotDotToken> dotDotDotToken;
        PTR(Identifier) name;
        Token<SyntaxKind::QuestionToken> questionToken;
        PTR(TypeNode) type;
    };

    struct OptionalTypeNode : TypeNode
    {
        // kind: SyntaxKind::OptionalType;
        PTR(TypeNode) type;
    };

    struct RestTypeNode : TypeNode
    {
        // kind: SyntaxKind::RestType;
        PTR(TypeNode) type;
    };

    struct UnionTypeNode : TypeNode
    {
        // kind: SyntaxKind::UnionType;
        NodeArray<PTR(TypeNode)> types;
    };

    struct IntersectionTypeNode : TypeNode
    {
        // kind: SyntaxKind::IntersectionType;
        NodeArray<PTR(TypeNode)> types;
    };

    struct ConditionalTypeNode : TypeNode
    {
        // kind: SyntaxKind::ConditionalType;
        PTR(TypeNode) checkType;
        PTR(TypeNode) extendsType;
        PTR(TypeNode) trueType;
        PTR(TypeNode) falseType;
    };

    struct InferTypeNode : TypeNode
    {
        // kind: SyntaxKind::InferType;
        PTR(TypeParameterDeclaration) typeParameter;
    };

    struct ParenthesizedTypeNode : TypeNode
    {
        // kind: SyntaxKind::ParenthesizedType;
        PTR(TypeNode) type;
    };

    struct TypeOperatorNode : TypeNode
    {
        // kind: SyntaxKind::TypeOperator;
        SyntaxKind _operator;
        PTR(TypeNode) type;
    };

    /* @internal */
    struct UniqueTypeOperatorNode : TypeOperatorNode
    {
        SyntaxKind _operator;
    };

    struct IndexedAccessTypeNode : TypeNode
    {
        // kind: SyntaxKind::IndexedAccessType;
        PTR(TypeNode) objectType;
        PTR(TypeNode) indexType;
    };

    struct MappedTypeNode : TypeNode, Declaration
    {
        // kind: SyntaxKind::MappedType;
        PTR(Node) /**ReadonlyToken | PlusToken | MinusToken*/ readonlyToken;
        PTR(TypeParameterDeclaration) typeParameter;
        PTR(TypeNode) nameType;
        PTR(Node) /**QuestionToken | PlusToken | MinusToken*/ questionToken;
        PTR(TypeNode) type;
    };

    struct TemplateLiteralTypeNode : TypeNode
    {
        PTR(TemplateHead) head;
        NodeArray<PTR(TemplateLiteralTypeSpan)> templateSpans;
    };

    struct TemplateLiteralTypeSpan : TypeNode
    {
        PTR(TypeNode) type;
        PTR(Node) /**TemplateMiddle | TemplateTail*/ literal;
    };

    // Note: 'brands' in our syntax nodes serve to give us a small amount of nominal typing.
    // Consider 'Expression'.  Without the brand, 'Expression' is actually no different
    // (structurally) than 'Node'.  Because of this you can pass any Node to a function that
    // takes an Expression without any error.  By using the 'brands' we ensure that the type
    // checker actually thinks you have something of the right type.  Note: the brands are
    // never actually given values.  At runtime they have zero cost.

    struct Expression : virtual Node
    {
        any _expressionBrand;
    };

    struct OmittedExpression : Expression
    {
        // kind: SyntaxKind::OmittedExpression;
    };

    struct UnaryExpression : Expression
    {
        any _unaryExpressionBrand;
    };

    /** Deprecated, please use UpdateExpression */
    struct UpdateExpression : UnaryExpression
    {
        any _updateExpressionBrand;
    };

    // see: https://tc39.github.io/ecma262/#prod-UpdateExpression
    // see: https://tc39.github.io/ecma262/#prod-UnaryExpression
    using PrefixUnaryOperator = SyntaxKind;

    struct PrefixUnaryExpression : UpdateExpression
    {
        // kind: SyntaxKind::PrefixUnaryExpression;
        PrefixUnaryOperator _operator;
        PTR(UnaryExpression) operand;
    };

    // see: https://tc39.github.io/ecma262/#prod-UpdateExpression
    using PostfixUnaryOperator = SyntaxKind;

    struct PostfixUnaryExpression : UpdateExpression
    {
        // kind: SyntaxKind::PostfixUnaryExpression;
        PTR(LeftHandSideExpression) operand;
        PostfixUnaryOperator _operator;
    };

    struct LeftHandSideExpression : UpdateExpression
    {
        any _leftHandSideExpressionBrand;
    };

    // Represents an expression that is elided as part of a transformation to emit comments on a
    // not-emitted node. The 'expression' property of a PartiallyEmittedExpression should be emitted.
    struct PartiallyEmittedExpression : LeftHandSideExpression
    {
        // kind: SyntaxKind::PartiallyEmittedExpression;
        PTR(Expression) expression;
    };

    struct MemberExpression : LeftHandSideExpression
    {
        any _memberExpressionBrand;
    };

    struct PrimaryExpression : MemberExpression
    {
        any _primaryExpressionBrand;
    };

    // The text property of a LiteralExpression stores the interpreted value of the literal in text form. For a StringLiteral,
    // or any literal of a template, this means quotes have been removed and escapes have been converted to actual characters.
    // For a NumericLiteral, the stored value is the toString() representation of the number. For example 1, 1.00, and 1e0 are all stored as just "1".
    struct LiteralLikeNode : PrimaryExpression
    {
        string text;
        boolean isUnterminated;
        boolean hasExtendedUnicodeEscape;
    };

    // The text property of a LiteralExpression stores the interpreted value of the literal in text form. For a StringLiteral,
    // or any literal of a template, this means quotes have been removed and escapes have been converted to actual characters.
    // For a NumericLiteral, the stored value is the toString() representation of the number. For example 1, 1.00, and 1e0 are all stored as just "1".
    struct LiteralExpression : LiteralLikeNode
    {
        any _literalExpressionBrand;
    };

    template <SyntaxKind TKind>
    struct LiteralToken : LiteralLikeNode
    {
    };

    struct StringLiteral : LiteralExpression, Declaration
    {
        // kind: SyntaxKind::StringLiteral;
        /* @internal */ PTR(Node) /**Identifier | StringLiteralLike | NumericLiteral*/ textSourceNode; // Allows a StringLiteral to get its text from another node (used by transforms).
                                                                                                 /** Note: this is only set when synthesizing a node, not during parsing. */
        /* @internal */ boolean singleQuote;
    };

    // TODO: review Declaration
    struct Identifier : PrimaryExpression/*, Declaration*/
    {
        Identifier() = default;
        Identifier(SyntaxKind kind_, number pos_, number end_) 
        {
            kind = kind_;
            pos = pos_;
            end = end_;
        }

        // kind: SyntaxKind::Identifier;
        /**
     * Prefer to use `id.unescapedText`. (Note: This is available only in services, not internally to the TypeScript compiler.)
     * Text of identifier, but if the identifier begins with two underscores, this will begin with three.
     */
        string escapedText;
        SyntaxKind originalKeywordKind;                                                      // Original syntaxKind which get set so that we can report an error later
        /*@internal*/ GeneratedIdentifierFlags autoGenerateFlags;                            // Specifies whether to auto-generate the text for an identifier.
        /*@internal*/ number autoGenerateId;                                                 // Ensures unique generated identifiers get unique names, but clones get the same name.
        /*@internal*/ PTR(ImportSpecifier) generatedImportReference;                         // Reference to the generated import specifier this identifier refers to
        boolean isInJSDocNamespace;                                                          // if the node is a member in a JSDoc namespace
        /*@internal*/ NodeArray<PTR(Node /*TypeNode | TypeParameterDeclaration*/)> typeArguments; // Only defined on synthesized nodes. Though not syntactically valid, used in emitting diagnostics, quickinfo, and signature help.
        /*@internal*/ number jsdocDotPos;                                                    // Identifier occurs in JSDoc-style generic: Id.<T>
    };

    // Transient identifier node (marked by id === -1)
    struct TransientIdentifier : Identifier
    {
        PTR(Symbol) resolvedSymbol;
    };

    /*@internal*/
    struct GeneratedIdentifier : Identifier
    {
        GeneratedIdentifierFlags autoGenerateFlags;
    };

    struct NullLiteral : PrimaryExpression
    {
        // kind: SyntaxKind::NullKeyword;
    };

    struct TrueLiteral : PrimaryExpression
    {
        // kind: SyntaxKind::TrueKeyword;
    };

    struct FalseLiteral : PrimaryExpression
    {
        // kind: SyntaxKind::FalseKeyword;
    };

    struct ThisExpression : PrimaryExpression
    {
        // kind: SyntaxKind::ThisKeyword;
    };

    struct SuperExpression : PrimaryExpression
    {
        // kind: SyntaxKind::SuperKeyword;
    };

    struct ImportExpression : PrimaryExpression
    {
        // kind: SyntaxKind::ImportKeyword;
    };

    struct DeleteExpression : UnaryExpression
    {
        // kind: SyntaxKind::DeleteExpression;
        PTR(UnaryExpression) expression;
    };

    struct TypeOfExpression : UnaryExpression
    {
        // kind: SyntaxKind::TypeOfExpression;
        PTR(UnaryExpression) expression;
    };

    struct VoidExpression : UnaryExpression
    {
        // kind: SyntaxKind::VoidExpression;
        PTR(UnaryExpression) expression;
    };

    struct AwaitExpression : UnaryExpression
    {
        // kind: SyntaxKind::AwaitExpression;
        PTR(UnaryExpression) expression;
    };

    struct YieldExpression : Expression
    {
        // kind: SyntaxKind::YieldExpression;
        PTR(AsteriskToken) asteriskToken;
        PTR(Expression) expression;
    };

    struct SyntheticExpression : Expression
    {
        // kind: SyntaxKind::SyntheticExpression;
        boolean isSpread;
        PTR(Type) type;
        PTR(Node) /**ParameterDeclaration | NamedTupleMember*/ tupleNameSource;
    };

    // see: https://tc39.github.io/ecma262/#prod-ExponentiationExpression
    using ExponentiationOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-MultiplicativeOperator
    using MultiplicativeOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-MultiplicativeExpression
    using MultiplicativeOperatorOrHigher = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-AdditiveExpression
    using AdditiveOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-AdditiveExpression
    using AdditiveOperatorOrHigher = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-ShiftExpression
    using ShiftOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-ShiftExpression
    using ShiftOperatorOrHigher = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-RelationalExpression
    using RelationalOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-RelationalExpression
    using RelationalOperatorOrHigher = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-EqualityExpression
    using EqualityOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-EqualityExpression
    using EqualityOperatorOrHigher = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-BitwiseANDExpression
    // see: https://tc39.github.io/ecma262/#prod-BitwiseXORExpression
    // see: https://tc39.github.io/ecma262/#prod-BitwiseORExpression
    using BitwiseOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-BitwiseANDExpression
    // see: https://tc39.github.io/ecma262/#prod-BitwiseXORExpression
    // see: https://tc39.github.io/ecma262/#prod-BitwiseORExpression
    using BitwiseOperatorOrHigher = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-LogicalANDExpression
    // see: https://tc39.github.io/ecma262/#prod-LogicalORExpression
    using LogicalOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-LogicalANDExpression
    // see: https://tc39.github.io/ecma262/#prod-LogicalORExpression
    using LogicalOperatorOrHigher = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-AssignmentOperator
    using CompoundAssignmentOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-AssignmentExpression
    using AssignmentOperator = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-AssignmentExpression
    using AssignmentOperatorOrHigher = SyntaxKind;

    // see: https://tc39.github.io/ecma262/#prod-Expression
    using BinaryOperator = SyntaxKind;

    using LogicalOrCoalescingAssignmentOperator = SyntaxKind;

    struct BinaryExpression : Expression, Declaration
    {
        // kind: SyntaxKind::BinaryExpression;
        PTR(Expression) left;
        PTR(BinaryOperatorToken) operatorToken;
        PTR(Expression) right;
        SyntaxKind cachedLiteralKind;
    };

    /* @internal */
    struct DynamicNamedBinaryExpression : BinaryExpression
    {
        PTR(ElementAccessExpression) left;
    };

    /* @internal */
    struct LateBoundBinaryExpressionDeclaration : DynamicNamedBinaryExpression
    {
        PTR(LateBoundElementAccessExpression) left;
    };

    using AssignmentOperatorToken = Token<SyntaxKind::EqualsToken, SyntaxKind::QuestionQuestionEqualsToken /*to keep it short, [from, to]*/>;

    template <typename TOperator /*AssignmentOperatorToken*/>
    struct AssignmentExpression : BinaryExpression
    {
        PTR(LeftHandSideExpression) left;
        PTR(TOperator) operatorToken;
    };

    struct ObjectDestructuringAssignment : AssignmentExpression<EqualsToken>
    {
        PTR(ObjectLiteralExpression) left;
    };

    struct ArrayDestructuringAssignment : AssignmentExpression<EqualsToken>
    {
        PTR(ArrayLiteralExpression) left;
    };

    struct ConditionalExpression : Expression
    {
        // kind: SyntaxKind::ConditionalExpression;
        PTR(Expression) condition;
        PTR(QuestionToken) questionToken;
        PTR(Expression) whenTrue;
        PTR(ColonToken) colonToken;
        PTR(Expression) whenFalse;
    };

    struct FunctionExpression : PrimaryExpression, FunctionLikeDeclarationBase
    {
        // kind: SyntaxKind::FunctionExpression;
        PTR(Identifier) name;
        PTR(FunctionBody) body; // Required, whereas the member inherited from FunctionDeclaration is optional
    };

    struct ArrowFunction : Expression, FunctionLikeDeclarationBase
    {
        // kind: SyntaxKind::ArrowFunction;
        PTR(EqualsGreaterThanToken) equalsGreaterThanToken;
        PTR(ConciseBody) body;
        never name;
    };

    struct TemplateLiteralLikeNode : LiteralLikeNode
    {
        string rawText;
        /* @internal */
        TokenFlags templateFlags;
    };

    struct RegularExpressionLiteral : LiteralExpression
    {
        // kind: SyntaxKind::RegularExpressionLiteral;
    };

    struct NoSubstitutionTemplateLiteral : LiteralExpression, TemplateLiteralLikeNode, Declaration
    {
        // kind: SyntaxKind::NoSubstitutionTemplateLiteral;
        /* @internal */
        TokenFlags templateFlags;
    };

    struct NumericLiteral : LiteralExpression, Declaration
    {
        // kind: SyntaxKind::NumericLiteral;
        /* @internal */
        TokenFlags numericLiteralFlags;
    };

    struct BigIntLiteral : LiteralExpression
    {
        // kind: SyntaxKind::BigIntLiteral;
    };

    struct PseudoBigInt 
    {
        boolean negative;
        string base10Value;
    };

    struct TemplateHead : TemplateLiteralLikeNode
    {
    };

    struct TemplateMiddle : TemplateLiteralLikeNode
    {
    };

    struct TemplateTail : TemplateLiteralLikeNode
    {
    };

    struct TemplateExpression : PrimaryExpression
    {
        // kind: SyntaxKind::TemplateExpression;
        PTR(TemplateHead) head;
        NodeArray<PTR(TemplateSpan)> templateSpans;
    };

    // Each of these corresponds to a substitution expression and a template literal, in that order.
    // The template literal must have kind TemplateMiddleLiteral or TemplateTailLiteral.
    struct TemplateSpan : virtual Node
    {
        // kind: SyntaxKind::TemplateSpan;
        PTR(TemplateExpression) parent;
        PTR(Expression) expression;
        PTR(Node) /**TemplateMiddle | TemplateTail*/ literal;
    };

    struct ParenthesizedExpression : PrimaryExpression, JSDocContainer
    {
        // kind: SyntaxKind::ParenthesizedExpression;
        PTR(Expression) expression;
    };

    struct ArrayLiteralExpression : PrimaryExpression
    {
        // kind: SyntaxKind::ArrayLiteralExpression;
        NodeArray<PTR(Expression)> elements;
        /* @internal */
        boolean multiLine;
    };

    struct SpreadElement : Expression
    {
        // kind: SyntaxKind::SpreadElement;
        PTR(Node) /**ArrayLiteralExpression | CallExpression | NewExpression*/ parent;
        PTR(Expression) expression;
    };

    /**
 * This interface is a base interface for ObjectLiteralExpression and JSXAttributes to extend from. JSXAttributes is similar to
 * ObjectLiteralExpression in that it contains array of properties; however, JSXAttributes' properties can only be
 * JSXAttribute or JSXSpreadAttribute. ObjectLiteralExpression, on the other hand, can only have properties of type
 * ObjectLiteralElement (e.g. PropertyAssignment, ShorthandPropertyAssignment etc.)
 */
    template <typename T /*: ObjectLiteralElement*/>
    struct ObjectLiteralExpressionBase : PrimaryExpression, Declaration
    {
        NodeArray<PTR(T)> properties;
    };

    // An ObjectLiteralExpression is the declaration node for an anonymous symbol.
    struct ObjectLiteralExpression : ObjectLiteralExpressionBase<ObjectLiteralElementLike>
    {
        // kind: SyntaxKind::ObjectLiteralExpression;
        /* @internal */
        boolean multiLine;
    };

    struct PropertyAccessExpression : MemberExpression, NamedDeclaration
    {
        // kind: SyntaxKind::PropertyAccessExpression;
        PTR(LeftHandSideExpression) expression;
        PTR(QuestionDotToken) questionDotToken;
        PTR(MemberName) name;
    };

    /*@internal*/
    struct PrivateIdentifierPropertyAccessExpression : PropertyAccessExpression
    {
        PTR(PrivateIdentifier) name;
    };

    struct PropertyAccessChain : PropertyAccessExpression
    {
        any _optionalChainBrand;
        PTR(MemberName) name;
    };

    /* @internal */
    struct PropertyAccessChainRoot : PropertyAccessChain
    {
        PTR(QuestionDotToken) questionDotToken;
    };

    struct SuperPropertyAccessExpression : PropertyAccessExpression
    {
        PTR(SuperExpression) expression;
    };

    /** Brand for a PropertyAccessExpression which, like a QualifiedName, consists of a sequence of identifiers separated by dots. */
    struct PropertyAccessEntityNameExpression : PropertyAccessExpression
    {
        any _propertyAccessExpressionLikeQualifiedNameBrand;
        PTR(EntityNameExpression) expression;
        PTR(Identifier) name;
    };

    struct ElementAccessExpression : MemberExpression
    {
        // kind: SyntaxKind::ElementAccessExpression;
        PTR(LeftHandSideExpression) expression;
        PTR(QuestionDotToken) questionDotToken;
        PTR(Expression) argumentExpression;
    };

    /* @internal */
    struct LateBoundElementAccessExpression : ElementAccessExpression
    {
        PTR(EntityNameExpression) argumentExpression;
    };

    struct ElementAccessChain : ElementAccessExpression
    {
        any _optionalChainBrand;
    };

    /* @internal */
    struct ElementAccessChainRoot : ElementAccessChain
    {
        PTR(QuestionDotToken) questionDotToken;
    };

    struct SuperElementAccessExpression : ElementAccessExpression
    {
        PTR(SuperExpression) expression;
    };

    struct CallExpression : LeftHandSideExpression, Declaration
    {
        // kind: SyntaxKind::CallExpression;
        PTR(LeftHandSideExpression) expression;
        PTR(QuestionDotToken) questionDotToken;
        NodeArray<PTR(TypeNode)> typeArguments;
        NodeArray<PTR(Expression)> arguments;
    };

    struct CallChain : CallExpression
    {
        any _optionalChainBrand;
    };

    /* @internal */
    struct CallChainRoot : CallChain
    {
        PTR(QuestionDotToken) questionDotToken;
    };

    /** @internal */
    struct BindableObjectDefinePropertyCall : CallExpression
    {
        NodeArray<PTR(Node)> arguments;
    };

    /** @internal */
    struct LiteralLikeElementAccessExpression : ElementAccessExpression, Declaration
    {
        PTR(Node) /**StringLiteralLike | NumericLiteral*/ argumentExpression;
    };

    /** @internal */
    struct BindableStaticElementAccessExpression : LiteralLikeElementAccessExpression
    {
        PTR(BindableStaticNameExpression) expression;
    };

    /** @internal */
    struct BindableElementAccessExpression : ElementAccessExpression
    {
        PTR(BindableStaticNameExpression) expression;
    };

    /** @internal */
    struct BindableStaticPropertyAssignmentExpression : BinaryExpression
    {
        PTR(BindableStaticAccessExpression) left;
    };

    /** @internal */
    struct BindablePropertyAssignmentExpression : BinaryExpression
    {
        PTR(BindableAccessExpression) left;
    };

    // see: https://tc39.github.io/ecma262/#prod-SuperCall
    struct SuperCall : CallExpression
    {
        PTR(SuperExpression) expression;
    };

    struct ImportCall : CallExpression
    {
        PTR(ImportExpression) expression;
    };

    struct ExpressionWithTypeArguments : NodeWithTypeArguments
    {
        // kind: SyntaxKind::ExpressionWithTypeArguments;
        PTR(Node) /**HeritageClause | JSDocAugmentsTag | JSDocImplementsTag*/ parent;
        PTR(LeftHandSideExpression) expression;
    };

    struct NewExpression : PrimaryExpression, Declaration
    {
        // kind: SyntaxKind::NewExpression;
        PTR(LeftHandSideExpression) expression;
        NodeArray<PTR(TypeNode)> typeArguments;
        NodeArray<PTR(Expression)> arguments;
    };

    struct TaggedTemplateExpression : MemberExpression
    {
        // kind: SyntaxKind::TaggedTemplateExpression;
        PTR(LeftHandSideExpression) tag;
        NodeArray<PTR(TypeNode)> typeArguments;
        PTR(TemplateLiteral) _template;
        /*@internal*/ PTR(QuestionDotToken) questionDotToken; // NOTE: Invalid syntax, only used to report a grammar error.
    };

    struct AsExpression : Expression
    {
        // kind: SyntaxKind::AsExpression;
        PTR(Expression) expression;
        PTR(TypeNode) type;
    };

    struct TypeAssertion : UnaryExpression
    {
        // kind: SyntaxKind::TypeAssertionExpression;
        PTR(TypeNode) type;
        PTR(UnaryExpression) expression;
    };

    struct NonNullExpression : LeftHandSideExpression
    {
        // kind: SyntaxKind::NonNullExpression;
        PTR(Expression) expression;
    };

    struct NonNullChain : NonNullExpression
    {
        any _optionalChainBrand;
    };

    // NOTE: MetaProperty is really a MemberExpression, but we consider it a PrimaryExpression
    //       for the same reasons we treat NewExpression as a PrimaryExpression.
    struct MetaProperty : PrimaryExpression
    {
        // kind: SyntaxKind::MetaProperty;
        SyntaxKind keywordToken;
        PTR(Identifier) name;
    };

    /* @internal */
    struct ImportMetaProperty : MetaProperty
    {
        SyntaxKind keywordToken;
        struct _name : Identifier
        {
            string escapedText;
        } name;
    };

    /// A JSX expression of the form <TagName attrs>...</TagName>
    struct JsxElement : PrimaryExpression
    {
        // kind: SyntaxKind::JsxElement;
        PTR(JsxOpeningElement) openingElement;
        NodeArray<PTR(JsxChild)> children;
        PTR(JsxClosingElement) closingElement;
    };

    struct JsxTagNamePropertyAccess : PropertyAccessExpression
    {
        PTR(JsxTagNameExpression) expression;
    };

    struct JsxAttributes : ObjectLiteralExpressionBase<JsxAttributeLike>
    {
        // kind: SyntaxKind::JsxAttributes;
        PTR(JsxOpeningLikeElement) parent;
    };

    /// The opening element of a <Tag>...</Tag> JsxElement
    struct JsxOpeningElement : Expression
    {
        // kind: SyntaxKind::JsxOpeningElement;
        PTR(JsxElement) parent;
        PTR(JsxTagNameExpression) tagName;
        NodeArray<PTR(TypeNode)> typeArguments;
        PTR(JsxAttributes) attributes;
    };

    /// A JSX expression of the form <TagName attrs />
    struct JsxSelfClosingElement : PrimaryExpression
    {
        // kind: SyntaxKind::JsxSelfClosingElement;
        PTR(JsxTagNameExpression) tagName;
        NodeArray<PTR(TypeNode)> typeArguments;
        PTR(JsxAttributes) attributes;
    };

    /// A JSX expression of the form <>...</>
    struct JsxFragment : PrimaryExpression
    {
        // kind: SyntaxKind::JsxFragment;
        PTR(JsxOpeningFragment) openingFragment;
        NodeArray<PTR(JsxChild)> children;
        PTR(JsxClosingFragment) closingFragment;
    };

    /// The opening element of a <>...</> JsxFragment
    struct JsxOpeningFragment : Expression
    {
        // kind: SyntaxKind::JsxOpeningFragment;
        PTR(JsxFragment) parent;
    };

    /// The closing element of a <>...</> JsxFragment
    struct JsxClosingFragment : Expression
    {
        // kind: SyntaxKind::JsxClosingFragment;
        PTR(JsxFragment) parent;
    };

    struct JsxAttribute : ObjectLiteralElement
    {
        // kind: SyntaxKind::JsxAttribute;
        PTR(JsxAttributes) parent;
        PTR(Identifier) name;
        /// JSX attribute initializers are optional; <X y /> is sugar for <X y={true} />
        PTR(Node) /**StringLiteral | JsxExpression*/ initializer;
    };

    struct JsxSpreadAttribute : ObjectLiteralElement
    {
        // kind: SyntaxKind::JsxSpreadAttribute;
        PTR(JsxAttributes) parent;
        PTR(Expression) expression;
    };

    struct JsxClosingElement : virtual Node
    {
        // kind: SyntaxKind::JsxClosingElement;
        PTR(JsxElement) parent;
        PTR(JsxTagNameExpression) tagName;
    };

    struct JsxExpression : Expression
    {
        // kind: SyntaxKind::JsxExpression;
        PTR(Node) /**JsxElement | JsxAttributeLike*/ parent;
        Token<SyntaxKind::DotDotDotToken> dotDotDotToken;
        PTR(Expression) expression;
    };

    struct JsxText : LiteralLikeNode
    {
        // kind: SyntaxKind::JsxText;
        PTR(JsxElement) parent;
        boolean containsOnlyTriviaWhiteSpaces;
    };

    // Represents a statement that is elided as part of a transformation to emit comments on a
    // not-emitted node.
    struct NotEmittedStatement : Statement
    {
        // kind: SyntaxKind::NotEmittedStatement;
    };

    /**
 * Marks the end of transformed declaration to properly emit exports.
 */
    /* @internal */
    struct EndOfDeclarationMarker : Statement
    {
        // kind: SyntaxKind::EndOfDeclarationMarker;
    };

    /**
 * A list of comma-separated expressions. This node is only created by transformations.
 */
    struct CommaListExpression : Expression
    {
        // kind: SyntaxKind::CommaListExpression;
        NodeArray<PTR(Expression)> elements;
    };

    /**
 * Marks the beginning of a merged transformed declaration.
 */
    /* @internal */
    struct MergeDeclarationMarker : Statement
    {
        // kind: SyntaxKind::MergeDeclarationMarker;
    };

    /* @internal */
    struct SyntheticReferenceExpression : LeftHandSideExpression
    {
        // kind: SyntaxKind::SyntheticReferenceExpression;
        PTR(Expression) expression;
        PTR(Expression) thisArg;
    };

    struct EmptyStatement : Statement
    {
        // kind: SyntaxKind::EmptyStatement;
    };

    struct DebuggerStatement : Statement
    {
        // kind: SyntaxKind::DebuggerStatement;
    };

    struct MissingDeclaration : DeclarationStatement
    {
        /*@internal*/ NodeArray<PTR(Decorator)> decorators; // Present for use with reporting a grammar error
        /*@internal*/ ModifiersArray modifiers;        // Present for use with reporting a grammar error
        // kind: SyntaxKind::MissingDeclaration;
        PTR(Identifier) name;
    };

    struct Block : Statement
    {
        // kind: SyntaxKind::Block;
        NodeArray<PTR(Statement)> statements;
        /*@internal*/ boolean multiLine;
    };

    struct VariableStatement : Statement, JSDocContainer
    {
        /* @internal*/ NodeArray<PTR(Decorator)> decorators; // Present for use with reporting a grammar error
        // kind: SyntaxKind::VariableStatement;
        PTR(VariableDeclarationList) declarationList;
    };

    struct ExpressionStatement : Statement, JSDocContainer
    {
        // kind: SyntaxKind::ExpressionStatement;
        PTR(Expression) expression;
    };

    /* @internal */
    struct PrologueDirective : ExpressionStatement
    {
        PTR(StringLiteral) expression;
    };

    struct IfStatement : Statement
    {
        // kind: SyntaxKind::IfStatement;
        PTR(Expression) expression;
        PTR(Statement) thenStatement;
        PTR(Statement) elseStatement;
    };

    struct IterationStatement : Statement
    {
        PTR(Statement) statement;
    };

    struct DoStatement : IterationStatement
    {
        // kind: SyntaxKind::DoStatement;
        PTR(Expression) expression;
    };

    struct WhileStatement : IterationStatement
    {
        // kind: SyntaxKind::WhileStatement;
        PTR(Expression) expression;
    };

    struct ForStatement : IterationStatement
    {
        // kind: SyntaxKind::ForStatement;
        PTR(ForInitializer) initializer;
        PTR(Expression) condition;
        PTR(Expression) incrementor;
    };

    struct ForInStatement : IterationStatement
    {
        // kind: SyntaxKind::ForInStatement;
        PTR(ForInitializer) initializer;
        PTR(Expression) expression;
    };

    struct ForOfStatement : IterationStatement
    {
        // kind: SyntaxKind::ForOfStatement;
        PTR(AwaitKeywordToken) awaitModifier;
        PTR(ForInitializer) initializer;
        PTR(Expression) expression;
    };

    struct BreakStatement : Statement
    {
        // kind: SyntaxKind::BreakStatement;
        PTR(Identifier) label;
    };

    struct ContinueStatement : Statement
    {
        // kind: SyntaxKind::ContinueStatement;
        PTR(Identifier) label;
    };

    struct ReturnStatement : Statement
    {
        // kind: SyntaxKind::ReturnStatement;
        PTR(Expression) expression;
    };

    struct WithStatement : Statement
    {
        // kind: SyntaxKind::WithStatement;
        PTR(Expression) expression;
        PTR(Statement) statement;
    };

    struct SwitchStatement : Statement
    {
        // kind: SyntaxKind::SwitchStatement;
        PTR(Expression) expression;
        PTR(CaseBlock) caseBlock;
        boolean possiblyExhaustive; // initialized by binding
    };

    struct CaseBlock : virtual Node
    {
        // kind: SyntaxKind::CaseBlock;
        PTR(SwitchStatement) parent;
        NodeArray<PTR(CaseOrDefaultClause)> clauses;
    };

    struct CaseClause : virtual Node
    {
        // kind: SyntaxKind::CaseClause;
        PTR(CaseBlock) parent;
        PTR(Expression) expression;
        NodeArray<PTR(Statement)> statements;
        ///* @internal */ PTR(FlowNode) fallthroughFlowNode;
    };

    struct DefaultClause : virtual Node
    {
        // kind: SyntaxKind::DefaultClause;
        PTR(CaseBlock) parent;
        NodeArray<PTR(Statement)> statements;
        ///* @internal */ PTR(FlowNode) fallthroughFlowNode;
    };

    struct LabeledStatement : Statement, JSDocContainer
    {
        // kind: SyntaxKind::LabeledStatement;
        PTR(Identifier) label;
        PTR(Statement) statement;
    };

    struct ThrowStatement : Statement
    {
        // kind: SyntaxKind::ThrowStatement;
        PTR(Expression) expression;
    };

    struct TryStatement : Statement
    {
        // kind: SyntaxKind::TryStatement;
        PTR(Block) tryBlock;
        PTR(CatchClause) catchClause;
        PTR(Block) finallyBlock;
    };

    struct CatchClause : virtual Node
    {
        // kind: SyntaxKind::CatchClause;
        PTR(TryStatement) parent;
        PTR(VariableDeclaration) variableDeclaration;
        PTR(Block) block;
    };

    struct ClassLikeDeclarationBase : NamedDeclaration, JSDocContainer
    {
        // kind: SyntaxKind::ClassDeclaration | SyntaxKind::ClassExpression;
        PTR(Identifier) name;
        NodeArray<PTR(TypeParameterDeclaration)> typeParameters;
        NodeArray<PTR(HeritageClause)> heritageClauses;
        NodeArray<PTR(ClassElement)> members;
    };

    struct ClassDeclaration : ClassLikeDeclarationBase, DeclarationStatement
    {
        // kind: SyntaxKind::ClassDeclaration;
        /** May be undefined in `export default class { ... }`. */
        PTR(Identifier) name;
    };

    struct ClassExpression : ClassLikeDeclarationBase, PrimaryExpression
    {
        // kind: SyntaxKind::ClassExpression;
    };

    struct InterfaceDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::InterfaceDeclaration;
        PTR(Identifier) name;
        NodeArray<PTR(TypeParameterDeclaration)> typeParameters;
        NodeArray<PTR(HeritageClause)> heritageClauses;
        NodeArray<PTR(TypeElement)> members;
    };

    struct HeritageClause : virtual Node
    {
        // kind: SyntaxKind::HeritageClause;
        NodeRef /*InterfaceDeclaration | ClassLikeDeclaration*/ parent;
        SyntaxKind token;
        NodeArray<PTR(ExpressionWithTypeArguments)> types;
    };

    struct TypeAliasDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::TypeAliasDeclaration;
        PTR(Identifier) name;
        NodeArray<PTR(TypeParameterDeclaration)> typeParameters;
        PTR(TypeNode) type;
    };

    struct EnumMember : NamedDeclaration, JSDocContainer
    {
        // kind: SyntaxKind::EnumMember;
        PTR(EnumDeclaration) parent;
        // This does include ComputedPropertyName, but the parser will give an error
        // if it parses a ComputedPropertyName in an EnumMember
        PTR(PropertyName) name;
        PTR(Expression) initializer;
    };

    struct EnumDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::EnumDeclaration;
        PTR(Identifier) name;
        NodeArray<PTR(EnumMember)> members;
    };

    struct ModuleDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::ModuleDeclaration;
        PTR(Node) /**ModuleBody | SourceFile*/ parent;
        PTR(ModuleName) name;
        PTR(Node) /**ModuleBody | JSDocNamespaceDeclaration*/ body;
    };

    /* @internal */
    struct AmbientModuleDeclaration : ModuleDeclaration
    {
        PTR(ModuleBlock) body;
    };

    struct NamespaceDeclaration : ModuleDeclaration
    {
        PTR(Identifier) name;
        PTR(NamespaceBody) body;
    };

    struct JSDocNamespaceDeclaration : ModuleDeclaration
    {
        PTR(Identifier) name;
        PTR(JSDocNamespaceBody) body;
    };

    struct ModuleBlock : Statement
    {
        // kind: SyntaxKind::ModuleBlock;
        PTR(ModuleDeclaration) parent;
        NodeArray<PTR(Statement)> statements;
    };

    /**
 * One of:
 * - import x = require("mod");
 * - import x = M.x;
 */
    struct ImportEqualsDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::ImportEqualsDeclaration;
        PTR(Node) /**SourceFile | ModuleBlock*/ parent;
        PTR(Identifier) name;
        boolean isTypeOnly;

        // 'EntityName' for an internal module reference, 'ExternalModuleReference' for an external
        // module reference.
        PTR(ModuleReference) moduleReference;
    };

    struct ExternalModuleReference : virtual Node
    {
        // kind: SyntaxKind::ExternalModuleReference;
        PTR(ImportEqualsDeclaration) parent;
        PTR(Expression) expression;
    };

    // In case of:
    // import "mod"  => importClause = undefined, moduleSpecifier = "mod"
    // In rest of the cases, module specifier is string literal corresponding to module
    // ImportClause information is shown at its declaration below.
    struct ImportDeclaration : Statement, JSDocContainer
    {
        // kind: SyntaxKind::ImportDeclaration;
        PTR(Node) /**SourceFile | ModuleBlock*/ parent;
        PTR(ImportClause) importClause;
        /** If this is not a StringLiteral it will be a grammar error. */
        PTR(Expression) moduleSpecifier;
    };

    // In case of:
    // import d from "mod" => name = d, namedBinding = undefined
    // import * as ns from "mod" => name = undefined, namedBinding: NamespaceImport = { name: ns }
    // import d, * as ns from "mod" => name = d, namedBinding: NamespaceImport = { name: ns }
    // import { a, b as x } from "mod" => name = undefined, namedBinding: NamedImports = { elements: [{ name: a }, { name: x, propertyName: b}]}
    // import d, { a, b as x } from "mod" => name = d, namedBinding: NamedImports = { elements: [{ name: a }, { name: x, propertyName: b}]}
    struct ImportClause : NamedDeclaration
    {
        // kind: SyntaxKind::ImportClause;
        PTR(ImportDeclaration) parent;
        boolean isTypeOnly;
        PTR(Identifier) name; // Default binding
        PTR(NamedImportBindings) namedBindings;
    };

    struct NamespaceImport : NamedDeclaration
    {
        // kind: SyntaxKind::NamespaceImport;
        PTR(ImportClause) parent;
        PTR(Identifier) name;
    };

    struct NamespaceExport : NamedDeclaration
    {
        // kind: SyntaxKind::NamespaceExport;
        PTR(ExportDeclaration) parent;
        PTR(Identifier) name;
    };

    struct NamespaceExportDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::NamespaceExportDeclaration name;
        PTR(Identifier) name;
        /* @internal */ NodeArray<PTR(Decorator)> decorators; // Present for use with reporting a grammar error
        /* @internal */ ModifiersArray modifiers;        // Present for use with reporting a grammar error
    };

    struct ExportDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::ExportDeclaration;
        PTR(Node) /**SourceFile | ModuleBlock*/ parent;
        boolean isTypeOnly;
        /** Will not be assigned in the case of `export * from "foo";` */
        PTR(NamedExportBindings) exportClause;
        /** If this is not a StringLiteral it will be a grammar error. */
        PTR(Expression) moduleSpecifier;
    };

    struct NamedImports : virtual Node
    {
        // kind: SyntaxKind::NamedImports;
        PTR(ImportClause) parent;
        NodeArray<PTR(ImportSpecifier)> elements;
    };

    struct NamedExports : virtual Node
    {
        // kind: SyntaxKind::NamedExports;
        PTR(ExportDeclaration) parent;
        NodeArray<PTR(ExportSpecifier)> elements;
    };

    struct ImportSpecifier : NamedDeclaration
    {
        // kind: SyntaxKind::ImportSpecifier;
        PTR(NamedImports) parent;
        PTR(Identifier) propertyName; // Name preceding "as" keyword (or undefined when "as" is absent)
        PTR(Identifier) name;         // Declared name
    };

    struct ExportSpecifier : NamedDeclaration
    {
        // kind: SyntaxKind::ExportSpecifier;
        PTR(NamedExports) parent;
        PTR(Identifier) propertyName; // Name preceding "as" keyword (or undefined when "as" is absent)
        PTR(Identifier) name;         // Declared name
    };

    /**
 * This is either an `export =` or an `export default` declaration.
 * Unless `isExportEquals` is set, this node was parsed as an `export default`.
 */
    struct ExportAssignment : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::ExportAssignment;
        PTR(SourceFile) parent;
        boolean isExportEquals;
        PTR(Expression) expression;
    };

    struct FileReference : TextRange
    {
        string fileName;
    };

    struct CheckJsDirective : TextRange
    {
        boolean enabled;
    };

    using CommentKind = SyntaxKind; //SyntaxKind::SingleLineCommentTrivia | SyntaxKind::MultiLineCommentTrivia;

    struct CommentRange : TextRange
    {
        CommentRange(number pos, number end, boolean hasTrailingNewLine, CommentKind kind) : TextRange{pos, end}, hasTrailingNewLine(hasTrailingNewLine), kind(kind) {}

        boolean hasTrailingNewLine;
        CommentKind kind;
    };

    struct SynthesizedComment : CommentRange
    {
        string text;
        number pos;
        number end;
        boolean hasLeadingNewline;
    };

    // represents a top level: { type } expression in a JSDoc comment.
    struct JSDocTypeExpression : TypeNode
    {
        // kind: SyntaxKind::JSDocTypeExpression;
        PTR(TypeNode) type;
    };

    struct JSDocNameReference : virtual Node
    {
        // kind: SyntaxKind::JSDocNameReference;
        PTR(EntityName) name;
    };

    struct JSDocType : TypeNode
    {
        any _jsDocTypeBrand;
    };

    struct JSDocAllType : JSDocType
    {
        // kind: SyntaxKind::JSDocAllType;
    };

    struct JSDocUnknownType : JSDocType
    {
        // kind: SyntaxKind::JSDocUnknownType;
    };

    struct JSDocNonNullableType : JSDocType
    {
        // kind: SyntaxKind::JSDocNonNullableType;
        PTR(TypeNode) type;
    };

    struct JSDocNullableType : JSDocType
    {
        // kind: SyntaxKind::JSDocNullableType;
        PTR(TypeNode) type;
    };

    struct JSDocOptionalType : JSDocType
    {
        // kind: SyntaxKind::JSDocOptionalType;
        PTR(TypeNode) type;
    };

    struct JSDocFunctionType : JSDocType, SignatureDeclarationBaseNoNode
    {
        // kind: SyntaxKind::JSDocFunctionType;
    };

    struct JSDocVariadicType : JSDocType
    {
        // kind: SyntaxKind::JSDocVariadicType;
        PTR(TypeNode) type;
    };

    struct JSDocNamepathType : JSDocType
    {
        // kind: SyntaxKind::JSDocNamepathType;
        PTR(TypeNode) type;
    };

    struct JSDoc : virtual Node
    {
        // kind: SyntaxKind::JSDocComment;
        PTR(HasJSDoc) parent;
        NodeArray<PTR(JSDocTag)> tags;
        string comment;
    };

    struct JSDocTag : virtual Node
    {
        PTR(Node) /**JSDoc | JSDocTypeLiteral*/ parent;
        PTR(Identifier) tagName;
        string comment;
    };

    struct JSDocUnknownTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocTag;
    };

    /**
 * Note that `@extends` is a synonym of `@augments`.
 * Both tags are represented by this interface.
 */
    struct JSDocAugmentsTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocAugmentsTag;
        struct _classArg : ExpressionWithTypeArguments
        {
            PTR(Node) /**Identifier | PropertyAccessEntityNameExpression*/ expression;
        } _class;
    };

    struct JSDocImplementsTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocImplementsTag;
        struct _classArg : ExpressionWithTypeArguments
        {
            PTR(Node) /**Identifier | PropertyAccessEntityNameExpression*/ expression;
        } _class;
    };

    struct JSDocAuthorTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocAuthorTag;
    };

    struct JSDocDeprecatedTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocDeprecatedTag;
    };

    struct JSDocClassTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocClassTag;
    };

    struct JSDocPublicTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocPublicTag;
    };

    struct JSDocPrivateTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocPrivateTag;
    };

    struct JSDocProtectedTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocProtectedTag;
    };

    struct JSDocReadonlyTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocReadonlyTag;
    };

    struct JSDocEnumTag : JSDocTag, Declaration
    {
        // kind: SyntaxKind::JSDocEnumTag;
        PTR(JSDoc) parent;
        PTR(JSDocTypeExpression) typeExpression;
    };

    struct JSDocThisTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocThisTag;
        PTR(JSDocTypeExpression) typeExpression;
    };

    struct JSDocTemplateTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocTemplateTag;
        PTR(Node) /**JSDocTypeExpression | undefined*/ constraint;
        NodeArray<PTR(TypeParameterDeclaration)> typeParameters;
    };

    struct JSDocSeeTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocSeeTag;
        PTR(JSDocNameReference) name;
    };

    struct JSDocReturnTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocReturnTag;
        PTR(JSDocTypeExpression) typeExpression;
    };

    struct JSDocTypeTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocTypeTag;
        PTR(JSDocTypeExpression) typeExpression;
    };

    struct JSDocTypedefTag : JSDocTag, NamedDeclaration
    {
        // kind: SyntaxKind::JSDocTypedefTag;
        PTR(JSDoc) parent;
        PTR(Node) /**JSDocNamespaceDeclaration | Identifier*/ fullName;
        PTR(Identifier) name;
        PTR(Node) /**JSDocTypeExpression | JSDocTypeLiteral*/ typeExpression;
    };

    struct JSDocCallbackTag : JSDocTag, NamedDeclaration
    {
        // kind: SyntaxKind::JSDocCallbackTag;
        PTR(JSDoc) parent;
        PTR(Node) /**JSDocNamespaceDeclaration | Identifier*/ fullName;
        PTR(Identifier) name;
        PTR(JSDocSignature) typeExpression;
    };

    struct JSDocSignature : JSDocType, Declaration
    {
        // kind: SyntaxKind::JSDocSignature;
        std::vector<JSDocTemplateTag> typeParameters;
        std::vector<JSDocParameterTag> parameters;
        PTR(Node) /**JSDocReturnTag | undefined*/ type;
    };

    struct JSDocPropertyLikeTag : JSDocTag, Declaration
    {
        PTR(JSDoc) parent;
        PTR(EntityName) name;
        PTR(JSDocTypeExpression) typeExpression;
        /** Whether the property name came before the type -- non-standard for JSDoc, but Typescript-like */
        boolean isNameFirst;
        boolean isBracketed;
    };

    struct JSDocPropertyTag : JSDocPropertyLikeTag
    {
        // kind: SyntaxKind::JSDocPropertyTag;
    };

    struct JSDocParameterTag : JSDocPropertyLikeTag
    {
        // kind: SyntaxKind::JSDocParameterTag;
    };

    struct JSDocTypeLiteral : JSDocType
    {
        // kind: SyntaxKind::JSDocTypeLiteral;
        std::vector<JSDocPropertyLikeTag> jsDocPropertyTags;
        /** If true, then this type literal represents an *array* of its type. */
        boolean isArrayType;
    };

    using Path = string;

    struct AmdDependency
    {
        string path;
        string name;
    };

    /* @internal */
    struct CommentDirective
    {
        CommentDirective(number pos, number end, CommentDirectiveType type) : range{pos, end}, type(type) {}

        TextRange range;
        CommentDirectiveType type;
    };

    /*@internal*/
    using ExportedModulesFromDeclarationEmit = std::vector<Symbol>;

    /* @internal */
    /**
 * Subset of properties from SourceFile that are used in multiple utility functions
 */
    struct SourceFileLike
    {
        string text;
        std::vector<number> lineMap;
        /* @internal */
        std::function<number(number, number, boolean allowEdits)> getPositionOfLineAndCharacter;
    };

    struct RedirectInfo
    {
        /** Source file this redirects to. */
        PTR(SourceFile) redirectTarget;
        /**
     * Source file for the duplicate package. This will not be used by the Program,
     * but we need to keep this around so we can watch for changes in underlying.
     */
        PTR(SourceFile) unredirected;
    };

    struct DiagnosticMessage
    {
        string key;
        DiagnosticCategory category;
        number code;
        string message;
        std::vector<string> reportsUnnecessary;
        std::vector<string> reportsDeprecated;
        /* @internal */
        boolean elidedInCompatabilityPyramid;

        DiagnosticMessage() = default;
        DiagnosticMessage(DiagnosticMessageStore &item) : code(item.code), category(item.category), key(item.label), message(item.message) {}
    };

    struct DiagnosticMessageChain
    {
        string messageText;
        DiagnosticCategory category;
        number code;
        std::vector<DiagnosticMessageChain> next;
    };

    struct DiagnosticRelatedInformation
    {
        DiagnosticCategory category;
        number code;
        PTR(SourceFile) file;
        number start;
        number length;
        string messageText;
        PTR(DiagnosticMessageChain) messageChain;
    };

    struct Diagnostic : DiagnosticRelatedInformation
    {
        /** May store more in future. For now, this will simply be `true` to indicate when a diagnostic is an unused-identifier diagnostic. */
        std::vector<string> reportsUnnecessary;
        std::vector<string> reportsDeprecated;
        string source;
        std::vector<DiagnosticRelatedInformation> relatedInformation;
        /* @internal */ string /*keyof CompilerOptions*/ skippedOn;
    };

    struct DiagnosticWithLocation : Diagnostic
    {
        PTR(SourceFile) file;
        number start;
        number length;
    };

    struct DiagnosticWithDetachedLocation : Diagnostic
    {
        undefined_t file;
        string fileName;
        number start;
        number length;
    };

    struct ResolvedModule
    {
        /** Path of the file the module was resolved to. */
        string resolvedFileName;
        /** True if `resolvedFileName` comes from `node_modules`. */
        boolean isExternalLibraryImport;
    };

    struct PackageId
    {
        /**
     * Name of the package.
     * Should not include `@types`.
     * If accessing a non-index file, this should include its name e.g. "foo/bar".
     */
        string name;
        /**
     * Name of a submodule within this package.
     * May be "".
     */
        string subModuleName;
        /** Version of the package, e.g. "1.2.3" */
        string version;
    };

    struct ResolvedModuleFull : ResolvedModule
    {
        /* @internal */
        string originalPath;
        /**
     * Extension of resolvedFileName. This must match what's at the end of resolvedFileName.
     * This is optional for backwards-compatibility, but will be added if not provided.
     */
        string /*Extension*/ extension;
        PTR(PackageId) packageId;
    };

    struct ResolvedTypeReferenceDirective
    {
        // True if the type declaration file was found in a primary lookup location
        boolean primary;
        // The location of the .d.ts file we located, or undefined if resolution failed
        string resolvedFileName;
        PTR(PackageId) packageId;
        /** True if `resolvedFileName` comes from `node_modules`. */
        boolean isExternalLibraryImport;
    };

    struct PatternAmbientModule
    {
        PTR(Node) pattern;
        PTR(Symbol) symbol;
    };

    struct SourceFile : Declaration
    {
        SourceFile() = default;
        SourceFile(SyntaxKind kind_, number pos_, number end_)
        {
            kind = kind_;
            pos = pos_;
            end = end_;
        }

        // kind: SyntaxKind::SourceFile;
        NodeArray<PTR(Statement)> statements;
        Token<SyntaxKind::EndOfFileToken> endOfFileToken;

        string fileName;
        /* @internal */ Path path;
        string text;
        /** Resolved path can be different from path property,
     * when file is included through project reference is mapped to its output instead of source
     * in that case resolvedPath = path to output file
     * path = input file's path
     */
        /* @internal */ Path resolvedPath;
        /** Original file name that can be different from fileName,
     * when file is included through project reference is mapped to its output instead of source
     * in that case originalFileName = name of input file
     * fileName = output file's name
     */
        /* @internal */ string originalFileName;

        /**
     * If two source files are for the same version of the same package, one will redirect to the other.
     * (See `createRedirectSourceFile` in program.ts.)
     * The redirect will have this set. The redirected-to source file will be in `redirectTargetsMap`.
     */
        /* @internal */ PTR(RedirectInfo) redirectInfo;

        std::vector<AmdDependency> amdDependencies;
        string moduleName;
        std::vector<FileReference> referencedFiles;
        std::vector<FileReference> typeReferenceDirectives;
        std::vector<FileReference> libReferenceDirectives;
        LanguageVariant languageVariant;
        boolean isDeclarationFile;

        // this map is used by transpiler to supply alternative names for dependencies (i.e. in case of bundling)
        /* @internal */
        std::map<string, string> renamedDependencies;

        /**
     * lib.d.ts should have a reference comment like
     *
     *  /// <reference no-default-lib="true"/>
     *
     * If any other file has this comment, it signals not to include lib.d.ts
     * because this containing file is intended to act as a default library.
     */
        boolean hasNoDefaultLib;

        ScriptTarget languageVersion;
        /* @internal */ ScriptKind scriptKind;

        /**
     * The first "most obvious" node that makes a file an external module.
     * This is intended to be the first top-level import/export,
     * but could be arbitrarily nested (e.g. `import.meta`).
     */
        /* @internal */ PTR(Node) externalModuleIndicator;
        // The first node that causes this file to be a CommonJS module
        /* @internal */ PTR(Node) commonJsModuleIndicator;
        // JS identifier-declarations that are intended to merge with globals
        /* @internal */ SymbolTable jsGlobalAugmentations;

        /* @internal */ std::map<string, string> identifiers; // Map from a string to an interned string
        /* @internal */ number nodeCount;
        /* @internal */ number identifierCount;
        /* @internal */ number symbolCount;

        // File-level diagnostics reported by the parser (includes diagnostics about /// references
        // as well as code diagnostics).
        /* @internal */ std::vector<DiagnosticWithLocation> parseDiagnostics;

        // File-level diagnostics reported by the binder.
        /* @internal */ std::vector<DiagnosticWithLocation> bindDiagnostics;
        /* @internal */ std::vector<DiagnosticWithLocation> bindSuggestionDiagnostics;

        // File-level JSDoc diagnostics reported by the JSDoc parser
        /* @internal */ std::vector<DiagnosticWithLocation> jsDocDiagnostics;

        // Stores additional file-level diagnostics reported by the program
        /* @internal */ std::vector<DiagnosticWithLocation> additionalSyntacticDiagnostics;

        // Stores a line map for the file.
        // This field should never be used directly to obtain line map, use getLineMap function instead.
        /* @internal */ std::vector<number> lineMap;
        /* @internal */ std::set<string> classifiableNames;
        // Comments containing @ts-* directives, in order.
        /* @internal */ std::vector<CommentDirective> commentDirectives;
        // Stores a mapping 'external module reference text' -> 'resolved file name' | undefined
        // It is used to resolve module names in the checker.
        // Content of this field should never be used directly - use getResolvedModuleFileName/setResolvedModuleFileName functions instead
        /* @internal */ std::map<string, ResolvedModuleFull> resolvedModules;
        /* @internal */ std::map<string, ResolvedTypeReferenceDirective> resolvedTypeReferenceDirectiveNames;
        /* @internal */ std::vector<StringLiteralLike> imports;
        // Identifier only if `declare global`
        /* @internal */ NodeArray<PTR(Node)> moduleAugmentations;
        /* @internal */ std::vector<PatternAmbientModule> patternAmbientModules;
        /* @internal */ std::vector<string> ambientModuleNames;
        /* @internal */ PTR(CheckJsDirective) checkJsDirective;
        /* @internal */ string version;
        /* @internal */ std::map<string, string> pragmas;
        /* @internal */ string localJsxNamespace;
        /* @internal */ string localJsxFragmentNamespace;
        /* @internal */ PTR(EntityName) localJsxFactory;
        /* @internal */ PTR(EntityName) localJsxFragmentFactory;

        /* @internal */ ExportedModulesFromDeclarationEmit exportedModulesFromDeclarationEmit;
    };

    struct UnparsedSection : virtual Node 
    {
        PTR(UnparsedSource) parent;
        string data;
    };

    struct UnparsedPrologue : UnparsedSection 
    {
        PTR(UnparsedSource) parent;
        string data;
    };

    struct UnparsedPrepend : UnparsedSection 
    {
        PTR(UnparsedSource) parent;
        string data;
        std::vector<UnparsedTextLike> texts;
    };

    struct UnparsedTextLike : UnparsedSection 
    {
        PTR(UnparsedSource) parent;
    };

    struct UnparsedSyntheticReference : UnparsedSection 
    {
        PTR(UnparsedSource) parent;
        /*@internal*/ Node /*BundleFileHasNoDefaultLib | BundleFileReference*/ section;
    };  

    struct EmitHelperBase 
    {
        string name;                                                    // A unique name for this helper.
        boolean scoped;                                                 // Indicates whether the helper MUST be emitted in the current scope.
        string text;                                                    // ES3-compatible raw script text, or a function yielding such a string
        number priority;                                                // Helpers with a higher priority are emitted earlier than other helpers on the node.
        std::vector<EmitHelper> dependencies;
    };

    struct ScopedEmitHelper : EmitHelperBase 
    {
        boolean scoped;
    };

    struct UnscopedEmitHelper : EmitHelperBase 
    {
        boolean scoped;                                      // Indicates whether the helper MUST be emitted in the current scope.
        /* @internal */
        string importName;                                   // The name of the helper to use when importing via `--importHelpers`.
        string text;                                         // ES3-compatible raw script text, or a function yielding such a string
    };

    struct RawSourceMap 
    {
        number version;
        string file;
        string sourceRoot;
        std::vector<string> sources;
        string sourcesContent;
        string mappings;
        std::vector<string> names;
    };

    struct UnparsedSource : virtual Node 
    {
        string fileName;
        string text;
        std::vector<UnparsedPrologue> prologues;
        std::vector<UnscopedEmitHelper> helpers;

        // References and noDefaultLibAre Dts only
        std::vector<FileReference> referencedFiles;
        std::vector<string> typeReferenceDirectives;
        std::vector<FileReference> libReferenceDirectives;
        boolean hasNoDefaultLib;

        string sourceMapPath;
        string sourceMapText;
        std::vector<UnparsedSyntheticReference> syntheticReferences;
        std::vector<UnparsedSourceText> texts;
        /*@internal*/ boolean oldFileOfCurrentEmit;
        /*@internal*/ RawSourceMap parsedSourceMap;
        // Adding this to satisfy services, fix later
        /*@internal*/
        auto getLineAndCharacterOfPosition(number pos) -> LineAndCharacter;
    };

    struct BuildInfo 
    {
        // TODO: finish later
        //BundleBuildInfo bundle;
        //ProgramBuildInfo program;
        string version;
    };

    struct InputFiles : virtual Node 
    {
        string javascriptPath;
        string javascriptText;
        string javascriptMapPath;
        string javascriptMapText;
        string declarationPath;
        string declarationText;
        string declarationMapPath;
        string declarationMapText;
        /*@internal*/ string buildInfoPath;
        /*@internal*/ BuildInfo buildInfo;
        /*@internal*/ boolean oldFileOfCurrentEmit;
    };

    struct SyntaxList : virtual Node 
    {
        std::vector<Node> _children;
    };

    struct PropertyDescriptorAttributes 
    {
        PTR(Expression) enumerable;
        PTR(Expression) configurable;
        PTR(Expression) writable;
        PTR(Expression) value;
        PTR(Expression) get;
        PTR(Expression) set;
    };

    struct CallBinding
    {
        PTR(LeftHandSideExpression) target;
        PTR(Expression) thisArg;
    };

    struct NodeWithDiagnostics 
    { 
        PTR(Node) node;
        PTR(JSDoc) jsDoc;
        std::vector<Diagnostic> diagnostics;
    };

    struct JsonMinusNumericLiteral : PrefixUnaryExpression 
    {
        SyntaxKind _operator;
        PTR(NumericLiteral) operand;
    };

    struct JsonObjectExpressionStatement : ExpressionStatement 
    {
        PTR(JsonObjectExpression) expression;
    };

} // namespace data

#endif // NEW_PARSER_TYPES_H