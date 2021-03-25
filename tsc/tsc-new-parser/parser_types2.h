#ifndef NEW_PARSER_TYPES_H
#define NEW_PARSER_TYPES_H

#include "config.h"
#include "enums.h"
#include "scanner_enums.h"

#include <vector>
#include <map>
#include <set>

namespace ver2
{
    using NodeId = number;

    using any = char *;
    struct never
    {
    };

    using SymbolId = number;

    struct Symbol;
    using SymbolTable = std::map<string, Symbol>;

    using SymbolRef = std::reference_wrapper<Symbol>;

    /* @internal */
    using TypeId = number;

    struct Type;
    using TypeRef = std::reference_wrapper<Type>;

    struct Node;
    using NodeRef = std::reference_wrapper<Node>;

    struct SourceFile;
    using SourceFileRef = std::reference_wrapper<SourceFile>;

    struct Decorator;
    struct Modifier;
    
    struct Declaration;
    using DeclarationRef = std::reference_wrapper<Declaration>;

    struct JSDoc;
    struct JSDocTag;

    ///////////////////////////////////////////////////////////////////////

    struct TextRange
    {
        number pos;
        number end;
    };

    struct ReadonlyTextRange : TextRange
    {
    };

    template <typename T /*extends Node*/>
    struct ReadonlyArray : std::vector<T>
    {
    };

    template <typename T /*extends Node*/>
    struct NodeArray : ReadonlyArray<T>, ReadonlyTextRange
    {
        boolean hasTrailingComma;
        /* @internal */ TransformFlags transformFlags; // Flags for transforms, possibly undefined
    };

    using ModifiersArray = NodeArray<Modifier>;

    struct Symbol
    {
        SymbolFlags flags;                                                          // Symbol flags
        string escapedName;                                                         // Name of symbol
        std::vector<Declaration> declarations;                                      // Declarations associated with this symbol
        DeclarationRef valueDeclaration;                                            // First value declaration of the symbol
        SymbolTable members;                                                        // Class, interface or object literal instance members
        SymbolTable exports;                                                        // Module exports
        SymbolTable globalExports;                                                  // Conditional global UMD exports
        /* @internal */ SymbolId id;                                                // Unique id (used to look up SymbolLinks)
        /* @internal */ number mergeId;                                             // Merge id (used to look up merged symbol)
        /* @internal */ SymbolRef parent;                                           // Parent symbol
        /* @internal */ SymbolRef exportSymbol;                                     // Exported symbol associated with this symbol
        /* @internal */ boolean constEnumOnlyModule;                                // True if module contains only const enums or other modules with only const enums
        /* @internal */ SymbolFlags isReferenced;                                   // True if the symbol is referenced elsewhere. Keeps track of the meaning of a reference in case a symbol is both a type parameter and parameter.
        /* @internal */ boolean isReplaceableByMethod;                              // Can this Javascript class property be replaced by a method symbol?
        /* @internal */ boolean isAssigned;                                         // True if the symbol is a parameter with assignments
        /* @internal */ std::map<number, Declaration> assignmentDeclarationMembers; // detected late-bound assignment declarations associated with the symbol
    };

    using DestructuringPattern = NodeRef /*BindingPattern | ObjectLiteralExpression | ArrayLiteralExpression*/;

    // Properties common to all types
    struct Type
    {
        TypeFlags flags;           // Flags
        /* @internal */ TypeId id; // Unique ID
        ///* @internal */ TypeChecker checker;
        Symbol symbol;                                            // Symbol associated with type (if any)
        DestructuringPattern pattern;                             // Destructuring pattern represented by type (if any)
        Symbol aliasSymbol;                                       // Alias associated with type
        std::vector<Type> aliasTypeArguments;                     // Alias type arguments (if any)
        /* @internal */ boolean aliasTypeArgumentsContainsMarker; // Alias type arguments (if any)
        /* @internal */
        TypeRef permissiveInstantiation; // Instantiation with type parameters mapped to wildcard type
        /* @internal */
        TypeRef restrictiveInstantiation; // Instantiation with type parameters mapped to unconstrained form
        /* @internal */
        TypeRef immediateBaseConstraint; // Immediate base constraint cache
        /* @internal */
        TypeRef widened; // Cached widened form of the type
    };

    struct Node : TextRange
    {
        SyntaxKind kind;
        NodeFlags flags;
        /* @internal */ ModifierFlags modifierFlagsCache;
        /* @internal */ TransformFlags transformFlags; // Flags for transforms
        NodeArray<Decorator> decorators;               // Array of decorators (in document order)
        ModifiersArray modifiers;                      // Array of modifiers
        /* @internal */ NodeId id;                     // Unique id (used to look up NodeLinks)
        NodeRef parent;                                // Parent node (initialized by binding)
        /* @internal */ NodeRef original;              // The original node if this is an updated node.
        /* @internal */ Symbol symbol;                 // Symbol declared by node (initialized by binding)
        /* @internal */ SymbolTable locals;            // Locals associated with node (initialized by binding)
        /* @internal */ NodeRef nextContainer;         // Next container in declaration order (initialized by binding)
        /* @internal */ Symbol localSymbol;            // Local symbol declared by node (initialized by binding only for exported nodes)
        ///* @internal */ FlowNode flowNode;                  // Associated FlowNode (initialized by binding)
        ///* @internal */ EmitNode emitNode;                  // Associated EmitNode (initialized by transforms)
        ///* @internal */ Type contextualType;                // Used to temporarily assign a contextual type during overload resolution
        ///* @internal */ InferenceContext inferenceContext;  // Inference context for contextual type
    };

    // TODO(rbuckton): Constraint 'TKind' to 'TokenSyntaxKind'
    template <SyntaxKind... TKind>
    struct Token : Node
    {
    };

    struct JSDocContainer
    {
        /* @internal */ std::vector<JSDoc> jsDoc;         // JSDoc that directly precedes this node
        /* @internal */ std::vector<JSDocTag> jsDocCache; // Cache for getJSDocTags
    };

    struct EndOfFileToken : Token<SyntaxKind::EndOfFileToken>, JSDocContainer
    {
    };

    // Punctuation
    template <SyntaxKind TKind>
    struct PunctuationToken : Token<TKind>
    {
    };

    using DotToken = PunctuationToken<SyntaxKind::DotToken>;
    using DotDotDotToken = PunctuationToken<SyntaxKind::DotDotDotToken>;
    using QuestionToken = PunctuationToken<SyntaxKind::QuestionToken>;
    using ExclamationToken = PunctuationToken<SyntaxKind::ExclamationToken>;
    using ColonToken = PunctuationToken<SyntaxKind::ColonToken>;
    using EqualsToken = PunctuationToken<SyntaxKind::EqualsToken>;
    using AsteriskToken = PunctuationToken<SyntaxKind::AsteriskToken>;
    using EqualsGreaterThanToken = PunctuationToken<SyntaxKind::EqualsGreaterThanToken>;
    using PlusToken = PunctuationToken<SyntaxKind::PlusToken>;
    using MinusToken = PunctuationToken<SyntaxKind::MinusToken>;
    using QuestionDotToken = PunctuationToken<SyntaxKind::QuestionDotToken>;

    // Keywords
    template <SyntaxKind TKind>
    struct KeywordToken : Token<TKind>
    {
    };

    using AssertsKeyword = KeywordToken<SyntaxKind::AssertsKeyword>;
    using AwaitKeyword = KeywordToken<SyntaxKind::AwaitKeyword>;

    /** @deprecated Use `AwaitKeyword` instead. */
    using AwaitKeywordToken = AwaitKeyword;

    /** @deprecated Use `AssertsKeyword` instead. */
    using AssertsToken = AssertsKeyword;

    template <SyntaxKind TKind>
    struct ModifierToken : KeywordToken<TKind>
    {
    };

    using AbstractKeyword = ModifierToken<SyntaxKind::AbstractKeyword>;
    using AsyncKeyword = ModifierToken<SyntaxKind::AsyncKeyword>;
    using ConstKeyword = ModifierToken<SyntaxKind::ConstKeyword>;
    using DeclareKeyword = ModifierToken<SyntaxKind::DeclareKeyword>;
    using DefaultKeyword = ModifierToken<SyntaxKind::DefaultKeyword>;
    using ExportKeyword = ModifierToken<SyntaxKind::ExportKeyword>;
    using PrivateKeyword = ModifierToken<SyntaxKind::PrivateKeyword>;
    using ProtectedKeyword = ModifierToken<SyntaxKind::ProtectedKeyword>;
    using PublicKeyword = ModifierToken<SyntaxKind::PublicKeyword>;
    using ReadonlyKeyword = ModifierToken<SyntaxKind::ReadonlyKeyword>;
    using StaticKeyword = ModifierToken<SyntaxKind::StaticKeyword>;

    using EntityName = Node /*Identifier | QualifiedName*/;

    using PropertyName = Node /*Identifier | StringLiteral | NumericLiteral | ComputedPropertyName | PrivateIdentifier*/;

    using MemberName = Node /*Identifier | PrivateIdentifier*/;

    struct QualifiedName : Node
    {
        // kind: SyntaxKind::QualifiedName;
        EntityName left;
        Identifier right;
        /*@internal*/ number jsdocDotPos; // QualifiedName occurs in JSDoc-style generic: Id1.Id2.<T>
    };

    using DeclarationName = Node /*
    | Identifier
    | PrivateIdentifier
    | StringLiteralLike
    | NumericLiteral
    | ComputedPropertyName
    | ElementAccessExpression
    | BindingPattern
    | EntityNameExpression
    */
        ;

    struct Declaration : Node
    {
        any _declarationBrand;
    };

    struct NamedDeclaration : Declaration
    {
        DeclarationName name;
    };

    /* @internal */
    struct DynamicNamedDeclaration : NamedDeclaration
    {
        ComputedPropertyName name;
    };

    /* @internal */
    struct DynamicNamedBinaryExpression : BinaryExpression
    {
        ElementAccessExpression left;
    };

    /* @internal */
    // A declaration that supports late-binding (used in checker)
    struct LateBoundDeclaration : DynamicNamedDeclaration
    {
        LateBoundName name;
    };

    /* @internal */
    struct LateBoundBinaryExpressionDeclaration : DynamicNamedBinaryExpression
    {
        LateBoundElementAccessExpression left;
    };

    /* @internal */
    struct LateBoundElementAccessExpression : ElementAccessExpression
    {
        EntityNameExpression argumentExpression;
    };

    struct DeclarationStatement : NamedDeclaration, Statement
    {
        Node /*Identifier | StringLiteral | NumericLiteral*/ name;
    };

    struct ComputedPropertyName : Node
    {
        // kind: SyntaxKind::ComputedPropertyName;
        Declaration parent;
        Expression expression;
    };

    struct PrivateIdentifier : Node
    {
        // kind: SyntaxKind::PrivateIdentifier;
        // escaping not strictly necessary
        // avoids gotchas in transforms and utils
        string escapedText;
    };

    /* @internal */
    // A name that supports late-binding (used in checker)
    struct LateBoundName : ComputedPropertyName
    {
        EntityNameExpression expression;
    };

    struct Decorator : Node
    {
        // kind: SyntaxKind::Decorator;
        NamedDeclaration parent;
        LeftHandSideExpression expression;
    };

    struct TypeParameterDeclaration : NamedDeclaration
    {
        // kind: SyntaxKind::TypeParameter;
        Node /*DeclarationWithTypeParameterChildren | InferTypeNode*/ parent;
        Identifier name;
        /** Note: Consider calling `getEffectiveConstraintOfTypeParameter` */
        TypeNode constraint;
        TypeNode default;

        // For error recovery purposes.
        Expression expression;
    };

    struct SignatureDeclarationBase : NamedDeclaration, JSDocContainer
    {
        SyntaxKind kind;
        PropertyName name;
        NodeArray<TypeParameterDeclaration> typeParameters;
        NodeArray<ParameterDeclaration> parameters;
        TypeNode type;
        /* @internal */ NodeArray<TypeNode> typeArguments; // Used for quick info, replaces typeParameters for instantiated signatures
    };

    using SignatureDeclaration = Node /*
    | CallSignatureDeclaration
    | ConstructSignatureDeclaration
    | MethodSignature
    | IndexSignatureDeclaration
    | FunctionTypeNode
    | ConstructorTypeNode
    | JSDocFunctionType
    | FunctionDeclaration
    | MethodDeclaration
    | ConstructorDeclaration
    | AccessorDeclaration
    | FunctionExpression
    | ArrowFunction
    */
        ;

    struct CallSignatureDeclaration : SignatureDeclarationBase, TypeElement
    {
        // kind: SyntaxKind::CallSignature;
    };

    struct ConstructSignatureDeclaration : SignatureDeclarationBase, TypeElement
    {
        // kind: SyntaxKind::ConstructSignature;
    };

    using BindingName = Node /*Identifier | BindingPattern*/;

    struct VariableDeclaration : NamedDeclaration, JSDocContainer
    {
        // kind: SyntaxKind::VariableDeclaration;
        Node /*VariableDeclarationList | CatchClause*/ parent;
        BindingName name;                  // Declared variable name
        ExclamationToken exclamationToken; // Optional definite assignment assertion
        TypeNode type;                     // Optional type annotation
        Expression initializer;            // Optional initializer
    };

    /* @internal */
    struct InitializedVariableDeclaration : VariableDeclaration
    {
        Expression initializer;
    };

    struct VariableDeclarationList : Node
    {
        // kind: SyntaxKind::VariableDeclarationList;
        Node /*VariableStatement | ForStatement | ForOfStatement | ForInStatement*/ parent;
        NodeArray<VariableDeclaration> declarations;
    };

    struct ParameterDeclaration : NamedDeclaration, JSDocContainer
    {
        // kind: SyntaxKind::Parameter;
        SignatureDeclaration parent;
        DotDotDotToken dotDotDotToken; // Present on rest parameter
        BindingName name;              // Declared parameter name.
        QuestionToken questionToken;   // Present on optional parameter
        TypeNode type;                 // Optional type annotation
        Expression initializer;        // Optional initializer
    };

    struct BindingElement : NamedDeclaration
    {
        // kind: SyntaxKind::BindingElement;
        BindingPattern parent;
        PropertyName propertyName;     // Binding property name (in object binding pattern)
        DotDotDotToken dotDotDotToken; // Present on rest element (in object binding pattern)
        BindingName name;              // Declared binding element name
        Expression initializer;        // Optional initializer
    };

    /*@internal*/
    using BindingElementGrandparent = Node /*BindingElement["parent"]["parent"]*/;

    struct PropertySignature : TypeElement, JSDocContainer
    {
        // kind: SyntaxKind::PropertySignature;
        PropertyName name;           // Declared property name
        QuestionToken questionToken; // Present on optional property
        TypeNode type;               // Optional type annotation
        Expression initializer;      // Present for use with reporting a grammar error
    };

    struct PropertyDeclaration : ClassElement, JSDocContainer
    {
        // kind: SyntaxKind::PropertyDeclaration;
        ClassLikeDeclaration parent;
        PropertyName name;
        QuestionToken questionToken; // Present for use with reporting a grammar error
        ExclamationToken exclamationToken;
        TypeNode type;
        Expression initializer; // Optional initializer
    };

    /*@internal*/
    struct PrivateIdentifierPropertyDeclaration : PropertyDeclaration
    {
        PrivateIdentifier name;
    };

    /* @internal */
    struct InitializedPropertyDeclaration : PropertyDeclaration
    {
        Expression initializer;
    };

    struct ObjectLiteralElement : NamedDeclaration
    {
        any _objectLiteralBrand;
        PropertyName name;
    };

    /** Unlike ObjectLiteralElement, excludes JSXAttribute and JSXSpreadAttribute. */
    using ObjectLiteralElementLike = Node /*
    | PropertyAssignment
    | ShorthandPropertyAssignment
    | SpreadAssignment
    | MethodDeclaration
    | AccessorDeclaration
    */
        ;

    struct PropertyAssignment : ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::PropertyAssignment;
        ObjectLiteralExpression parent;
        PropertyName name;
        QuestionToken questionToken;       // Present for use with reporting a grammar error
        ExclamationToken exclamationToken; // Present for use with reporting a grammar error
        Expression initializer;
    };

    struct ShorthandPropertyAssignment : ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::ShorthandPropertyAssignment;
        ObjectLiteralExpression parent;
        Identifier name;
        QuestionToken questionToken;
        ExclamationToken exclamationToken;
        // used when ObjectLiteralExpression is used in ObjectAssignmentPattern
        // it is a grammar error to appear in actual object initializer:
        EqualsToken equalsToken;
        Expression objectAssignmentInitializer;
    };

    struct SpreadAssignment : ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::SpreadAssignment;
        ObjectLiteralExpression parent;
        Expression expression;
    };

    using VariableLikeDeclaration = Node /*
    | VariableDeclaration
    | ParameterDeclaration
    | BindingElement
    | PropertyDeclaration
    | PropertyAssignment
    | PropertySignature
    | JsxAttribute
    | ShorthandPropertyAssignment
    | EnumMember
    | JSDocPropertyTag
    | JSDocParameterTag
    */
        ;

    struct PropertyLikeDeclaration : NamedDeclaration
    {
        PropertyName name;
    };

    struct ObjectBindingPattern : Node
    {
        // kind: SyntaxKind::ObjectBindingPattern;
        Node /*VariableDeclaration | ParameterDeclaration | BindingElement*/ parent;
        NodeArray<BindingElement> elements;
    };

    struct ArrayBindingPattern : Node
    {
        // kind: SyntaxKind::ArrayBindingPattern;
        Node /*VariableDeclaration | ParameterDeclaration | BindingElement*/ parent;
        NodeArray<ArrayBindingElement> elements;
    };

    using BindingPattern = Node /*ObjectBindingPattern | ArrayBindingPattern*/;

    using ArrayBindingElement = Node /*BindingElement | OmittedExpression*/;

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

        AsteriskToken asteriskToken;
        QuestionToken questionToken;
        ExclamationToken exclamationToken;
        Node /*Block | Expression*/ body;
        ///* @internal */ FlowNode endFlowNode;
        ///* @internal */ FlowNode returnFlowNode;
    };

    using FunctionLikeDeclaration = Node /*
    | FunctionDeclaration
    | MethodDeclaration
    | GetAccessorDeclaration
    | SetAccessorDeclaration
    | ConstructorDeclaration
    | FunctionExpression
    | ArrowFunction
    */
        ;

    /** @deprecated Use SignatureDeclaration */
    using FunctionLike = SignatureDeclaration;

    struct FunctionDeclaration : FunctionLikeDeclarationBase, DeclarationStatement
    {
        // kind: SyntaxKind::FunctionDeclaration;
        Identifier name;
        FunctionBody body;
    };

    struct MethodSignature : SignatureDeclarationBase, TypeElement
    {
        // kind: SyntaxKind::MethodSignature;
        ObjectTypeDeclaration parent;
        PropertyName name;
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
    struct MethodDeclaration : FunctionLikeDeclarationBase, ClassElement, ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::MethodDeclaration;
        Node /*ClassLikeDeclaration | ObjectLiteralExpression*/ parent;
        PropertyName name;
        FunctionBody body;
        /* @internal*/ ExclamationToken exclamationToken; // Present for use with reporting a grammar error
    };

    struct ConstructorDeclaration : FunctionLikeDeclarationBase, ClassElement, JSDocContainer
    {
        // kind: SyntaxKind::Constructor;
        ClassLikeDeclaration parent;
        FunctionBody body;
        /* @internal */ NodeArray<TypeParameterDeclaration> typeParameters; // Present for use with reporting a grammar error
        /* @internal */ TypeNode type;                                      // Present for use with reporting a grammar error
    };

    /** For when we encounter a semicolon in a class declaration. ES6 allows these as class elements. */
    struct SemicolonClassElement : ClassElement
    {
        // kind: SyntaxKind::SemicolonClassElement;
        ClassLikeDeclaration parent;
    };

    // See the comment on MethodDeclaration for the intuition behind GetAccessorDeclaration being a
    // ClassElement and an ObjectLiteralElement.
    struct GetAccessorDeclaration : FunctionLikeDeclarationBase, ClassElement, ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::GetAccessor;
        Node /*ClassLikeDeclaration | ObjectLiteralExpression*/ parent;
        PropertyName name;
        FunctionBody body;
        /* @internal */ NodeArray<TypeParameterDeclaration> typeParameters; // Present for use with reporting a grammar error
    };

    // See the comment on MethodDeclaration for the intuition behind SetAccessorDeclaration being a
    // ClassElement and an ObjectLiteralElement.
    struct SetAccessorDeclaration : FunctionLikeDeclarationBase, ClassElement, ObjectLiteralElement, JSDocContainer
    {
        // kind: SyntaxKind::SetAccessor;
        Node /*ClassLikeDeclaration | ObjectLiteralExpression*/ parent;
        PropertyName name;
        FunctionBody body;
        /* @internal */ NodeArray<TypeParameterDeclaration> typeParameters; // Present for use with reporting a grammar error
        /* @internal */ TypeNode type;                                      // Present for use with reporting a grammar error
    };

    using AccessorDeclaration = Node /*GetAccessorDeclaration | SetAccessorDeclaration*/;

    struct IndexSignatureDeclaration : SignatureDeclarationBase, ClassElement, TypeElement
    {
        // kind: SyntaxKind::IndexSignature;
        ObjectTypeDeclaration parent;
        TypeNode type;
    };

    struct TypeNode : Node
    {
        any _typeNodeBrand;
    };

    /* @internal */
    struct TypeNode : Node
    {
        // kind: TypeNodeSyntaxKind;
    };

    template <SyntaxKind TKind>
    struct KeywordTypeNode : KeywordToken<TKind>, TypeNode
    {
    };

    struct ImportTypeNode : NodeWithTypeArguments
    {
        // kind: SyntaxKind::ImportType;
        boolean isTypeOf;
        TypeNode argument;
        EntityName qualifier;
    };

    /* @internal */
    struct argumentType_ : LiteralTypeNode
    {
        StringLiteral iteral;
    };

    struct LiteralImportTypeNode : ImportTypeNode
    {
        argumentType_ argument;
    };

    struct ThisTypeNode : TypeNode
    {
        // kind: SyntaxKind::ThisType;
    };

    using FunctionOrConstructorTypeNode = Node /*FunctionTypeNode | ConstructorTypeNode*/;

    struct FunctionOrConstructorTypeNodeBase : TypeNode, SignatureDeclarationBase
    {
        // kind: SyntaxKind::FunctionType | SyntaxKind::ConstructorType;
        TypeNode type;
    };

    struct FunctionTypeNode : FunctionOrConstructorTypeNodeBase
    {
        // kind: SyntaxKind::FunctionType;
    };

    struct ConstructorTypeNode : FunctionOrConstructorTypeNodeBase
    {
        // kind: SyntaxKind::ConstructorType;
    };

    struct NodeWithTypeArguments : TypeNode
    {
        NodeArray<TypeNode> typeArguments;
    };

    using TypeReferenceType = Node /*TypeReferenceNode | ExpressionWithTypeArguments*/;

    struct TypeReferenceNode : NodeWithTypeArguments
    {
        // kind: SyntaxKind::TypeReference;
        EntityName typeName;
    };

    struct TypePredicateNode : TypeNode
    {
        // kind: SyntaxKind::TypePredicate;
        Node /*SignatureDeclaration | JSDocTypeExpression*/ parent;
        AssertsToken assertsModifier;
        Node /*Identifier | ThisTypeNode*/ parameterName;
        TypeNode type;
    };

    struct TypeQueryNode : TypeNode
    {
        // kind: SyntaxKind::TypeQuery;
        EntityName exprName;
    };

    // A TypeLiteral is the declaration node for an anonymous symbol.
    struct TypeLiteralNode : TypeNode, Declaration
    {
        // kind: SyntaxKind::TypeLiteral;
        NodeArray<TypeElement> members;
    };

    struct ArrayTypeNode : TypeNode
    {
        // kind: SyntaxKind::ArrayType;
        TypeNode elementType;
    };

    struct TupleTypeNode : TypeNode
    {
        // kind: SyntaxKind::TupleType;
        NodeArray<Node /*TypeNode | NamedTupleMember*/> elements;
    };

    struct NamedTupleMember : TypeNode, JSDocContainer, Declaration
    {
        // kind: SyntaxKind::NamedTupleMember;
        Token<SyntaxKind::DotDotDotToken> dotDotDotToken;
        Identifier name;
        Token<SyntaxKind::QuestionToken> questionToken;
        TypeNode type;
    };

    struct OptionalTypeNode : TypeNode
    {
        // kind: SyntaxKind::OptionalType;
        TypeNode type;
    };

    struct RestTypeNode : TypeNode
    {
        // kind: SyntaxKind::RestType;
        TypeNode type;
    };

    using UnionOrIntersectionTypeNode = Node /*UnionTypeNode | IntersectionTypeNode*/;

    struct UnionTypeNode : TypeNode
    {
        // kind: SyntaxKind::UnionType;
        NodeArray<TypeNode> types;
    };

    struct IntersectionTypeNode : TypeNode
    {
        // kind: SyntaxKind::IntersectionType;
        NodeArray<TypeNode> types;
    };

    struct ConditionalTypeNode : TypeNode
    {
        // kind: SyntaxKind::ConditionalType;
        TypeNode checkType;
        TypeNode extendsType;
        TypeNode trueType;
        TypeNode falseType;
    };

    struct InferTypeNode : TypeNode
    {
        // kind: SyntaxKind::InferType;
        TypeParameterDeclaration typeParameter;
    };

    struct ParenthesizedTypeNode : TypeNode
    {
        // kind: SyntaxKind::ParenthesizedType;
        TypeNode type;
    };

    struct TypeOperatorNode : TypeNode
    {
        // kind: SyntaxKind::TypeOperator;
        SyntaxKind _operator;
        TypeNode type;
    };

    /* @internal */
    struct UniqueTypeOperatorNode : TypeOperatorNode
    {
        SyntaxKind _operator;
    };

    struct IndexedAccessTypeNode : TypeNode
    {
        // kind: SyntaxKind::IndexedAccessType;
        TypeNode objectType;
        TypeNode indexType;
    };

    struct MappedTypeNode : TypeNode, Declaration
    {
        // kind: SyntaxKind::MappedType;
        Node /*ReadonlyToken | PlusToken | MinusToken*/ readonlyToken;
        TypeParameterDeclaration typeParameter;
        TypeNode nameType;
        Node /*QuestionToken | PlusToken | MinusToken*/ questionToken;
        TypeNode type;
    };

    struct LiteralTypeNode : TypeNode
    {
        // kind: SyntaxKind::LiteralType;
        Node /*NullLiteral | BooleanLiteral | LiteralExpression | PrefixUnaryExpression*/ literal;
    };

    struct StringLiteral : LiteralExpression, Declaration
    {
        // kind: SyntaxKind::StringLiteral;
        /* @internal */ Node /*Identifier | StringLiteralLike | NumericLiteral*/ textSourceNode; // Allows a StringLiteral to get its text from another node (used by transforms).
                                                                                                 /** Note: this is only set when synthesizing a node, not during parsing. */
        /* @internal */ boolean singleQuote;
    };

    using StringLiteralLike = Node /*StringLiteral | NoSubstitutionTemplateLiteral*/;
    using PropertyNameLiteral = Node /*Identifier | StringLiteralLike | NumericLiteral*/;

    struct TemplateLiteralTypeNode : TypeNode
    {
        NodeArray<TemplateLiteralTypeSpan> templateSpans;
    };

    struct TemplateLiteralTypeSpan : TypeNode
    {
        TypeNode type;
        Node /*TemplateMiddle | TemplateTail*/ literal;
    };

    // Note: 'brands' in our syntax nodes serve to give us a small amount of nominal typing.
    // Consider 'Expression'.  Without the brand, 'Expression' is actually no different
    // (structurally) than 'Node'.  Because of this you can pass any Node to a function that
    // takes an Expression without any error.  By using the 'brands' we ensure that the type
    // checker actually thinks you have something of the right type.  Note: the brands are
    // never actually given values.  At runtime they have zero cost.

    struct Expression : Node
    {
        any _expressionBrand;
    };

    struct OmittedExpression : Expression
    {
        // kind: SyntaxKind::OmittedExpression;
    };

    // Represents an expression that is elided as part of a transformation to emit comments on a
    // not-emitted node. The 'expression' property of a PartiallyEmittedExpression should be emitted.
    struct PartiallyEmittedExpression : LeftHandSideExpression
    {
        // kind: SyntaxKind::PartiallyEmittedExpression;
        Expression expression;
    };

    struct UnaryExpression : Expression
    {
        any _unaryExpressionBrand;
    };

    /** Deprecated, please use UpdateExpression */
    using IncrementExpression = UpdateExpression;
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
        UnaryExpression operand;
    };

    // see: https://tc39.github.io/ecma262/#prod-UpdateExpression
    using PostfixUnaryOperator = SyntaxKind;

    struct PostfixUnaryExpression : UpdateExpression
    {
        // kind: SyntaxKind::PostfixUnaryExpression;
        LeftHandSideExpression operand;
        PostfixUnaryOperator _operator;
    };

    struct LeftHandSideExpression : UpdateExpression
    {
        any _leftHandSideExpressionBrand;
    };

    struct MemberExpression : LeftHandSideExpression
    {
        any _memberExpressionBrand;
    };

    struct PrimaryExpression : MemberExpression
    {
        any _primaryExpressionBrand;
    };

    struct Identifier : PrimaryExpression, Declaration
    {
        // kind: SyntaxKind::Identifier;
        /**
     * Prefer to use `id.unescapedText`. (Note: This is available only in services, not internally to the TypeScript compiler.)
     * Text of identifier, but if the identifier begins with two underscores, this will begin with three.
     */
        string escapedText;
        SyntaxKind originalKeywordKind;                                                      // Original syntaxKind which get set so that we can report an error later
        /*@internal*/ GeneratedIdentifierFlags autoGenerateFlags;                            // Specifies whether to auto-generate the text for an identifier.
        /*@internal*/ number autoGenerateId;                                                 // Ensures unique generated identifiers get unique names, but clones get the same name.
        /*@internal*/ ImportSpecifier generatedImportReference;                              // Reference to the generated import specifier this identifier refers to
        boolean isInJSDocNamespace;                                                          // if the node is a member in a JSDoc namespace
        /*@internal*/ NodeArray<Node /*TypeNode | TypeParameterDeclaration*/> typeArguments; // Only defined on synthesized nodes. Though not syntactically valid, used in emitting diagnostics, quickinfo, and signature help.
        /*@internal*/ number jsdocDotPos;                                                    // Identifier occurs in JSDoc-style generic: Id.<T>
    };

    // Transient identifier node (marked by id === -1)
    struct TransientIdentifier : Identifier
    {
        Symbol resolvedSymbol;
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

    using BooleanLiteral = Node /*TrueLiteral | FalseLiteral*/;

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
        UnaryExpression expression;
    };

    struct TypeOfExpression : UnaryExpression
    {
        // kind: SyntaxKind::TypeOfExpression;
        UnaryExpression expression;
    };

    struct VoidExpression : UnaryExpression
    {
        // kind: SyntaxKind::VoidExpression;
        UnaryExpression expression;
    };

    struct AwaitExpression : UnaryExpression
    {
        // kind: SyntaxKind::AwaitExpression;
        UnaryExpression expression;
    };

    struct YieldExpression : Expression
    {
        // kind: SyntaxKind::YieldExpression;
        AsteriskToken asteriskToken;
        Expression expression;
    };

    struct SyntheticExpression : Expression
    {
        // kind: SyntaxKind::SyntheticExpression;
        boolean isSpread;
        Type type;
        Node /*ParameterDeclaration | NamedTupleMember*/ tupleNameSource;
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

    using BinaryOperatorToken = Token<SyntaxKind::AsteriskAsteriskToken, SyntaxKind::CommaToken /*to keep it short, [from, to]*/>;

    struct BinaryExpression : Expression, Declaration
    {
        // kind: SyntaxKind::BinaryExpression;
        Expression left;
        BinaryOperatorToken operatorToken;
        Expression right;
    };

    using AssignmentOperatorToken = Token<SyntaxKind::EqualsToken, SyntaxKind::QuestionQuestionEqualsToken /*to keep it short, [from, to]*/>;

    template <typename TOperator /*AssignmentOperatorToken*/>
    struct AssignmentExpression : BinaryExpression
    {
        LeftHandSideExpression left;
        TOperator operatorToken;
    };

    struct ObjectDestructuringAssignment : AssignmentExpression<EqualsToken>
    {
        ObjectLiteralExpression left;
    };

    struct ArrayDestructuringAssignment : AssignmentExpression<EqualsToken>
    {
        ArrayLiteralExpression left;
    };

    using DestructuringAssignment = Node /*
    | ObjectDestructuringAssignment
    | ArrayDestructuringAssignment
    */
        ;

    using BindingOrAssignmentElement = Node /*
    | VariableDeclaration
    | ParameterDeclaration
    | ObjectBindingOrAssignmentElement
    | ArrayBindingOrAssignmentElement
    */
        ;

    using ObjectBindingOrAssignmentElement = Node /*
    | BindingElement
    | PropertyAssignment // AssignmentProperty
    | ShorthandPropertyAssignment // AssignmentProperty
    | SpreadAssignment // AssignmentRestProperty
    */
        ;

    using ArrayBindingOrAssignmentElement = Node /*
    | BindingElement
    | OmittedExpression // Elision
    | SpreadElement // AssignmentRestElement
    | ArrayLiteralExpression // ArrayAssignmentPattern
    | ObjectLiteralExpression // ObjectAssignmentPattern
    | AssignmentExpression<EqualsToken> // AssignmentElement
    | Identifier // DestructuringAssignmentTarget
    | PropertyAccessExpression // DestructuringAssignmentTarget
    | ElementAccessExpression // DestructuringAssignmentTarget
    */
        ;

    using BindingOrAssignmentElementRestIndicator = Node /*
    | DotDotDotToken // from BindingElement
    | SpreadElement // AssignmentRestElement
    | SpreadAssignment // AssignmentRestProperty
    */
        ;

    using BindingOrAssignmentElementTarget = Node /*
    | BindingOrAssignmentPattern
    | Identifier
    | PropertyAccessExpression
    | ElementAccessExpression
    | OmittedExpression
    */
        ;

    using ObjectBindingOrAssignmentPattern = Node /*
    | ObjectBindingPattern
    | ObjectLiteralExpression // ObjectAssignmentPattern
    */
        ;

    using ArrayBindingOrAssignmentPattern = Node /*
    | ArrayBindingPattern
    | ArrayLiteralExpression // ArrayAssignmentPattern
    */
        ;

    using AssignmentPattern = Node /*ObjectLiteralExpression | ArrayLiteralExpression*/;

    using BindingOrAssignmentPattern = Node /*ObjectBindingOrAssignmentPattern | ArrayBindingOrAssignmentPattern*/;

    struct ConditionalExpression : Expression
    {
        // kind: SyntaxKind::ConditionalExpression;
        Expression condition;
        QuestionToken questionToken;
        Expression whenTrue;
        ColonToken colonToken;
        Expression whenFalse;
    };

    using FunctionBody = Block;
    using ConciseBody = Node /*FunctionBody | Expression*/;

    struct FunctionExpression : PrimaryExpression, FunctionLikeDeclarationBase, JSDocContainer
    {
        // kind: SyntaxKind::FunctionExpression;
        Identifier name;
        FunctionBody body; // Required, whereas the member inherited from FunctionDeclaration is optional
    };

    struct ArrowFunction : Expression, FunctionLikeDeclarationBase, JSDocContainer
    {
        // kind: SyntaxKind::ArrowFunction;
        EqualsGreaterThanToken equalsGreaterThanToken;
        ConciseBody body;
        never name;
    };

    // The text property of a LiteralExpression stores the interpreted value of the literal in text form. For a StringLiteral,
    // or any literal of a template, this means quotes have been removed and escapes have been converted to actual characters.
    // For a NumericLiteral, the stored value is the toString() representation of the number. For example 1, 1.00, and 1e0 are all stored as just "1".
    struct LiteralLikeNode : Node
    {
        string text;
        boolean isUnterminated;
        boolean hasExtendedUnicodeEscape;
    };

    struct TemplateLiteralLikeNode : LiteralLikeNode
    {
        string rawText;
        /* @internal */
        TokenFlags templateFlags;
    };

    // The text property of a LiteralExpression stores the interpreted value of the literal in text form. For a StringLiteral,
    // or any literal of a template, this means quotes have been removed and escapes have been converted to actual characters.
    // For a NumericLiteral, the stored value is the toString() representation of the number. For example 1, 1.00, and 1e0 are all stored as just "1".
    struct LiteralExpression : LiteralLikeNode, PrimaryExpression
    {
        any _literalExpressionBrand;
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

    using LiteralToken = Node /*
    | NumericLiteral
    | BigIntLiteral
    | StringLiteral
    | JsxText
    | RegularExpressionLiteral
    | NoSubstitutionTemplateLiteral
    */
        ;

    struct TemplateHead : TemplateLiteralLikeNode
    {
        // kind: SyntaxKind::TemplateHead;
        Node /*TemplateExpression | TemplateLiteralTypeNode*/ parent;
        /* @internal */
        TokenFlags templateFlags;
    };

    struct TemplateMiddle : TemplateLiteralLikeNode
    {
        // kind: SyntaxKind::TemplateMiddle;
        Node /*TemplateSpan | TemplateLiteralTypeSpan*/ parent;
        /* @internal */
        TokenFlags templateFlags;
    };

    struct TemplateTail : TemplateLiteralLikeNode
    {
        // kind: SyntaxKind::TemplateTail;
        Node /*TemplateSpan | TemplateLiteralTypeSpan*/ parent;
        /* @internal */
        TokenFlags templateFlags;
    };

    using PseudoLiteralToken = Node /*
    | TemplateHead
    | TemplateMiddle
    | TemplateTail
    */
        ;

    using TemplateLiteralToken = Node /*
    | NoSubstitutionTemplateLiteral
    | PseudoLiteralToken
    */
        ;

    struct TemplateExpression : PrimaryExpression
    {
        // kind: SyntaxKind::TemplateExpression;
        TemplateHead head;
        NodeArray<TemplateSpan> templateSpans;
    };

    using TemplateLiteral = Node /*
    | TemplateExpression
    | NoSubstitutionTemplateLiteral
    */
        ;

    // Each of these corresponds to a substitution expression and a template literal, in that order.
    // The template literal must have kind TemplateMiddleLiteral or TemplateTailLiteral.
    struct TemplateSpan : Node
    {
        // kind: SyntaxKind::TemplateSpan;
        TemplateExpression parent;
        Expression expression;
        Node /*TemplateMiddle | TemplateTail*/ literal;
    };

    struct ParenthesizedExpression : PrimaryExpression, JSDocContainer
    {
        // kind: SyntaxKind::ParenthesizedExpression;
        Expression expression;
    };

    struct ArrayLiteralExpression : PrimaryExpression
    {
        // kind: SyntaxKind::ArrayLiteralExpression;
        NodeArray<Expression> elements;
        /* @internal */
        boolean multiLine;
    };

    struct SpreadElement : Expression
    {
        // kind: SyntaxKind::SpreadElement;
        Node /*ArrayLiteralExpression | CallExpression | NewExpression*/ parent;
        Expression expression;
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
        NodeArray<T> properties;
    };

    // An ObjectLiteralExpression is the declaration node for an anonymous symbol.
    struct ObjectLiteralExpression : ObjectLiteralExpressionBase<ObjectLiteralElementLike>
    {
        // kind: SyntaxKind::ObjectLiteralExpression;
        /* @internal */
        boolean multiLine;
    };

    using EntityNameExpression = Node /*Identifier | PropertyAccessEntityNameExpression*/;
    using EntityNameOrEntityNameExpression = Node /*EntityName | EntityNameExpression*/;
    using AccessExpression = Node /*PropertyAccessExpression | ElementAccessExpression*/;

    struct PropertyAccessExpression : MemberExpression, NamedDeclaration
    {
        // kind: SyntaxKind::PropertyAccessExpression;
        LeftHandSideExpression expression;
        QuestionDotToken questionDotToken;
        MemberName name;
    };

    /*@internal*/
    struct PrivateIdentifierPropertyAccessExpression : PropertyAccessExpression
    {
        PrivateIdentifier name;
    };

    struct PropertyAccessChain : PropertyAccessExpression
    {
        any _optionalChainBrand;
        MemberName name;
    };

    /* @internal */
    struct PropertyAccessChainRoot : PropertyAccessChain
    {
        QuestionDotToken questionDotToken;
    };

    struct SuperPropertyAccessExpression : PropertyAccessExpression
    {
        SuperExpression expression;
    };

    /** Brand for a PropertyAccessExpression which, like a QualifiedName, consists of a sequence of identifiers separated by dots. */
    struct PropertyAccessEntityNameExpression : PropertyAccessExpression
    {
        any _propertyAccessExpressionLikeQualifiedNameBrand;
        EntityNameExpression expression;
        Identifier name;
    };

    struct ElementAccessExpression : MemberExpression
    {
        // kind: SyntaxKind::ElementAccessExpression;
        LeftHandSideExpression expression;
        QuestionDotToken questionDotToken;
        Expression argumentExpression;
    };

    struct ElementAccessChain : ElementAccessExpression
    {
        any _optionalChainBrand;
    };

    /* @internal */
    struct ElementAccessChainRoot : ElementAccessChain
    {
        QuestionDotToken questionDotToken;
    };

    struct SuperElementAccessExpression : ElementAccessExpression
    {
        SuperExpression expression;
    };

    // see: https://tc39.github.io/ecma262/#prod-SuperProperty
    using SuperProperty = Node /*SuperPropertyAccessExpression | SuperElementAccessExpression*/;

    struct CallExpression : LeftHandSideExpression, Declaration
    {
        // kind: SyntaxKind::CallExpression;
        LeftHandSideExpression expression;
        QuestionDotToken questionDotToken;
        NodeArray<TypeNode> typeArguments;
        NodeArray<Expression> arguments;
    };

    struct CallChain : CallExpression
    {
        any _optionalChainBrand;
    };

    /* @internal */
    struct CallChainRoot : CallChain
    {
        QuestionDotToken questionDotToken;
    };

    using OptionalChain = Node /*
    | PropertyAccessChain
    | ElementAccessChain
    | CallChain
    | NonNullChain
    */
        ;

    /* @internal */
    using OptionalChainRoot = Node /*
    | PropertyAccessChainRoot
    | ElementAccessChainRoot
    | CallChainRoot
    */
        ;

    /** @internal */
    struct BindableObjectDefinePropertyCall : CallExpression
    {
        struct arguments_ : NodeArray<Node>, TextRange
        {
        } arguments;
    };

    /** @internal */
    using BindableStaticNameExpression =
        | EntityNameExpression | BindableStaticElementAccessExpression;

    /** @internal */
    struct LiteralLikeElementAccessExpression : ElementAccessExpression, Declaration
    {
        Node /*StringLiteralLike | NumericLiteral*/ argumentExpression;
    };

    /** @internal */
    struct BindableStaticElementAccessExpression : LiteralLikeElementAccessExpression
    {
        BindableStaticNameExpression expression;
    };

    /** @internal */
    struct BindableElementAccessExpression : ElementAccessExpression
    {
        BindableStaticNameExpression expression;
    };

    /** @internal */
    using BindableStaticAccessExpression = Node /*
    | PropertyAccessEntityNameExpression
    | BindableStaticElementAccessExpression
    */
        ;

    /** @internal */
    using BindableAccessExpression = Node /*
    | PropertyAccessEntityNameExpression
    | BindableElementAccessExpression
    */
        ;

    /** @internal */
    struct BindableStaticPropertyAssignmentExpression : BinaryExpression
    {
        BindableStaticAccessExpression left;
    };

    /** @internal */
    struct BindablePropertyAssignmentExpression : BinaryExpression
    {
        BindableAccessExpression left;
    };

    // see: https://tc39.github.io/ecma262/#prod-SuperCall
    struct SuperCall : CallExpression
    {
        SuperExpression expression;
    };

    struct ImportCall : CallExpression
    {
        ImportExpression expression;
    };

    struct ExpressionWithTypeArguments : NodeWithTypeArguments
    {
        // kind: SyntaxKind::ExpressionWithTypeArguments;
        Node /*HeritageClause | JSDocAugmentsTag | JSDocImplementsTag*/ parent;
        LeftHandSideExpression expression;
    };

    struct NewExpression : PrimaryExpression, Declaration
    {
        // kind: SyntaxKind::NewExpression;
        LeftHandSideExpression expression;
        NodeArray<TypeNode> typeArguments;
        NodeArray<Expression> arguments;
    };

    struct TaggedTemplateExpression : MemberExpression
    {
        // kind: SyntaxKind::TaggedTemplateExpression;
        LeftHandSideExpression tag;
        NodeArray<TypeNode> typeArguments;
        TemplateLiteral _template;
        /*@internal*/ QuestionDotToken questionDotToken; // NOTE: Invalid syntax, only used to report a grammar error.
    };

    using CallLikeExpression = Node /*
    | CallExpression
    | NewExpression
    | TaggedTemplateExpression
    | Decorator
    | JsxOpeningLikeElement
    */
        ;

    struct AsExpression : Expression
    {
        // kind: SyntaxKind::AsExpression;
        Expression expression;
        TypeNode type;
    };

    struct TypeAssertion : UnaryExpression
    {
        // kind: SyntaxKind::TypeAssertionExpression;
        TypeNode type;
        UnaryExpression expression;
    };

    using AssertionExpression = Node /*
    | TypeAssertion
    | AsExpression
    */
        ;

    struct NonNullExpression : LeftHandSideExpression
    {
        // kind: SyntaxKind::NonNullExpression;
        Expression expression;
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
        Identifier name;
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
        JsxOpeningElement openingElement;
        NodeArray<JsxChild> children;
        JsxClosingElement closingElement;
    };

    /// Either the opening tag in a <Tag>...</Tag> pair or the lone <Tag /> in a self-closing form
    using JsxOpeningLikeElement = Node /*
    | JsxSelfClosingElement
    | JsxOpeningElement
    */
        ;

    using JsxAttributeLike = Node /*
    | JsxAttribute
    | JsxSpreadAttribute
    */
        ;

    using JsxTagNameExpression = Node /*
    | Identifier
    | ThisExpression
    | JsxTagNamePropertyAccess
    */
        ;

    struct JsxTagNamePropertyAccess : PropertyAccessExpression
    {
        JsxTagNameExpression expression;
    };

    struct JsxAttributes : ObjectLiteralExpressionBase<JsxAttributeLike>
    {
        // kind: SyntaxKind::JsxAttributes;
        JsxOpeningLikeElement parent;
    };

    /// The opening element of a <Tag>...</Tag> JsxElement
    struct JsxOpeningElement : Expression
    {
        // kind: SyntaxKind::JsxOpeningElement;
        JsxElement parent;
        JsxTagNameExpression tagName;
        NodeArray<TypeNode> typeArguments;
        JsxAttributes attributes;
    };

    /// A JSX expression of the form <TagName attrs />
    struct JsxSelfClosingElement : PrimaryExpression
    {
        // kind: SyntaxKind::JsxSelfClosingElement;
        JsxTagNameExpression tagName;
        NodeArray<TypeNode> typeArguments;
        JsxAttributes attributes;
    };

    /// A JSX expression of the form <>...</>
    struct JsxFragment : PrimaryExpression
    {
        // kind: SyntaxKind::JsxFragment;
        JsxOpeningFragment openingFragment;
        NodeArray<JsxChild> children;
        JsxClosingFragment closingFragment;
    };

    /// The opening element of a <>...</> JsxFragment
    struct JsxOpeningFragment : Expression
    {
        // kind: SyntaxKind::JsxOpeningFragment;
        JsxFragment parent;
    };

    /// The closing element of a <>...</> JsxFragment
    struct JsxClosingFragment : Expression
    {
        // kind: SyntaxKind::JsxClosingFragment;
        JsxFragment parent;
    };

    struct JsxAttribute : ObjectLiteralElement
    {
        // kind: SyntaxKind::JsxAttribute;
        JsxAttributes parent;
        Identifier name;
        /// JSX attribute initializers are optional; <X y /> is sugar for <X y={true} />
        Node /*StringLiteral | JsxExpression*/ initializer;
    };

    struct JsxSpreadAttribute : ObjectLiteralElement
    {
        // kind: SyntaxKind::JsxSpreadAttribute;
        JsxAttributes parent;
        Expression expression;
    };

    struct JsxClosingElement : Node
    {
        // kind: SyntaxKind::JsxClosingElement;
        JsxElement parent;
        JsxTagNameExpression tagName;
    };

    struct JsxExpression : Expression
    {
        // kind: SyntaxKind::JsxExpression;
        Node /*JsxElement | JsxAttributeLike*/ parent;
        Token<SyntaxKind::DotDotDotToken> dotDotDotToken;
        Expression expression;
    };

    struct JsxText : LiteralLikeNode
    {
        // kind: SyntaxKind::JsxText;
        JsxElement parent;
        boolean containsOnlyTriviaWhiteSpaces;
    };

    using JsxChild = Node /*
    | JsxText
    | JsxExpression
    | JsxElement
    | JsxSelfClosingElement
    | JsxFragment
    */
        ;

    struct Statement : Node
    {
        any _statementBrand;
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
        NodeArray<Expression> elements;
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
        Expression expression;
        Expression thisArg;
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
        /*@internal*/ NodeArray<Decorator> decorators; // Present for use with reporting a grammar error
        /*@internal*/ ModifiersArray modifiers;        // Present for use with reporting a grammar error
        // kind: SyntaxKind::MissingDeclaration;
        Identifier name;
    };

    using BlockLike = Node /*
    | SourceFile
    | Block
    | ModuleBlock
    | CaseOrDefaultClause
    */
        ;

    struct Block : Statement
    {
        // kind: SyntaxKind::Block;
        NodeArray<Statement> statements;
        /*@internal*/ boolean multiLine;
    };

    struct VariableStatement : Statement, JSDocContainer
    {
        /* @internal*/ NodeArray<Decorator> decorators; // Present for use with reporting a grammar error
        // kind: SyntaxKind::VariableStatement;
        VariableDeclarationList declarationList;
    };

    struct ExpressionStatement : Statement, JSDocContainer
    {
        // kind: SyntaxKind::ExpressionStatement;
        Expression expression;
    };

    /* @internal */
    struct PrologueDirective : ExpressionStatement
    {
        StringLiteral expression;
    };

    struct IfStatement : Statement
    {
        // kind: SyntaxKind::IfStatement;
        Expression expression;
        Statement thenStatement;
        Statement elseStatement;
    };

    struct IterationStatement : Statement
    {
        Statement statement;
    };

    struct DoStatement : IterationStatement
    {
        // kind: SyntaxKind::DoStatement;
        Expression expression;
    };

    struct WhileStatement : IterationStatement
    {
        // kind: SyntaxKind::WhileStatement;
        Expression expression;
    };

    using ForInitializer = Node /*
    | VariableDeclarationList
    | Expression
    */
        ;

    struct ForStatement : IterationStatement
    {
        // kind: SyntaxKind::ForStatement;
        ForInitializer initializer;
        Expression condition;
        Expression incrementor;
    };

    using ForInOrOfStatement = Node /*
    | ForInStatement
    | ForOfStatement
    */
        ;

    struct ForInStatement : IterationStatement
    {
        // kind: SyntaxKind::ForInStatement;
        ForInitializer initializer;
        Expression expression;
    };

    struct ForOfStatement : IterationStatement
    {
        // kind: SyntaxKind::ForOfStatement;
        AwaitKeywordToken awaitModifier;
        ForInitializer initializer;
        Expression expression;
    };

    struct BreakStatement : Statement
    {
        // kind: SyntaxKind::BreakStatement;
        Identifier label;
    };

    struct ContinueStatement : Statement
    {
        // kind: SyntaxKind::ContinueStatement;
        Identifier label;
    };

    using BreakOrContinueStatement = Node /*
    | BreakStatement
    | ContinueStatement
    */
        ;

    struct ReturnStatement : Statement
    {
        // kind: SyntaxKind::ReturnStatement;
        Expression expression;
    };

    struct WithStatement : Statement
    {
        // kind: SyntaxKind::WithStatement;
        Expression expression;
        Statement statement;
    };

    struct SwitchStatement : Statement
    {
        // kind: SyntaxKind::SwitchStatement;
        Expression expression;
        CaseBlock caseBlock;
        boolean possiblyExhaustive; // initialized by binding
    };

    struct CaseBlock : Node
    {
        // kind: SyntaxKind::CaseBlock;
        SwitchStatement parent;
        NodeArray<CaseOrDefaultClause> clauses;
    };

    struct CaseClause : Node
    {
        // kind: SyntaxKind::CaseClause;
        CaseBlock parent;
        Expression expression;
        NodeArray<Statement> statements;
        ///* @internal */ FlowNode fallthroughFlowNode;
    };

    struct DefaultClause : Node
    {
        // kind: SyntaxKind::DefaultClause;
        CaseBlock parent;
        NodeArray<Statement> statements;
        ///* @internal */ FlowNode fallthroughFlowNode;
    };

    using CaseOrDefaultClause = Node /*
    | CaseClause
    | DefaultClause
    */
        ;

    struct LabeledStatement : Statement, JSDocContainer
    {
        // kind: SyntaxKind::LabeledStatement;
        Identifier label;
        Statement statement;
    };

    struct ThrowStatement : Statement
    {
        // kind: SyntaxKind::ThrowStatement;
        Expression expression;
    };

    struct TryStatement : Statement
    {
        // kind: SyntaxKind::TryStatement;
        Block tryBlock;
        CatchClause catchClause;
        Block finallyBlock;
    };

    struct CatchClause : Node
    {
        // kind: SyntaxKind::CatchClause;
        TryStatement parent;
        VariableDeclaration variableDeclaration;
        Block block;
    };

    using ObjectTypeDeclaration = Node /*
    | ClassLikeDeclaration
    | InterfaceDeclaration
    | TypeLiteralNode
    */
        ;

    using DeclarationWithTypeParameters = Node /*
    | DeclarationWithTypeParameterChildren
    | JSDocTypedefTag
    | JSDocCallbackTag
    | JSDocSignature
    */
        ;

    using DeclarationWithTypeParameterChildren = Node /*
    | SignatureDeclaration
    | ClassLikeDeclaration
    | InterfaceDeclaration
    | TypeAliasDeclaration
    | JSDocTemplateTag
    */
        ;

    struct ClassLikeDeclarationBase : NamedDeclaration, JSDocContainer
    {
        // kind: SyntaxKind::ClassDeclaration | SyntaxKind::ClassExpression;
        Identifier name;
        NodeArray<TypeParameterDeclaration> typeParameters;
        NodeArray<HeritageClause> heritageClauses;
        NodeArray<ClassElement> members;
    };

    struct ClassDeclaration : ClassLikeDeclarationBase, DeclarationStatement
    {
        // kind: SyntaxKind::ClassDeclaration;
        /** May be undefined in `export default class { ... }`. */
        Identifier name;
    };

    struct ClassExpression : ClassLikeDeclarationBase, PrimaryExpression
    {
        // kind: SyntaxKind::ClassExpression;
    };

    using ClassLikeDeclaration = Node /*
    | ClassDeclaration
    | ClassExpression
    */
        ;

    struct ClassElement : NamedDeclaration
    {
        any _classElementBrand;
        PropertyName name;
    };

    struct TypeElement : NamedDeclaration
    {
        any _typeElementBrand;
        PropertyName name;
        QuestionToken questionToken;
    };

    struct InterfaceDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::InterfaceDeclaration;
        Identifier name;
        NodeArray<TypeParameterDeclaration> typeParameters;
        NodeArray<HeritageClause> heritageClauses;
        NodeArray<TypeElement> members;
    };

    struct HeritageClause : Node
    {
        // kind: SyntaxKind::HeritageClause;
        NodeRef /*InterfaceDeclaration | ClassLikeDeclaration*/ parent;
        SyntaxKind token;
        NodeArray<ExpressionWithTypeArguments> types;
    };

    struct TypeAliasDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::TypeAliasDeclaration;
        Identifier name;
        NodeArray<TypeParameterDeclaration> typeParameters;
        TypeNode type;
    };

    struct EnumMember : NamedDeclaration, JSDocContainer
    {
        // kind: SyntaxKind::EnumMember;
        EnumDeclaration parent;
        // This does include ComputedPropertyName, but the parser will give an error
        // if it parses a ComputedPropertyName in an EnumMember
        PropertyName name;
        Expression initializer;
    };

    struct EnumDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::EnumDeclaration;
        Identifier name;
        NodeArray<EnumMember> members;
    };

    using ModuleName = Node /*
    | Identifier
    | StringLiteral
    */
        ;

    using ModuleBody = Node /*
    | NamespaceBody
    | JSDocNamespaceBody
    */
        ;

    /* @internal */
    struct AmbientModuleDeclaration : ModuleDeclaration
    {
        ModuleBlock body;
    };

    struct ModuleDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::ModuleDeclaration;
        Node /*ModuleBody | SourceFile*/ parent;
        ModuleName name;
        Node /*ModuleBody | JSDocNamespaceDeclaration*/ body;
    };

    using NamespaceBody = Node /*
    | ModuleBlock
    | NamespaceDeclaration
    */
        ;

    struct NamespaceDeclaration : ModuleDeclaration
    {
        Identifier name;
        NamespaceBody body;
    };

    using JSDocNamespaceBody = Node /*
    | Identifier
    | JSDocNamespaceDeclaration
    */
        ;

    struct JSDocNamespaceDeclaration : ModuleDeclaration
    {
        Identifier name;
        JSDocNamespaceBody body;
    };

    struct ModuleBlock : Node, Statement
    {
        // kind: SyntaxKind::ModuleBlock;
        ModuleDeclaration parent;
        NodeArray<Statement> statements;
    };

    using ModuleReference = Node /*
    | EntityName
    | ExternalModuleReference
    */
        ;

    /**
 * One of:
 * - import x = require("mod");
 * - import x = M.x;
 */
    struct ImportEqualsDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::ImportEqualsDeclaration;
        Node /*SourceFile | ModuleBlock*/ parent;
        Identifier name;
        boolean isTypeOnly;

        // 'EntityName' for an internal module reference, 'ExternalModuleReference' for an external
        // module reference.
        ModuleReference moduleReference;
    };

    struct ExternalModuleReference : Node
    {
        // kind: SyntaxKind::ExternalModuleReference;
        ImportEqualsDeclaration parent;
        Expression expression;
    };

    // In case of:
    // import "mod"  => importClause = undefined, moduleSpecifier = "mod"
    // In rest of the cases, module specifier is string literal corresponding to module
    // ImportClause information is shown at its declaration below.
    struct ImportDeclaration : Statement, JSDocContainer
    {
        // kind: SyntaxKind::ImportDeclaration;
        Node /*SourceFile | ModuleBlock*/ parent;
        ImportClause importClause;
        /** If this is not a StringLiteral it will be a grammar error. */
        Expression moduleSpecifier;
    };

    using NamedImportBindings = Node /*
    | NamespaceImport
    | NamedImports
    */
        ;

    using NamedExportBindings = Node /*
    | NamespaceExport
    | NamedExports
    */
        ;

    // In case of:
    // import d from "mod" => name = d, namedBinding = undefined
    // import * as ns from "mod" => name = undefined, namedBinding: NamespaceImport = { name: ns }
    // import d, * as ns from "mod" => name = d, namedBinding: NamespaceImport = { name: ns }
    // import { a, b as x } from "mod" => name = undefined, namedBinding: NamedImports = { elements: [{ name: a }, { name: x, propertyName: b}]}
    // import d, { a, b as x } from "mod" => name = d, namedBinding: NamedImports = { elements: [{ name: a }, { name: x, propertyName: b}]}
    struct ImportClause : NamedDeclaration
    {
        // kind: SyntaxKind::ImportClause;
        ImportDeclaration parent;
        boolean isTypeOnly;
        Identifier name; // Default binding
        NamedImportBindings namedBindings;
    };

    struct NamespaceImport : NamedDeclaration
    {
        // kind: SyntaxKind::NamespaceImport;
        ImportClause parent;
        Identifier name;
    };

    struct NamespaceExport : NamedDeclaration
    {
        // kind: SyntaxKind::NamespaceExport;
        ExportDeclaration parent;
        Identifier
    };

    struct NamespaceExportDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::NamespaceExportDeclaration name;
        Identifier name;
        /* @internal */ NodeArray<Decorator> decorators; // Present for use with reporting a grammar error
        /* @internal */ ModifiersArray modifiers;        // Present for use with reporting a grammar error
    };

    struct ExportDeclaration : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::ExportDeclaration;
        Node /*SourceFile | ModuleBlock*/ parent;
        boolean isTypeOnly;
        /** Will not be assigned in the case of `export * from "foo";` */
        NamedExportBindings exportClause;
        /** If this is not a StringLiteral it will be a grammar error. */
        Expression moduleSpecifier;
    };

    struct NamedImports : Node
    {
        // kind: SyntaxKind::NamedImports;
        ImportClause parent;
        NodeArray<ImportSpecifier> elements;
    };

    struct NamedExports : Node
    {
        // kind: SyntaxKind::NamedExports;
        ExportDeclaration parent;
        NodeArray<ExportSpecifier> elements;
    };

    using NamedImportsOrExports = Node /*NamedImports | NamedExports*/;

    struct ImportSpecifier : NamedDeclaration
    {
        // kind: SyntaxKind::ImportSpecifier;
        NamedImports parent;
        Identifier propertyName; // Name preceding "as" keyword (or undefined when "as" is absent)
        Identifier name;         // Declared name
    };

    struct ExportSpecifier : NamedDeclaration
    {
        // kind: SyntaxKind::ExportSpecifier;
        NamedExports parent;
        Identifier propertyName; // Name preceding "as" keyword (or undefined when "as" is absent)
        Identifier name;         // Declared name
    };

    using ImportOrExportSpecifier = Node /*
    | ImportSpecifier
    | ExportSpecifier
    */
        ;

    using TypeOnlyCompatibleAliasDeclaration = Node /*
    | ImportClause
    | ImportEqualsDeclaration
    | NamespaceImport
    | ImportOrExportSpecifier
    */
        ;

    /**
 * This is either an `export =` or an `export default` declaration.
 * Unless `isExportEquals` is set, this node was parsed as an `export default`.
 */
    struct ExportAssignment : DeclarationStatement, JSDocContainer
    {
        // kind: SyntaxKind::ExportAssignment;
        SourceFile parent;
        boolean isExportEquals;
        Expression expression;
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
        TypeNode type;
    };

    struct JSDocNameReference : Node
    {
        // kind: SyntaxKind::JSDocNameReference;
        EntityName name;
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
        TypeNode type;
    };

    struct JSDocNullableType : JSDocType
    {
        // kind: SyntaxKind::JSDocNullableType;
        TypeNode type;
    };

    struct JSDocOptionalType : JSDocType
    {
        // kind: SyntaxKind::JSDocOptionalType;
        TypeNode type;
    };

    struct JSDocFunctionType : JSDocType, SignatureDeclarationBase
    {
        // kind: SyntaxKind::JSDocFunctionType;
    };

    struct JSDocVariadicType : JSDocType
    {
        // kind: SyntaxKind::JSDocVariadicType;
        TypeNode type;
    };

    struct JSDocNamepathType : JSDocType
    {
        // kind: SyntaxKind::JSDocNamepathType;
        TypeNode type;
    };

    using JSDocTypeReferencingNode = Node /*
    | JSDocVariadicType
    | JSDocOptionalType
    | JSDocNullableType
    | JSDocNonNullableType
    */
        ;

    struct JSDoc : Node
    {
        // kind: SyntaxKind::JSDocComment;
        HasJSDoc parent;
        NodeArray<JSDocTag> tags;
        string comment;
    };

    struct JSDocTag : Node
    {
        Node /*JSDoc | JSDocTypeLiteral*/ parent;
        Identifier tagName;
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
            Node /*Identifier | PropertyAccessEntityNameExpression*/ expression;
        } _class;
    };

    struct JSDocImplementsTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocImplementsTag;
        struct _classArg : ExpressionWithTypeArguments
        {
            Node /*Identifier | PropertyAccessEntityNameExpression*/ expression;
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
        JSDoc parent;
        JSDocTypeExpression typeExpression;
    };

    struct JSDocThisTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocThisTag;
        JSDocTypeExpression typeExpression;
    };

    struct JSDocTemplateTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocTemplateTag;
        Node /*JSDocTypeExpression | undefined*/ constraint;
        NodeArray<TypeParameterDeclaration> typeParameters;
    };

    struct JSDocSeeTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocSeeTag;
        JSDocNameReference name;
    };

    struct JSDocReturnTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocReturnTag;
        JSDocTypeExpression typeExpression;
    };

    struct JSDocTypeTag : JSDocTag
    {
        // kind: SyntaxKind::JSDocTypeTag;
        JSDocTypeExpression typeExpression;
    };

    struct JSDocTypedefTag : JSDocTag, NamedDeclaration
    {
        // kind: SyntaxKind::JSDocTypedefTag;
        JSDoc parent;
        Node /*JSDocNamespaceDeclaration | Identifier*/ fullName;
        Identifier name;
        Node /*JSDocTypeExpression | JSDocTypeLiteral*/ typeExpression;
    };

    struct JSDocCallbackTag : JSDocTag, NamedDeclaration
    {
        // kind: SyntaxKind::JSDocCallbackTag;
        JSDoc parent;
        Node /*JSDocNamespaceDeclaration | Identifier*/ fullName;
        Identifier name;
        JSDocSignature typeExpression;
    };

    struct JSDocSignature : JSDocType, Declaration
    {
        // kind: SyntaxKind::JSDocSignature;
        std::vector<JSDocTemplateTag> typeParameters;
        std::vector<JSDocParameterTag> parameters;
        Node /*JSDocReturnTag | undefined*/ type;
    };

    struct JSDocPropertyLikeTag : JSDocTag, Declaration
    {
        JSDoc parent;
        EntityName name;
        JSDocTypeExpression typeExpression;
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
        TextRange range;
        CommentDirectiveType type,
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
        auto getPositionOfLineAndCharacter(number line, number character, boolean allowEdits = true) -> number;
    };

    struct RedirectInfo
    {
        /** Source file this redirects to. */
        SourceFileRef redirectTarget;
        /**
     * Source file for the duplicate package. This will not be used by the Program,
     * but we need to keep this around so we can watch for changes in underlying.
     */
        SourceFileRef unredirected;
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
        SourceFile file;
        number start;
        number length;
        string messageText;
        DiagnosticMessageChain messageChain;
    };

    struct Diagnostic : DiagnosticRelatedInformation
    {
        /** May store more in future. For now, this will simply be `true` to indicate when a diagnostic is an unused-identifier diagnostic. */
        std::vector<string> reportsUnnecessary;
        std::vector<string> reportsDeprecated;
        string source;
        std::vector<DiagnosticRelatedInformation> relatedInformation;
        /* @internal */ string /*keyof CompilerOptions*/ skippedOn ? ;
    };

    struct DiagnosticWithLocation : Diagnostic
    {
        SourceFile file;
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
        PackageId packageId;
    };

    struct ResolvedTypeReferenceDirective
    {
        // True if the type declaration file was found in a primary lookup location
        boolean primary;
        // The location of the .d.ts file we located, or undefined if resolution failed
        string resolvedFileName;
        PackageId packageId;
        /** True if `resolvedFileName` comes from `node_modules`. */
        boolean isExternalLibraryImport;
    };

    struct PatternAmbientModule
    {
        Node pattern;
        Symbol symbol;
    };

    struct SourceFile : Declaration
    {
        // kind: SyntaxKind::SourceFile;
        NodeArray<Statement> statements;
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
        /* @internal */ RedirectInfo redirectInfo;

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
        /* @internal */ Node externalModuleIndicator;
        // The first node that causes this file to be a CommonJS module
        /* @internal */ Node commonJsModuleIndicator;
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
        /* @internal */ NodeArray<Node> moduleAugmentations;
        /* @internal */ std::vector<PatternAmbientModule> patternAmbientModules;
        /* @internal */ std::vector<string> ambientModuleNames;
        /* @internal */ CheckJsDirective checkJsDirective;
        /* @internal */ string version;
        /* @internal */ std::map<string, string> pragmas;
        /* @internal */ string localJsxNamespace;
        /* @internal */ string localJsxFragmentNamespace;
        /* @internal */ EntityName localJsxFactory;
        /* @internal */ EntityName localJsxFragmentFactory;

        /* @internal */ ExportedModulesFromDeclarationEmit exportedModulesFromDeclarationEmit;
    };

} // namespace ver2

#endif // NEW_PARSER_TYPES_H