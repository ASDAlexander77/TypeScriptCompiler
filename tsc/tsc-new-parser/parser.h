#ifndef PARSER_H
#define PARSER_H

#include "undefined.h"
#include "enums.h"
#include "types.h"
#include "scanner.h"
#include "debug.h"

#include <memory>

struct Node;

template <typename T>
using NodeFuncT = std::function<T(Node)>;

template <typename T>
using NodeWithParentFuncT = std::function<T(Node, Node)>;

typedef std::function<Node(SyntaxKind, number, number)> NodeCreateFunc;

typedef std::function<void(number, number, DiagnosticMessage)> PragmaDiagnosticReporter;

template <SyntaxKind kind>
using Token = Node;

struct Node;

template <typename T>
struct NodeArray
{
    TextRange range;

    NodeArray() = default;
    NodeArray(std::initializer_list<T> il) : items(il) {}
    NodeArray(undefined_t) : items() {}

    std::vector<T> items;
    boolean isMissingList;

    inline auto operator [](size_t i) const -> T
    {
        return items[i];
    }

    inline auto operator [](size_t i) -> T&
    {
        return items[i];
    }

    inline auto operator !() -> boolean
    {
        return items.size() == 0;
    }    

    inline auto clear() -> void {
        return items.clear();
    }

    inline auto size() -> size_t {
        return items.size();
    }

    auto push_back(Node node) -> void
    {
        items.push_back(node);
    }       

    TextRange* operator ->()
    {
        return &range;
    }

    inline auto operator ==(undefined_t) -> boolean
    {
        return items.size() == 0;
    } 

    inline auto operator !=(undefined_t) -> boolean
    {
        return items.size() != 0;
    }        

    auto begin() -> decltype(items.begin())
    {
        return items.begin();
    }

    auto end() -> decltype(items.end())
    {
        return items.end();
    }
};

template <typename T>
using NodeArrayFuncT = std::function<T(NodeArray<T>)>;

template <typename T>
using NodeWithParentArrayFuncT = std::function<T(NodeArray<T>, Node)>;

struct Decorator;
typedef NodeArray<Decorator> DecoratorsArray;
typedef Node Modifier;
typedef NodeArray<Modifier> ModifiersArray;


#define CLASS_DATA_BASE(x, b) struct x##Data : b##Data { using b##Data::b##Data;

#define CLASS_DATA(x) CLASS_DATA_BASE(x, Node)

#define CLASS_DATA_END(x) };                \
    struct x : BaseNode {                   \
        x() {}                              \
        x(undefined_t) {}                   \
        x(Node node) : BaseNode(node) { node.data = std::make_shared<x##Data>(*node.data); }   \
                                            \
        x##Data* operator->()               \
        {                                   \
            return static_cast<x##Data*>(node.operator->());   \
        }                                   \
        operator TextRange()                \
        {                                   \
            return *(TextRange*)node.data.get(); \
        }                                   \
        inline operator SyntaxKind()        \
        {                                   \
            return node.data->kind;         \
        }                                   \
        inline auto operator==(undefined_t) \
        {                                   \
            return !node.data;              \
        }                                   \
        inline auto operator!=(undefined_t) \
        {                                   \
            return !!node.data;             \
        }                                   \
    };  

struct NodeHolder
{
    Node* value;
    operator Node();
};

struct NodeData : TextRange
{   
    NodeData() = default;

    SyntaxKind kind;
    NodeFlags flags;
    string text;
    TransformFlags transformFlags;
    DecoratorsArray decorators;
    ModifiersArray modifiers;
    SyntaxKind originalKeywordKind;
    NodeHolder parent;
    number jsdocDotPos;

    NodeArray<Node> children;

    NodeData(number start, number end) : TextRange{start, end} {};

    virtual ~NodeData() {}
};

struct Node
{
    std::shared_ptr<NodeData> data;

    Node() {};
    Node(undefined_t) {};
    Node(SyntaxKind kind, number start, number end) : data(std::make_shared<NodeData>(kind, start, end)) {};
    template <typename T>
    Node(NodeArray<T> values) : data(std::make_shared<NodeData>(SyntaxKind::Array, -1, -1)) { data->children = (NodeArray<Node>) values; };

    NodeData* operator->()
    {
        return data.get();
    }

    operator TextRange()
    {
        return *(TextRange*)data.get();
    }

    template <typename T> 
    auto as() -> T
    {
        return T(*this);
    }

    template <typename T> 
    auto asMutable() -> T
    {
        return T(*this);
    }    

    operator bool()
    {
        return (bool)data;
    }

    operator NodeArray<Node>()
    {
        return data->children;
    }
    
    inline operator SyntaxKind()
    {
        return data->kind;
    }

    auto operator=(undefined_t) -> Node&
    {
        data->kind = SyntaxKind::Unknown;
        data->children.clear();
        return *this;
    }    

    auto operator=(NodeArray<Node> values) -> Node&
    {
        data->kind = SyntaxKind::ArrayType;
        data->children = values;
        return *this;
    }

    auto operator||(Node rhs) -> Node
    {
        if (operator bool())
        {
            return *this;
        }

        return rhs;
    }

    auto size() -> number
    {
        return data->children.size();
    }

    auto operator[](number i) const -> Node
    {
        return data->children[i];
    }    

    auto operator[](number i) -> Node&
    {
        return data->children[i];
    }    

    auto push_back(Node node) -> void
    {
        data->kind = SyntaxKind::ArrayType;
        data->children.push_back(node);
    }    

    auto operator==(undefined_t)
    {
        return !data;
    }

    auto operator!=(undefined_t)
    {
        return !!data;
    }
};

NodeHolder::operator Node()
{
    return *value;
}

struct BaseNode
{
    Node node;
    BaseNode() : node(Node()) {}
    BaseNode(Node node) : node(node) {}

    inline operator Node()
    {
        return node;
    }

    inline operator SyntaxKind()
    {
        return node->kind;
    }

    inline bool operator !()
    {
        return !static_cast<bool>(node);
    }

    inline NodeData* operator->()
    {
        return node.operator->();
    }    

    auto operator||(Node rhs) -> Node
    {
        if (static_cast<bool>(node))
        {
            return *this;
        }

        return rhs;
    }

    template <typename T> 
    auto as() -> T
    {
        return T(node);
    }

    template <typename T> 
    auto asMutable() -> T
    {
        return T(node);
    }          
};

static auto isArray(Node &node) -> boolean
{
    return node.data->kind == SyntaxKind::Array;
}

typedef SyntaxKind PrefixUnaryOperator, PostfixUnaryOperator;

typedef Node Identifier, PropertyName, PrivateIdentifier, LiteralExpression, EntityName, Expression, IndexSignatureDeclaration,
    TypeElement, UnaryExpression, UpdateExpression, LeftHandSideExpression, MemberExpression, JsxText, JsxChild, JsxTagNameExpression,
    JsxClosingFragment, PrimaryExpression, FunctionExpression, Statement, CaseOrDefaultClause, ArrayBindingElement,
    ObjectBindingPattern, ArrayBindingPattern, FunctionDeclaration, ConstructorDeclaration, AccessorDeclaration, ClassElement, ClassExpression,
    ModuleBlock, SuperExpression, ThisExpression, PseudoBigInt, MissingDeclaration, JsonObjectExpressionStatement, BindingName,
    CallSignatureDeclaration, MethodSignature, GetAccessorDeclaration, SetAccessorDeclaration, ConstructSignatureDeclaration, IndexSignatureDeclaration,
    MemberName, ElementAccessChain, CallChain, NewExpression, ConciseBody,
    Expression, OmittedExpression, NonNullChain, SemicolonClassElement, EmptyStatement, ForInitializer, ContinueStatement, 
    BreakStatement, DebuggerStatement, ModuleName, ModuleBody, ModuleReference, NamedImportBindings, ImportSpecifier, NamedImports,
    NamedExportBindings, ExportSpecifier, NamedExports, DestructuringAssignment, PropertyDescriptorAttributes, CallBinding, Declaration;

typedef Node FalseLiteral, TrueLiteral, NullLiteral, BooleanLiteral, NumericLiteral, BigIntLiteral, StringLiteral, PropertyNameLiteral, RegularExpressionLiteral, 
    ObjectLiteralElementLike, TemplateLiteral, NoSubstitutionTemplateLiteral;

typedef Node ThisTypeNode, UnionTypeNode, IntersectionTypeNode;

typedef Node BinaryOperatorToken, QuestionDotToken, EndOfFileToken, LiteralToken, DotDotDotToken, QuestionToken, PlusToken, MinusToken,
    AsteriskToken, EqualsGreaterThanToken, ColonToken, ExclamationToken, EqualsToken;

typedef Node ReadonlyKeyword, AssertsKeyword, AwaitKeyword;

typedef Node JSDocAllType, JSDocUnknownType, JSDocNonNullableType, JSDocNullableType, JSDocOptionalType, JSDocVariadicType, JSDocNamepathType,
    JSDocAuthorTag, JSDocClassTag, JSDocPublicTag, JSDocPrivateTag, JSDocProtectedTag, JSDocReadonlyTag, JSDocUnknownTag, JSDocDeprecatedTag,
    JSDocParameterTag, JSDocPropertyTag; 

typedef Node JsxSelfClosingElement, JsxOpeningFragment, JsxAttributeLike, JsxTagNamePropertyAccess;

typedef Node UnparsedPrologue, UnparsedSyntheticReference, UnparsedSourceText, UnparsedSource, UnparsedPrepend, UnparsedTextLike, InputFiles;

typedef Node SyntheticExpression, SyntaxList, NotEmittedStatement, EndOfDeclarationMarker, SyntheticReferenceExpression, MergeDeclarationMarker, Bundle;

typedef Node PrologueDirective;

template<typename T>
using AssignmentExpression = Node;

template<typename T>
using Push = Node;

template<typename T>
using VisitResult = Node;


CLASS_DATA(QualifiedName)
    Node left;
    Node right;
CLASS_DATA_END(QualifiedName)

CLASS_DATA(MethodDeclaration)
    Node name;
CLASS_DATA_END(MethodDeclaration)

CLASS_DATA(TypeParameterDeclaration)
    Node name;
    Node constraint;
    Node _default;
    Node expression;
CLASS_DATA_END(TypeParameterDeclaration)

CLASS_DATA(PropertyAssignment)
    Node name;
    Node questionToken;
    Node exclamationToken;
    Node initializer;
CLASS_DATA_END(PropertyAssignment)

CLASS_DATA_BASE(ShorthandPropertyAssignment, PropertyAssignment)
    Node exclamationToken;
    Node equalsToken;
    Node objectAssignmentInitializer;
CLASS_DATA_END(ShorthandPropertyAssignment)

CLASS_DATA(SpreadAssignment)
    Node expression;
CLASS_DATA_END(SpreadAssignment)

CLASS_DATA(ParameterDeclaration)
    Node dotDotDotToken;
    Node name;
    Node questionToken;
    Node type;
    Node initializer;
CLASS_DATA_END(ParameterDeclaration)

CLASS_DATA(PropertyDeclaration)
    Node name;
    Node questionToken;
    Node exclamationToken;
    Node type;
    Node initializer;
CLASS_DATA_END(PropertyDeclaration)

CLASS_DATA(PropertySignature)
    Node name;
    Node questionToken;
    Node type;
    Node initializer;
CLASS_DATA_END(PropertySignature)

CLASS_DATA(VariableDeclaration)
    Node name;
    Node exclamationToken;
    Node type;
    Node initializer;
CLASS_DATA_END(VariableDeclaration)

CLASS_DATA(BindingElement)
    Node dotDotDotToken;
    Node propertyName;
    Node name;
    Node initializer;
CLASS_DATA_END(BindingElement)

CLASS_DATA(SignatureDeclaration)
    Node typeParameters;
    Node parameters;
    Node type;
CLASS_DATA_END(SignatureDeclaration)

CLASS_DATA(FunctionLikeDeclaration)
    Node asteriskToken;
    Node name;
    Node questionToken;
    Node exclamationToken;
    Node typeParameters;
    Node parameters;
    Node type;
    Node equalsGreaterThanToken;
    Node body;
CLASS_DATA_END(FunctionLikeDeclaration)

CLASS_DATA(ArrowFunction)
    Node equalsGreaterThanToken;
CLASS_DATA_END(ArrowFunction)


CLASS_DATA(TypeReferenceNode)
    Node typeArguments;
    Node typeName;
CLASS_DATA_END(TypeReferenceNode)

CLASS_DATA(TypePredicateNode)
    Node assertsModifier;
    Node parameterName;
    Node type;
CLASS_DATA_END(TypePredicateNode)

CLASS_DATA(TypeQueryNode)
    Node exprName;
CLASS_DATA_END(TypeQueryNode)

CLASS_DATA(TypeLiteralNode)
    Node members;
CLASS_DATA_END(TypeLiteralNode)

CLASS_DATA(TypeNode)
    Node type;
CLASS_DATA_END(TypeNode)

CLASS_DATA(ArrayTypeNode)
    Node elementType;
CLASS_DATA_END(ArrayTypeNode)

CLASS_DATA(TupleTypeNode)
    Node elements;
CLASS_DATA_END(TupleTypeNode)

CLASS_DATA(UnionOrIntersectionTypeNode)
    Node types;
CLASS_DATA_END(UnionOrIntersectionTypeNode)

CLASS_DATA(ConditionalTypeNode)
    Node checkType;
    Node extendsType;
    Node trueType;
    Node falseType;
CLASS_DATA_END(ConditionalTypeNode)

CLASS_DATA(InferTypeNode)
    Node typeParameter;
CLASS_DATA_END(InferTypeNode)

CLASS_DATA(ImportTypeNode)
    Node argument;
    Node qualifier;
    Node typeArguments;
CLASS_DATA_END(ImportTypeNode)

typedef TypeNode ParenthesizedTypeNode;
typedef TypeNode TypeOperatorNode;

CLASS_DATA(IndexedAccessTypeNode)
    Node objectType;
    Node indexType;
CLASS_DATA_END(IndexedAccessTypeNode)

CLASS_DATA(MappedTypeNode)
    Node readonlyToken;
    Node typeParameter;
    Node nameType;
    Node questionToken;
    Node type;
CLASS_DATA_END(MappedTypeNode)

CLASS_DATA(LiteralTypeNode)
    Node literal;
CLASS_DATA_END(LiteralTypeNode)

CLASS_DATA(NamedTupleMember)
    Node dotDotDotToken;
    Node name;
    Node questionToken;
    Node type;
CLASS_DATA_END(NamedTupleMember)

CLASS_DATA(BindingPattern)
    Node elements;
CLASS_DATA_END(BindingPattern)

CLASS_DATA(ArrayLiteralExpression)
    Node elements;
CLASS_DATA_END(ArrayLiteralExpression)

CLASS_DATA(ObjectLiteralExpression)
    Node properties;
CLASS_DATA_END(ObjectLiteralExpression)

CLASS_DATA(PropertyAccessExpression)
    Node expression;
    Node questionDotToken;
    Node name;
CLASS_DATA_END(PropertyAccessExpression)

typedef PropertyAccessExpression PropertyAccessChain;

CLASS_DATA(ElementAccessExpression)
    Node expression;
    Node questionDotToken;
    Node argumentExpression;
CLASS_DATA_END(ElementAccessExpression)

CLASS_DATA(CallExpression)
    Node expression;
    Node questionDotToken;
    Node typeArguments;
    Node arguments;
CLASS_DATA_END(CallExpression)

CLASS_DATA(TaggedTemplateExpression)
    Node tag;
    Node questionDotToken;
    Node typeArguments;
    Node _template;
CLASS_DATA_END(TaggedTemplateExpression)

CLASS_DATA(TypeAssertion)
    Node type;
    Node expression;
CLASS_DATA_END(TypeAssertion)

CLASS_DATA(ParenthesizedExpression)
    Node expression;
CLASS_DATA_END(ParenthesizedExpression)

CLASS_DATA(DeleteExpression)
    Node expression;
CLASS_DATA_END(DeleteExpression)

CLASS_DATA(TypeOfExpression)
    Node expression;
CLASS_DATA_END(TypeOfExpression)

CLASS_DATA(VoidExpression)
    Node expression;
CLASS_DATA_END(VoidExpression)

CLASS_DATA(PrefixUnaryExpression)
    Node operand;
CLASS_DATA_END(PrefixUnaryExpression)

CLASS_DATA(YieldExpression)
    Node asteriskToken;
    Node expression;
CLASS_DATA_END(YieldExpression)

CLASS_DATA(AwaitExpression)
    Node expression;
CLASS_DATA_END(AwaitExpression)

CLASS_DATA(PostfixUnaryExpression)
    Node operand;
CLASS_DATA_END(PostfixUnaryExpression)

CLASS_DATA(BinaryExpression)
    Node left;
    Node operatorToken;
    Node right;
CLASS_DATA_END(BinaryExpression)

CLASS_DATA(AsExpression)
    Node expression;
    Node type;
CLASS_DATA_END(AsExpression)

CLASS_DATA(NonNullExpression)
    Node expression;
CLASS_DATA_END(NonNullExpression)

CLASS_DATA(MetaProperty)
    Node name;
CLASS_DATA_END(MetaProperty)

CLASS_DATA(ConditionalExpression)
    Node condition;
    Node questionToken;
    Node whenTrue;
    Node colonToken;
    Node whenFalse;
CLASS_DATA_END(ConditionalExpression)

CLASS_DATA(SpreadElement)
    Node expression;
CLASS_DATA_END(SpreadElement)

CLASS_DATA(PartiallyEmittedExpression)
    Node expression;
CLASS_DATA_END(PartiallyEmittedExpression)

CLASS_DATA(Block)
    Node statements;
CLASS_DATA_END(Block)

CLASS_DATA(SourceFile)
    Node statements;
    Node endOfFileToken;
    Node externalModuleIndicator;
    Node commonJsModuleIndicator;

    // extra fields
    std::vector<FileReference> referencedFiles;
    std::vector<FileReference> typeReferenceDirectives;
    std::vector<FileReference> libReferenceDirectives;
    std::vector<AmdDependency> amdDependencies;
    LanguageVariant languageVariant;
    boolean isDeclarationFile;

    std::map<string, string> renamedDependencies;
    boolean hasNoDefaultLib;
    ScriptTarget languageVersion;
    ScriptKind scriptKind;

    std::map<string, string> pragmas;

    // stats
    std::vector<CommentDirective> commentDirectives;
    string fileName;
    string text;
    number nodeCount;
    number identifierCount;
    std::map<string, string> identifiers;

    std::vector<DiagnosticWithDetachedLocation> parseDiagnostics;
    std::vector<DiagnosticWithDetachedLocation> bindDiagnostics;
    std::vector<DiagnosticWithDetachedLocation> bindSuggestionDiagnostics;
    std::vector<DiagnosticWithDetachedLocation> jsDocDiagnostics;
CLASS_DATA_END(SourceFile)

typedef SourceFile JsonSourceFile;

CLASS_DATA(VariableStatement)
    Node declarationList;
CLASS_DATA_END(VariableStatement)

CLASS_DATA(VariableDeclarationList)
    Node declarations;
CLASS_DATA_END(VariableDeclarationList)

CLASS_DATA(ExpressionStatement)
    Node expression;
CLASS_DATA_END(ExpressionStatement)

CLASS_DATA(IfStatement)
    Node expression;
    Node thenStatement;
    Node elseStatement;
CLASS_DATA_END(IfStatement)

CLASS_DATA(DoStatement)
    Node statement;
    Node expression;
CLASS_DATA_END(DoStatement)

CLASS_DATA(WhileStatement)
    Node expression;
    Node statement;
CLASS_DATA_END(WhileStatement)

CLASS_DATA(ForStatement)
    Node initializer;
    Node condition;
    Node incrementor;
    Node statement;
CLASS_DATA_END(ForStatement)

CLASS_DATA(ForInStatement)
    Node initializer;
    Node expression;
    Node statement;
CLASS_DATA_END(ForInStatement)

CLASS_DATA(ForOfStatement)
    Node awaitModifier;
    Node initializer;
    Node expression;
    Node statement;
CLASS_DATA_END(ForOfStatement)

CLASS_DATA(BreakOrContinueStatement)
    Node label;
CLASS_DATA_END(BreakOrContinueStatement)

CLASS_DATA(ReturnStatement)
    Node expression;
CLASS_DATA_END(ReturnStatement)

CLASS_DATA(WithStatement)
    Node expression;
    Node statement;
CLASS_DATA_END(WithStatement)

CLASS_DATA(SwitchStatement)
    Node expression;
    Node caseBlock;
CLASS_DATA_END(SwitchStatement)

CLASS_DATA(CaseBlock)
    Node clauses;
CLASS_DATA_END(CaseBlock)

CLASS_DATA(CaseClause)
    Node expression;
    Node statements;
CLASS_DATA_END(CaseClause)

CLASS_DATA(DefaultClause)
    Node statements;
CLASS_DATA_END(DefaultClause)

CLASS_DATA(LabeledStatement)
    Node label;
    Node statement;
CLASS_DATA_END(LabeledStatement)

CLASS_DATA(ThrowStatement)
    Node expression;
CLASS_DATA_END(ThrowStatement)

CLASS_DATA(TryStatement)
    Node tryBlock;
    Node catchClause;
    Node finallyBlock;
CLASS_DATA_END(TryStatement)

CLASS_DATA(CatchClause)
    Node variableDeclaration;
    Node block;
CLASS_DATA_END(CatchClause)

CLASS_DATA(Decorator)
    Node expression;
CLASS_DATA_END(Decorator)

CLASS_DATA(ClassLikeDeclaration)
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
CLASS_DATA_END(ClassLikeDeclaration)

CLASS_DATA(InterfaceDeclaration)
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
CLASS_DATA_END(InterfaceDeclaration)

CLASS_DATA(ClassDeclaration)
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
CLASS_DATA_END(ClassDeclaration)

CLASS_DATA(TypeAliasDeclaration)
    Node name;
    Node typeParameters;
    Node type;
CLASS_DATA_END(TypeAliasDeclaration)

CLASS_DATA(EnumDeclaration)
    Node name;
    Node members;
CLASS_DATA_END(EnumDeclaration)

CLASS_DATA(EnumMember)
    Node name;
    Node initializer;
CLASS_DATA_END(EnumMember)

CLASS_DATA(ModuleDeclaration)
    Node name;
    Node body;
CLASS_DATA_END(ModuleDeclaration)

CLASS_DATA(ImportEqualsDeclaration)
    Node name;
    Node moduleReference;
CLASS_DATA_END(ImportEqualsDeclaration)

CLASS_DATA(ImportDeclaration)
    Node importClause;
    Node moduleSpecifier;
CLASS_DATA_END(ImportDeclaration)

CLASS_DATA(ImportClause)
    Node name;
    Node namedBindings;
CLASS_DATA_END(ImportClause)

CLASS_DATA(NamespaceExportDeclaration)
    Node name;
CLASS_DATA_END(NamespaceExportDeclaration)

CLASS_DATA(NamespaceImport)
    Node name;
CLASS_DATA_END(NamespaceImport)

CLASS_DATA(NamespaceExport)
    Node name;
CLASS_DATA_END(NamespaceExport)

CLASS_DATA(NamedImportsOrExports)
    Node elements;
CLASS_DATA_END(NamedImportsOrExports)

CLASS_DATA(ExportDeclaration)
    Node exportClause;
    Node moduleSpecifier;
CLASS_DATA_END(ExportDeclaration)

CLASS_DATA(ImportOrExportSpecifier)
    Node propertyName;
    Node name;
CLASS_DATA_END(ImportOrExportSpecifier)

CLASS_DATA(ExportAssignment)
    Node expression;
CLASS_DATA_END(ExportAssignment)

CLASS_DATA(TemplateExpression)
    Node head;
    Node templateSpans;
CLASS_DATA_END(TemplateExpression)

CLASS_DATA(TemplateSpan)
    Node expression;
    Node literal;
CLASS_DATA_END(TemplateSpan)

CLASS_DATA(TemplateHead)
CLASS_DATA_END(TemplateHead)

CLASS_DATA(TemplateMiddle)
CLASS_DATA_END(TemplateMiddle)

CLASS_DATA(TemplateTail)
CLASS_DATA_END(TemplateTail)

CLASS_DATA(TemplateLiteralTypeNode)
    Node head;
    Node templateSpans;
CLASS_DATA_END(TemplateLiteralTypeNode)

CLASS_DATA(TemplateLiteralTypeSpan)
    Node type;
    Node literal;
CLASS_DATA_END(TemplateLiteralTypeSpan)

CLASS_DATA(ComputedPropertyName)
    Node expression;
CLASS_DATA_END(ComputedPropertyName)

CLASS_DATA(HeritageClause)
    Node types;
CLASS_DATA_END(HeritageClause)

CLASS_DATA(ExpressionWithTypeArguments)
    Node expression;
    Node typeArguments;
CLASS_DATA_END(ExpressionWithTypeArguments)

CLASS_DATA(ExternalModuleReference)
    Node expression;
CLASS_DATA_END(ExternalModuleReference)

CLASS_DATA(CommaListExpression)
    Node elements;
CLASS_DATA_END(CommaListExpression)

CLASS_DATA(JsxElement)
    Node openingElement;
    Node children;
    Node closingElement;
CLASS_DATA_END(JsxElement)

CLASS_DATA(JsxFragment)
    Node openingFragment;
    Node children;
    Node closingFragment;
CLASS_DATA_END(JsxFragment)

CLASS_DATA(JsxOpeningLikeElement)
    Node tagName;
    Node typeArguments;
    Node attributes;
CLASS_DATA_END(JsxOpeningLikeElement)

typedef JsxOpeningLikeElement JsxOpeningElement;

CLASS_DATA(JsxAttributes)
    Node properties;
CLASS_DATA_END(JsxAttributes)

CLASS_DATA(JsxAttribute)
    Node name;
    Node initializer;
CLASS_DATA_END(JsxAttribute)

CLASS_DATA(JsxSpreadAttribute)
    Node expression;
CLASS_DATA_END(JsxSpreadAttribute)

CLASS_DATA(JsxExpression)
    Node dotDotDotToken;
    Node expression;
CLASS_DATA_END(JsxExpression)

CLASS_DATA(JsxClosingElement)
    Node tagName;
CLASS_DATA_END(JsxClosingElement)

typedef TypeNode OptionalTypeNode, RestTypeNode, JSDocTypeExpression, JSDocNonNullableTypeNode, JSDocNullableTypeNode, JSDocOptionalTypeNode, JSDocVariadicTypeNode;

CLASS_DATA(JSDocFunctionType)
    Node parameters;
    Node type;
CLASS_DATA_END(JSDocFunctionType)

CLASS_DATA(JSDoc)
    Node tags;
CLASS_DATA_END(JSDoc)

CLASS_DATA(JSDocSeeTag)
    Node tagName;
    Node name;
CLASS_DATA_END(JSDocSeeTag)

CLASS_DATA(JSDocNameReference)
    Node name;
CLASS_DATA_END(JSDocNameReference)

CLASS_DATA(JSDocTag)
    Node tagName;
CLASS_DATA_END(JSDocTag)

CLASS_DATA(JSDocPropertyLikeTag)
    Node isNameFirst;
    Node name;
    Node typeExpression;
CLASS_DATA_END(JSDocPropertyLikeTag)

CLASS_DATA(JSDocImplementsTag)
    Node _class;
CLASS_DATA_END(JSDocImplementsTag)

CLASS_DATA(JSDocAugmentsTag)
    Node _class;
CLASS_DATA_END(JSDocAugmentsTag)

CLASS_DATA(JSDocTemplateTag)
    Node constraint;
    Node typeParameters;
CLASS_DATA_END(JSDocTemplateTag)

CLASS_DATA(JSDocTypedefTag)
    Node typeExpression;
    Node fullName;
CLASS_DATA_END(JSDocTypedefTag)

CLASS_DATA(JSDocCallbackTag)
    Node fullName;
    Node typeExpression;
CLASS_DATA_END(JSDocCallbackTag)

CLASS_DATA(JSDocReturnTag)
    Node typeExpression;
CLASS_DATA_END(JSDocReturnTag)

CLASS_DATA(JSDocTypeTag)
    Node typeExpression;
CLASS_DATA_END(JSDocTypeTag)

CLASS_DATA(JSDocThisTag)
    Node typeExpression;
CLASS_DATA_END(JSDocThisTag)

CLASS_DATA(JSDocEnumTag)
    Node typeExpression;
CLASS_DATA_END(JSDocEnumTag)

CLASS_DATA(JSDocSignature)
    Node typeParameters;
    Node parameters;
    Node type;
CLASS_DATA_END(JSDocSignature)

CLASS_DATA(JSDocTypeLiteral)
    Node jsDocPropertyTags;
CLASS_DATA_END(JSDocTypeLiteral)

CLASS_DATA(JSDocContainer)
    Node jsDocCache;
CLASS_DATA_END(JSDocContainer)

CLASS_DATA(LiteralLikeNode)
    string text;
    boolean isUnterminated;
    boolean hasExtendedUnicodeEscape;
CLASS_DATA_END(LiteralLikeNode)

CLASS_DATA_BASE(TemplateLiteralLikeNode, LiteralLikeNode)
    string rawText;
    /* @internal */
    TokenFlags templateFlags;
CLASS_DATA_END(TemplateLiteralLikeNode)

CLASS_DATA(FunctionOrConstructorTypeNode)
    TypeNode type;
    Node parameters;
CLASS_DATA_END(FunctionOrConstructorTypeNode)

typedef FunctionOrConstructorTypeNode FunctionTypeNode;
typedef FunctionOrConstructorTypeNode ConstructorTypeNode;

struct DiagnosticWithLocation : Diagnostic {
    SourceFile file;
};

struct NodeWithDiagnostics
{ 
    Node node;
    std::vector<Diagnostic> diagnostics;
};

namespace ts
{
    auto processCommentPragmas(SourceFile context, string sourceText) -> void;
    auto processPragmasIntoFields(SourceFile context, PragmaDiagnosticReporter reportDiagnostic) -> void;
    auto isExternalModule(SourceFile file) -> boolean;
    auto tagNamesAreEquivalent(JsxTagNameExpression lhs, JsxTagNameExpression rhs) -> boolean;

    namespace IncrementalParser {

        struct IncrementalElement : TextRange {
            Node parent;
            boolean intersectsChange;
            number length;
            std::vector<Node> _children;
        };

        struct IncrementalNode : Node, IncrementalElement {
            boolean hasBeenIncrementallyParsed;
        };

        struct IncrementalNodeArray : NodeArray<IncrementalNode>, IncrementalElement {
            number length;
        };

        // Allows finding nodes in the source file at a certain position in an efficient manner.
        // The implementation takes advantage of the calling pattern it knows the parser will
        // make in order to optimize finding nodes as quickly as possible.
        struct SyntaxCursor {
            SyntaxCursor() = default;

            std::function<IncrementalNode(number)> currentNode;
        };

        auto createSyntaxCursor(SourceFile sourceFile) -> SyntaxCursor;
    }
}

#endif // PARSER_H