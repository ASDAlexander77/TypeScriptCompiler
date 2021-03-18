#ifndef PARSER_H
#define PARSER_H

#include "scanner.h"

struct Node;

typedef std::vector<Node> NodeArray;

template <typename T>
using NodeFuncT = std::function<T(Node)>;

template <typename T>
using NodeWithParentFuncT = std::function<T(Node, Node)>;

template <typename T>
using NodeArrayFuncT = std::function<T(NodeArray)>;

template <typename T>
using NodeWithParentArrayFuncT = std::function<T(NodeArray, Node)>;

typedef std::function<Node(SyntaxKind, number, number)> NodeCreateFunc;

enum class SignatureFlags : number {
    None = 0,
    Yield = 1 << 0,
    Await = 1 << 1,
    Type  = 1 << 2,
    IgnoreMissingOpenBrace = 1 << 4,
    JSDoc = 1 << 5,
};

enum class SpeculationKind : number {
    TryParse,
    Lookahead,
    Reparse
};

struct Node
{
    SyntaxKind kind;

    NodeArray decorators;
    NodeArray modifiers;

    template <typename T> 
    auto as() -> T
    {
        return T();
    }

    operator bool()
    {
        return this->kind != SyntaxKind::Unknown;
    }

    Node operator||(Node rhs)
    {
        if (operator bool())
        {
            return *this;
        }

        return rhs;
    }
};

struct QualifiedName
{
    Node left;
    Node right;
};

struct TypeParameterDeclaration
{
    Node name;
    Node constraint;
    Node _default;
    Node expression;
};

struct ShorthandPropertyAssignment
{
    Node name;
    Node questionToken;
    Node exclamationToken;
    Node equalsToken;
    Node objectAssignmentInitializer;
};

struct SpreadAssignment
{
    Node expression;
};

struct ParameterDeclaration
{
    Node dotDotDotToken;
    Node name;
    Node questionToken;
    Node type;
    Node initializer;
};

struct PropertyDeclaration
{
    Node name;
    Node questionToken;
    Node exclamationToken;
    Node type;
    Node initializer;
};

struct PropertySignature
{
    Node name;
    Node questionToken;
    Node type;
    Node initializer;
};

struct PropertyAssignment
{
    Node name;
    Node questionToken;
    Node initializer;
};

struct VariableDeclaration
{
    Node name;
    Node exclamationToken;
    Node type;
    Node initializer;
};

struct BindingElement
{
    Node dotDotDotToken;
    Node propertyName;
    Node name;
    Node initializer;
};

struct SignatureDeclaration
{
    Node typeParameters;
    Node parameters;
    Node type;
};

struct FunctionLikeDeclaration
{
    Node asteriskToken;
    Node name;
    Node questionToken;
    Node exclamationToken;
    Node typeParameters;
    Node parameters;
    Node type;
    Node equalsGreaterThanToken;
    Node body;
};

struct ArrowFunction
{
    Node equalsGreaterThanToken;
};


struct TypeReferenceNode
{
    Node typeArguments;
    Node typeName;
};

struct TypePredicateNode
{
    Node assertsModifier;
    Node parameterName;
    Node type;
};

struct TypeQueryNode
{
    Node exprName;
};

struct TypeLiteralNode
{
    Node members;
};

struct ArrayTypeNode
{
    Node elementType;
};

struct TupleTypeNode
{
    Node elements;
};

struct UnionOrIntersectionTypeNode
{
    Node types;
};

struct ConditionalTypeNode
{
    Node checkType;
    Node extendsType;
    Node trueType;
    Node falseType;
};

struct InferTypeNode
{
    Node typeParameter;
};

struct ImportTypeNode
{
    Node argument;
    Node qualifier;
    Node typeArguments;
};

struct ParenthesizedTypeNode
{
    Node type;
};

struct TypeOperatorNode
{
    Node type;
};

struct IndexedAccessTypeNode
{
    Node objectType;
    Node indexType;
};

struct MappedTypeNode
{
    Node readonlyToken;
    Node typeParameter;
    Node nameType;
    Node questionToken;
    Node type;
};

struct LiteralTypeNode
{
    Node literal;
};

struct NamedTupleMember
{
    Node dotDotDotToken;
    Node name;
    Node questionToken;
    Node type;
};

struct BindingPattern
{
    Node elements;
};

struct ArrayLiteralExpression
{
    Node elements;
};

struct ObjectLiteralExpression
{
    Node properties;
};

struct PropertyAccessExpression
{
    Node expression;
    Node questionDotToken;
    Node name;
};

struct ElementAccessExpression
{
    Node expression;
    Node questionDotToken;
    Node argumentExpression;
};

struct CallExpression
{
    Node expression;
    Node questionDotToken;
    Node typeArguments;
    Node arguments;
};

struct TaggedTemplateExpression
{
    Node tag;
    Node questionDotToken;
    Node typeArguments;
    Node _template;
};

struct TypeAssertion
{
    Node type;
    Node expression;
};

struct ParenthesizedExpression
{
    Node expression;
};

struct DeleteExpression
{
    Node expression;
};

struct TypeOfExpression
{
    Node expression;
};

struct VoidExpression
{
    Node expression;
};

struct PrefixUnaryExpression
{
    Node operand;
};

struct YieldExpression
{
    Node asteriskToken;
    Node expression;
};

struct AwaitExpression
{
    Node expression;
};

struct PostfixUnaryExpression
{
    Node operand;
};

struct BinaryExpression
{
    Node left;
    Node operatorToken;
    Node right;
};

struct AsExpression
{
    Node expression;
    Node type;
};

struct NonNullExpression
{
    Node expression;
};

struct MetaProperty
{
    Node name;
};

struct ConditionalExpression
{
    Node condition;
    Node questionToken;
    Node whenTrue;
    Node colonToken;
    Node whenFalse;
};

struct SpreadElement
{
    Node expression;
};

struct PartiallyEmittedExpression
{
    Node expression;
};

struct Block
{
    Node statements;
};

struct SourceFile
{
    Node statements;
    Node endOfFileToken;
};

struct VariableStatement
{
    Node declarationList;
};

struct VariableDeclarationList
{
    Node declarations;
};

struct ExpressionStatement
{
    Node expression;
};

struct IfStatement
{
    Node expression;
    Node thenStatement;
    Node elseStatement;
};

struct DoStatement
{
    Node statement;
    Node expression;
};

struct WhileStatement
{
    Node expression;
    Node statement;
};

struct ForStatement
{
    Node initializer;
    Node condition;
    Node incrementor;
    Node statement;
};

struct ForInStatement
{
    Node initializer;
    Node expression;
    Node statement;
};

struct ForOfStatement
{
    Node awaitModifier;
    Node initializer;
    Node expression;
    Node statement;
};

struct BreakOrContinueStatement
{
    Node label;
};

struct ReturnStatement
{
    Node expression;
};

struct WithStatement
{
    Node expression;
    Node statement;
};

struct SwitchStatement
{
    Node expression;
    Node caseBlock;
};

struct CaseBlock
{
    Node clauses;
};

struct CaseClause
{
    Node expression;
    Node statements;
};

struct DefaultClause
{
    Node statements;
};

struct LabeledStatement
{
    Node label;
    Node statement;
};

struct ThrowStatement
{
    Node expression;
};

struct TryStatement
{
    Node tryBlock;
    Node catchClause;
    Node finallyBlock;
};

struct CatchClause
{
    Node variableDeclaration;
    Node block;
};

struct Decorator
{
    Node expression;
};

struct ClassLikeDeclaration
{
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
};

struct InterfaceDeclaration
{
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
};

struct ClassDeclaration
{
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
};

struct TypeAliasDeclaration
{
    Node name;
    Node typeParameters;
    Node type;
};

struct EnumDeclaration
{
    Node name;
    Node members;
};

struct EnumMember
{
    Node name;
    Node initializer;
};

struct ModuleDeclaration
{
    Node name;
    Node body;
};

struct ImportEqualsDeclaration
{
    Node name;
    Node moduleReference;
};

struct ImportDeclaration
{
    Node importClause;
    Node moduleSpecifier;
};

struct ImportClause
{
    Node name;
    Node namedBindings;
};

struct NamespaceExportDeclaration
{
    Node name;
};

struct NamespaceImport
{
    Node name;
};

struct NamespaceExport
{
    Node name;
};

struct NamedImportsOrExports
{
    Node elements;
};

struct ExportDeclaration
{
    Node exportClause;
    Node moduleSpecifier;
};

struct ImportOrExportSpecifier
{
    Node propertyName;
    Node name;
};

struct ExportAssignment
{
    Node expression;
};

struct TemplateExpression
{
    Node head;
    Node templateSpans;
};

struct TemplateSpan
{
    Node expression;
    Node literal;
};

struct TemplateLiteralTypeNode
{
    Node head;
    Node templateSpans;
};

struct TemplateLiteralTypeSpan
{
    Node type;
    Node literal;
};

struct ComputedPropertyName
{
    Node expression;
};

struct HeritageClause
{
    Node types;
};

struct ExpressionWithTypeArguments
{
    Node expression;
    Node typeArguments;
};

struct ExternalModuleReference
{
    Node expression;
};

struct CommaListExpression
{
    Node elements;
};

struct JsxElement
{
    Node openingElement;
    Node children;
    Node closingElement;
};

struct JsxFragment
{
    Node openingFragment;
    Node children;
    Node closingFragment;
};

struct JsxOpeningLikeElement
{
    Node tagName;
    Node typeArguments;
    Node attributes;
};

struct JsxAttributes
{
    Node properties;
};

struct JsxAttribute
{
    Node name;
    Node initializer;
};

struct JsxSpreadAttribute
{
    Node expression;
};

struct JsxExpression
{
    Node dotDotDotToken;
    Node expression;
};

struct JsxClosingElement
{
    Node tagName;
};

struct OptionalTypeNode
{
    Node type;
};

struct RestTypeNode
{
    Node type;
};

struct JSDocTypeExpression
{
    Node type;
};

struct JSDocNonNullableTypeNode
{
    Node type;
};

struct JSDocNullableTypeNode
{
    Node type;
};

struct JSDocOptionalTypeNode
{
    Node type;
};

struct JSDocVariadicTypeNode
{
    Node type;
};

struct JSDocFunctionType
{
    Node parameters;
    Node type;
};

struct JSDoc
{
    Node tags;
};

struct JSDocSeeTag
{
    Node tagName;
    Node name;
};

struct JSDocNameReference
{
    Node name;
};

struct JSDocTag
{
    Node tagName;
};

struct JSDocPropertyLikeTag
{
    Node isNameFirst;
    Node name;
    Node typeExpression;
};

struct JSDocImplementsTag
{
    Node _class;
};

struct JSDocAugmentsTag
{
    Node _class;
};

struct JSDocTemplateTag
{
    Node constraint;
    Node typeParameters;
};

struct JSDocTypedefTag
{
    Node typeExpression;
    Node fullName;
};

struct JSDocCallbackTag
{
    Node fullName;
    Node typeExpression;
};

struct JSDocReturnTag
{
    Node typeExpression;
};

struct JSDocTypeTag
{
    Node typeExpression;
};

struct JSDocThisTag
{
    Node typeExpression;
};

struct JSDocEnumTag
{
    Node typeExpression;
};

struct JSDocSignature
{
    Node typeParameters;
    Node parameters;
    Node type;
};

struct JSDocTypeLiteral
{
    Node jsDocPropertyTags;
};

#endif // PARSER_H