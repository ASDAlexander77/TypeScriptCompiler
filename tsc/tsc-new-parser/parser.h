#ifndef PARSER_H
#define PARSER_H

#include "undefined.h"
#include "enums.h"
#include "types.h"
#include "scanner.h"
#include "debug.h"

struct Node;

template <typename T>
using NodeFuncT = std::function<T(Node)>;

template <typename T>
using NodeWithParentFuncT = std::function<T(Node, Node)>;

typedef std::function<Node(SyntaxKind, number, number)> NodeCreateFunc;

typedef std::function<void(number, number, DiagnosticMessage)> PragmaDiagnosticReporter;

typedef SyntaxKind Modifier;

struct Node;

template <typename T>
struct NodeArray
{
    NodeArray() = default;

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

    inline auto size() -> size_t {
        return items.size();
    }

    auto push_back(Node node) -> void
    {
        items.push_back(node);
    }       
};

template <typename T>
using NodeArrayFuncT = std::function<T(NodeArray<T>)>;

template <typename T>
using NodeWithParentArrayFuncT = std::function<T(NodeArray<T>, Node)>;

typedef NodeArray<Modifier> ModifiersArray;

struct Node : TextRange
{
    Node() = default;
    Node(SyntaxKind kind, number start, number end) : kind(kind), TextRange{start, end} {};
    Node(undefined_t) : kind(SyntaxKind::Unknown) {};

    SyntaxKind kind;
    NodeFlags flags;
    SyntaxKind originalKeywordKind;
    string text;
    TransformFlags transformFlags;
    NodeArray<Node> decorators;
    ModifiersArray modifiers;
    number jsdocDotPos;

    bool isArray;
    NodeArray<Node> children;

    template <typename T> 
    auto as() -> T
    {
        return T();
    }

    template <typename T> 
    auto asMutable() -> T&
    {
        return T();
    }    

    operator bool()
    {
        return this->kind != SyntaxKind::Unknown;
    }

    auto operator=(NodeArray<Node> values) -> Node&
     {
        children = values;
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
        return children.size();
    }

    auto operator[](number i) const -> Node
    {
        return children[i];
    }    

    auto operator[](number i) -> Node&
    {
        return children[i];
    }    

    auto push_back(Node node) -> void
    {
        isArray = true;
        children.push_back(node);
    }    

    auto operator==(undefined_t)
    {
        return !(bool)*this;
    }
};

static auto isArray(Node &node) -> boolean
{
    return node.isArray;
}

typedef Node Identifier, PropertyName, PrivateIdentifier, ThisTypeNode, LiteralLikeNode, LiteralExpression, EntityName, Expression, IndexSignatureDeclaration,
    TypeElement, BinaryOperatorToken, UnaryExpression, UpdateExpression, LeftHandSideExpression, MemberExpression, JsxText, JsxChild, JsxTagNameExpression,
    JsxClosingFragment, QuestionDotToken, PrimaryExpression, ObjectLiteralElementLike, FunctionExpression, Statement, CaseOrDefaultClause, ArrayBindingElement,
    ObjectBindingPattern, ArrayBindingPattern, FunctionDeclaration, ConstructorDeclaration, AccessorDeclaration, ClassElement, ClassExpression,
    ModuleBlock, EndOfFileToken, BooleanLiteral, NullLiteral;

struct QualifiedName
{
    Node left;
    Node right;
};

struct MethodDeclaration
{
    Node name;
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

struct TypeNode
{
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

struct SourceFile : TextRange
{
    Node root;
    NodeFlags flags;
    TransformFlags transformFlags;    

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

    bool operator !()
    {
        return !root;
    }    

    operator Node&()
    {
        return root;
    }
};

typedef SourceFile JsonSourceFile;

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

struct TemplateHead
{
};

struct TemplateMiddle
{
};

struct TemplateTail
{
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

struct ComputedPropertyName : PropertyName
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

struct JSDocContainer
{
    Node jsDocCache;
};

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