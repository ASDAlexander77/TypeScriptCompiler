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

#define CLASS_NODE_BASE(x, b) struct x : b { \
    x() {}  \
    x(Node node) : b(node) {}

#define CLASS_NODE(x) CLASS_NODE_BASE(x, BaseNode)

struct NodeData : TextRange
{   
    NodeFlags flags;
    string text;
    TransformFlags transformFlags;
    NodeArray<Node> decorators;
    ModifiersArray modifiers;
    SyntaxKind originalKeywordKind;
    number jsdocDotPos;

    bool isArray;
    NodeArray<Node> children;

    NodeData(number start, number end) : TextRange{start, end} {};

    virtual ~NodeData() {}
};

struct Node
{
    SyntaxKind kind;
    std::shared_ptr<NodeData> data;

    Node() : kind(SyntaxKind::Unknown) {};
    Node(SyntaxKind kind, number start, number end) : kind(kind), data(std::make_shared<NodeData>(start, end)) {};
    Node(undefined_t) : kind(SyntaxKind::Unknown) {};

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
        return this->kind != SyntaxKind::Unknown;
    }

    auto operator=(NodeArray<Node> values) -> Node&
     {
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
        data->isArray = true;
        data->children.push_back(node);
    }    

    auto operator==(undefined_t)
    {
        return !(bool)*this;
    }
};

struct BaseNode
{
    Node node;
    BaseNode() : node(Node()) {}
    BaseNode(Node node) : node(node) {}

    operator Node()
    {
        return Node();
    }

    bool operator !()
    {
        return !node;
    }

    NodeData* operator->()
    {
        return node.operator->();
    }    
};

static auto isArray(Node &node) -> boolean
{
    return node.data->isArray;
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

CLASS_NODE(MethodDeclaration)
    Node name;
};

CLASS_NODE(TypeParameterDeclaration)
    Node name;
    Node constraint;
    Node _default;
    Node expression;
};

CLASS_NODE(ShorthandPropertyAssignment)
    Node name;
    Node questionToken;
    Node exclamationToken;
    Node equalsToken;
    Node objectAssignmentInitializer;
};

CLASS_NODE(SpreadAssignment)
    Node expression;
};

CLASS_NODE(ParameterDeclaration)
    Node dotDotDotToken;
    Node name;
    Node questionToken;
    Node type;
    Node initializer;
};

CLASS_NODE(PropertyDeclaration)
    Node name;
    Node questionToken;
    Node exclamationToken;
    Node type;
    Node initializer;
};

CLASS_NODE(PropertySignature)
    Node name;
    Node questionToken;
    Node type;
    Node initializer;
};

CLASS_NODE(PropertyAssignment)
    Node name;
    Node questionToken;
    Node initializer;
};

CLASS_NODE(VariableDeclaration)
    Node name;
    Node exclamationToken;
    Node type;
    Node initializer;
};

CLASS_NODE(BindingElement)
    Node dotDotDotToken;
    Node propertyName;
    Node name;
    Node initializer;
};

CLASS_NODE(SignatureDeclaration)
    Node typeParameters;
    Node parameters;
    Node type;
};

CLASS_NODE(FunctionLikeDeclaration)
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

CLASS_NODE(ArrowFunction)
    Node equalsGreaterThanToken;
};


CLASS_NODE(TypeReferenceNode)
    Node typeArguments;
    Node typeName;
};

CLASS_NODE(TypePredicateNode)
    Node assertsModifier;
    Node parameterName;
    Node type;
};

CLASS_NODE(TypeQueryNode)
    Node exprName;
};

CLASS_NODE(TypeLiteralNode)
    Node members;
};

CLASS_NODE(TypeNode)
};

CLASS_NODE(ArrayTypeNode)
    Node elementType;
};

CLASS_NODE(TupleTypeNode)
    Node elements;
};

CLASS_NODE(UnionOrIntersectionTypeNode)
    Node types;
};

CLASS_NODE(ConditionalTypeNode)
    Node checkType;
    Node extendsType;
    Node trueType;
    Node falseType;
};

CLASS_NODE(InferTypeNode)
    Node typeParameter;
};

CLASS_NODE(ImportTypeNode)
    Node argument;
    Node qualifier;
    Node typeArguments;
};

CLASS_NODE(ParenthesizedTypeNode)
    Node type;
};

CLASS_NODE(TypeOperatorNode)
    Node type;
};

CLASS_NODE(IndexedAccessTypeNode)
    Node objectType;
    Node indexType;
};

CLASS_NODE(MappedTypeNode)
    Node readonlyToken;
    Node typeParameter;
    Node nameType;
    Node questionToken;
    Node type;
};

CLASS_NODE(LiteralTypeNode)
    Node literal;
};

CLASS_NODE(NamedTupleMember)
    Node dotDotDotToken;
    Node name;
    Node questionToken;
    Node type;
};

CLASS_NODE(BindingPattern)
    Node elements;
};

CLASS_NODE(ArrayLiteralExpression)
    Node elements;
};

CLASS_NODE(ObjectLiteralExpression)
    Node properties;
};

CLASS_NODE(PropertyAccessExpression)
    Node expression;
    Node questionDotToken;
    Node name;
};

CLASS_NODE(ElementAccessExpression)
    Node expression;
    Node questionDotToken;
    Node argumentExpression;
};

CLASS_NODE(CallExpression)
    Node expression;
    Node questionDotToken;
    Node typeArguments;
    Node arguments;
};

CLASS_NODE(TaggedTemplateExpression)
    Node tag;
    Node questionDotToken;
    Node typeArguments;
    Node _template;
};

CLASS_NODE(TypeAssertion)
    Node type;
    Node expression;
};

CLASS_NODE(ParenthesizedExpression)
    Node expression;
};

CLASS_NODE(DeleteExpression)
    Node expression;
};

CLASS_NODE(TypeOfExpression)
    Node expression;
};

CLASS_NODE(VoidExpression)
    Node expression;
};

CLASS_NODE(PrefixUnaryExpression)
    Node operand;
};

CLASS_NODE(YieldExpression)
    Node asteriskToken;
    Node expression;
};

CLASS_NODE(AwaitExpression)
    Node expression;
};

CLASS_NODE(PostfixUnaryExpression)
    Node operand;
};

CLASS_NODE(BinaryExpression)
    Node left;
    Node operatorToken;
    Node right;
};

CLASS_NODE(AsExpression)
    Node expression;
    Node type;
};

CLASS_NODE(NonNullExpression)
    Node expression;
};

CLASS_NODE(MetaProperty)
    Node name;
};

CLASS_NODE(ConditionalExpression)
    Node condition;
    Node questionToken;
    Node whenTrue;
    Node colonToken;
    Node whenFalse;
};

CLASS_NODE(SpreadElement)
    Node expression;
};

CLASS_NODE(PartiallyEmittedExpression)
    Node expression;
};

CLASS_NODE(Block)
    Node statements;
};

CLASS_NODE(SourceFile)
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
};

typedef SourceFile JsonSourceFile;

CLASS_NODE(VariableStatement)
    Node declarationList;
};

CLASS_NODE(VariableDeclarationList)
    Node declarations;
};

CLASS_NODE(ExpressionStatement)
    Node expression;
};

CLASS_NODE(IfStatement)
    Node expression;
    Node thenStatement;
    Node elseStatement;
};

CLASS_NODE(DoStatement)
    Node statement;
    Node expression;
};

CLASS_NODE(WhileStatement)
    Node expression;
    Node statement;
};

CLASS_NODE(ForStatement)
    Node initializer;
    Node condition;
    Node incrementor;
    Node statement;
};

CLASS_NODE(ForInStatement)
    Node initializer;
    Node expression;
    Node statement;
};

CLASS_NODE(ForOfStatement)
    Node awaitModifier;
    Node initializer;
    Node expression;
    Node statement;
};

CLASS_NODE(BreakOrContinueStatement)
    Node label;
};

CLASS_NODE(ReturnStatement)
    Node expression;
};

CLASS_NODE(WithStatement)
    Node expression;
    Node statement;
};

CLASS_NODE(SwitchStatement)
    Node expression;
    Node caseBlock;
};

CLASS_NODE(CaseBlock)
    Node clauses;
};

CLASS_NODE(CaseClause)
    Node expression;
    Node statements;
};

CLASS_NODE(DefaultClause)
    Node statements;
};

CLASS_NODE(LabeledStatement)
    Node label;
    Node statement;
};

CLASS_NODE(ThrowStatement)
    Node expression;
};

CLASS_NODE(TryStatement)
    Node tryBlock;
    Node catchClause;
    Node finallyBlock;
};

CLASS_NODE(CatchClause)
    Node variableDeclaration;
    Node block;
};

CLASS_NODE(Decorator)
    Node expression;
};

CLASS_NODE(ClassLikeDeclaration)
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
};

CLASS_NODE(InterfaceDeclaration)
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
};

CLASS_NODE(ClassDeclaration)
    Node name;
    Node typeParameters;
    Node heritageClauses;
    Node members;
};

CLASS_NODE(TypeAliasDeclaration)
    Node name;
    Node typeParameters;
    Node type;
};

CLASS_NODE(EnumDeclaration)
    Node name;
    Node members;
};

CLASS_NODE(EnumMember)
    Node name;
    Node initializer;
};

CLASS_NODE(ModuleDeclaration)
    Node name;
    Node body;
};

CLASS_NODE(ImportEqualsDeclaration)
    Node name;
    Node moduleReference;
};

CLASS_NODE(ImportDeclaration)
    Node importClause;
    Node moduleSpecifier;
};

CLASS_NODE(ImportClause)
    Node name;
    Node namedBindings;
};

CLASS_NODE(NamespaceExportDeclaration)
    Node name;
};

CLASS_NODE(NamespaceImport)
    Node name;
};

CLASS_NODE(NamespaceExport)
    Node name;
};

CLASS_NODE(NamedImportsOrExports)
    Node elements;
};

CLASS_NODE(ExportDeclaration)
    Node exportClause;
    Node moduleSpecifier;
};

CLASS_NODE(ImportOrExportSpecifier)
    Node propertyName;
    Node name;
};

CLASS_NODE(ExportAssignment)
    Node expression;
};

CLASS_NODE(TemplateExpression)
    Node head;
    Node templateSpans;
};

CLASS_NODE(TemplateSpan)
    Node expression;
    Node literal;
};

CLASS_NODE(TemplateHead)
};

CLASS_NODE(TemplateMiddle)
};

CLASS_NODE(TemplateTail)
};

CLASS_NODE(TemplateLiteralTypeNode)
    Node head;
    Node templateSpans;
};

CLASS_NODE(TemplateLiteralTypeSpan)
    Node type;
    Node literal;
};

CLASS_NODE_BASE(ComputedPropertyName, PropertyName)
    Node expression;
};

CLASS_NODE(HeritageClause)
    Node types;
};

CLASS_NODE(ExpressionWithTypeArguments)
    Node expression;
    Node typeArguments;
};

CLASS_NODE(ExternalModuleReference)
    Node expression;
};

CLASS_NODE(CommaListExpression)
    Node elements;
};

CLASS_NODE(JsxElement)
    Node openingElement;
    Node children;
    Node closingElement;
};

CLASS_NODE(JsxFragment)
    Node openingFragment;
    Node children;
    Node closingFragment;
};

CLASS_NODE(JsxOpeningLikeElement)
    Node tagName;
    Node typeArguments;
    Node attributes;
};

CLASS_NODE(JsxAttributes)
    Node properties;
};

CLASS_NODE(JsxAttribute)
    Node name;
    Node initializer;
};

CLASS_NODE(JsxSpreadAttribute)
    Node expression;
};

CLASS_NODE(JsxExpression)
    Node dotDotDotToken;
    Node expression;
};

CLASS_NODE(JsxClosingElement)
    Node tagName;
};

CLASS_NODE(OptionalTypeNode)
    Node type;
};

CLASS_NODE(RestTypeNode)
    Node type;
};

CLASS_NODE(JSDocTypeExpression)
    Node type;
};

CLASS_NODE(JSDocNonNullableTypeNode)
    Node type;
};

CLASS_NODE(JSDocNullableTypeNode)
    Node type;
};

CLASS_NODE(JSDocOptionalTypeNode)
    Node type;
};

CLASS_NODE(JSDocVariadicTypeNode)
    Node type;
};

CLASS_NODE(JSDocFunctionType)
    Node parameters;
    Node type;
};

CLASS_NODE(JSDoc)
    Node tags;
};

CLASS_NODE(JSDocSeeTag)
    Node tagName;
    Node name;
};

CLASS_NODE(JSDocNameReference)
    Node name;
};

CLASS_NODE(JSDocTag)
    Node tagName;
};

CLASS_NODE(JSDocPropertyLikeTag)
    Node isNameFirst;
    Node name;
    Node typeExpression;
};

CLASS_NODE(JSDocImplementsTag)
    Node _class;
};

CLASS_NODE(JSDocAugmentsTag)
    Node _class;
};

CLASS_NODE(JSDocTemplateTag)
    Node constraint;
    Node typeParameters;
};

CLASS_NODE(JSDocTypedefTag)
    Node typeExpression;
    Node fullName;
};

CLASS_NODE(JSDocCallbackTag)
    Node fullName;
    Node typeExpression;
};

CLASS_NODE(JSDocReturnTag)
    Node typeExpression;
};

CLASS_NODE(JSDocTypeTag)
    Node typeExpression;
};

CLASS_NODE(JSDocThisTag)
    Node typeExpression;
};

CLASS_NODE(JSDocEnumTag)
    Node typeExpression;
};

CLASS_NODE(JSDocSignature)
    Node typeParameters;
    Node parameters;
    Node type;
};

CLASS_NODE(JSDocTypeLiteral)
    Node jsDocPropertyTags;
};

CLASS_NODE(JSDocContainer)
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