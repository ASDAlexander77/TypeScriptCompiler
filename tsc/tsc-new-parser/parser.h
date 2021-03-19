#ifndef PARSER_H
#define PARSER_H

#include "scanner.h"
#include "debug.h"

struct Node;

template <typename T>
using NodeArray = std::vector<T>;

typedef SyntaxKind Modifier;
typedef NodeArray<Modifier> ModifiersArray;

template <typename T>
using NodeFuncT = std::function<T(Node)>;

template <typename T>
using NodeWithParentFuncT = std::function<T(Node, Node)>;

template <typename T>
using NodeArrayFuncT = std::function<T(NodeArray<T>)>;

template <typename T>
using NodeWithParentArrayFuncT = std::function<T(NodeArray<T>, Node)>;

typedef std::function<Node(SyntaxKind, number, number)> NodeCreateFunc;

struct undefined_t
{
};

static undefined_t undefined;

template <typename T>
struct Undefined
{
    Undefined() : _hasValue(false)
    {
    }

    Undefined(undefined_t) : _hasValue(false)
    {
    }

    Undefined(T value) : _hasValue(true), _value(value)
    {
    }

    boolean _hasValue;
    T _value;

    operator bool()
    {
        if (!_hasValue)
        {
            return false;
        }

        return !!_value;
    }

    bool hasValue()
    {
        return _hasValue;
    }
};

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

enum class ScriptKind : number {
    Unknown = 0,
    JS = 1,
    JSX = 2,
    TS = 3,
    TSX = 4,
    External = 5,
    JSON = 6,
    /**
     * Used on extensions that doesn't define the ScriptKind but the content defines it.
     * Deferred extensions are going to be included in all project contexts.
     */
    Deferred = 7
};

enum class NodeFlags {
    None               = 0,
    Let                = 1 << 0,  // Variable declaration
    Const              = 1 << 1,  // Variable declaration
    NestedNamespace    = 1 << 2,  // Namespace declaration
    Synthesized        = 1 << 3,  // Node was synthesized during transformation
    Namespace          = 1 << 4,  // Namespace declaration
    OptionalChain      = 1 << 5,  // Chained MemberExpression rooted to a pseudo-OptionalExpression
    ExportContext      = 1 << 6,  // Export context (initialized by binding)
    ContainsThis       = 1 << 7,  // Interface contains references to "this"
    HasImplicitReturn  = 1 << 8,  // If function implicitly returns on one of codepaths (initialized by binding)
    HasExplicitReturn  = 1 << 9,  // If function has explicit reachable return on one of codepaths (initialized by binding)
    GlobalAugmentation = 1 << 10,  // Set if module declaration is an augmentation for the global scope
    HasAsyncFunctions  = 1 << 11, // If the file has async functions (initialized by binding)
    DisallowInContext  = 1 << 12, // If node was parsed in a context where 'in-expressions' are not allowed
    YieldContext       = 1 << 13, // If node was parsed in the 'yield' context created when parsing a generator
    DecoratorContext   = 1 << 14, // If node was parsed as part of a decorator
    AwaitContext       = 1 << 15, // If node was parsed in the 'await' context created when parsing an async function
    ThisNodeHasError   = 1 << 16, // If the parser encountered an error when parsing the code that created this node
    JavaScriptFile     = 1 << 17, // If node was parsed in a JavaScript
    ThisNodeOrAnySubNodesHasError = 1 << 18, // If this node or any of its children had an error
    HasAggregatedChildData = 1 << 19, // If we've computed data from children and cached it in this node

    // These flags will be set when the parser encounters a dynamic import expression or 'import.meta' to avoid
    // walking the tree if the flags are not set. However, these flags are just a approximation
    // (hence why it's named "PossiblyContainsDynamicImport") because once set, the flags never get cleared.
    // During editing, if a dynamic import is removed, incremental parsing will *NOT* clear this flag.
    // This means that the tree will always be traversed during module resolution, or when looking for external module indicators.
    // However, the removal operation should not occur often and in the case of the
    // removal, it is likely that users will add the import anyway.
    // The advantage of this approach is its simplicity. For the case of batch compilation,
    // we guarantee that users won't have to pay the price of walking the tree if a dynamic import isn't used.
    /* @internal */ PossiblyContainsDynamicImport = 1 << 20,
    /* @internal */ PossiblyContainsImportMeta    = 1 << 21,

    JSDoc                                         = 1 << 22, // If node was parsed inside jsdoc
    /* @internal */ Ambient                       = 1 << 23, // If node was inside an ambient context -- a declaration file, or inside something with the `declare` modifier.
    /* @internal */ InWithStatement               = 1 << 24, // If any ancestor of node was the `statement` of a WithStatement (not the `expression`)
    JsonFile                                      = 1 << 25, // If node was parsed in a Json
    /* @internal */ TypeCached                    = 1 << 26, // If a type was cached for node at any point
    /* @internal */ Deprecated                    = 1 << 27, // If has '@deprecated' JSDoc tag

    BlockScoped = Let | Const,

    ReachabilityCheckFlags = HasImplicitReturn | HasExplicitReturn,
    ReachabilityAndEmitFlags = ReachabilityCheckFlags | HasAsyncFunctions,

    // Parsing context flags
    ContextFlags = DisallowInContext | YieldContext | DecoratorContext | AwaitContext | JavaScriptFile | InWithStatement | Ambient,

    // Exclude these flags when parsing a Type
    TypeExcludesFlags = YieldContext | AwaitContext,

    // Represents all flags that are potentially set once and
    // never cleared on SourceFiles which get re-used in between incremental parses.
    // See the comment above on `PossiblyContainsDynamicImport` and `PossiblyContainsImportMeta`.
    /* @internal */ PermanentlySetIncrementalFlags = PossiblyContainsDynamicImport | PossiblyContainsImportMeta,
};

enum class OperatorPrecedence : number {
    // Expression:
    //     AssignmentExpression
    //     Expression `,` AssignmentExpression
    Comma,

    // NOTE: `Spread` is higher than `Comma` due to how it is parsed in |ElementList|
    // SpreadElement:
    //     `...` AssignmentExpression
    Spread,

    // AssignmentExpression:
    //     ConditionalExpression
    //     YieldExpression
    //     ArrowFunction
    //     AsyncArrowFunction
    //     LeftHandSideExpression `=` AssignmentExpression
    //     LeftHandSideExpression AssignmentOperator AssignmentExpression
    //
    // NOTE: AssignmentExpression is broken down into several precedences due to the requirements
    //       of the parenthesizer rules.

    // AssignmentExpression: YieldExpression
    // YieldExpression:
    //     `yield`
    //     `yield` AssignmentExpression
    //     `yield` `*` AssignmentExpression
    Yield,

    // AssignmentExpression: LeftHandSideExpression `=` AssignmentExpression
    // AssignmentExpression: LeftHandSideExpression AssignmentOperator AssignmentExpression
    // AssignmentOperator: one of
    //     `*=` `/=` `%=` `+=` `-=` `<<=` `>>=` `>>>=` `&=` `^=` `|=` `**=`
    Assignment,

    // NOTE: `Conditional` is considered higher than `Assignment` here, but in reality they have
    //       the same precedence.
    // AssignmentExpression: ConditionalExpression
    // ConditionalExpression:
    //     ShortCircuitExpression
    //     ShortCircuitExpression `?` AssignmentExpression `:` AssignmentExpression
    // ShortCircuitExpression:
    //     LogicalORExpression
    //     CoalesceExpression
    Conditional,

    // CoalesceExpression:
    //     CoalesceExpressionHead `??` BitwiseORExpression
    // CoalesceExpressionHead:
    //     CoalesceExpression
    //     BitwiseORExpression
    Coalesce = Conditional, // NOTE: This is wrong

    // LogicalORExpression:
    //     LogicalANDExpression
    //     LogicalORExpression `||` LogicalANDExpression
    LogicalOR,

    // LogicalANDExpression:
    //     BitwiseORExpression
    //     LogicalANDExprerssion `&&` BitwiseORExpression
    LogicalAND,

    // BitwiseORExpression:
    //     BitwiseXORExpression
    //     BitwiseORExpression `^` BitwiseXORExpression
    BitwiseOR,

    // BitwiseXORExpression:
    //     BitwiseANDExpression
    //     BitwiseXORExpression `^` BitwiseANDExpression
    BitwiseXOR,

    // BitwiseANDExpression:
    //     EqualityExpression
    //     BitwiseANDExpression `^` EqualityExpression
    BitwiseAND,

    // EqualityExpression:
    //     RelationalExpression
    //     EqualityExpression `==` RelationalExpression
    //     EqualityExpression `!=` RelationalExpression
    //     EqualityExpression `===` RelationalExpression
    //     EqualityExpression `!==` RelationalExpression
    Equality,

    // RelationalExpression:
    //     ShiftExpression
    //     RelationalExpression `<` ShiftExpression
    //     RelationalExpression `>` ShiftExpression
    //     RelationalExpression `<=` ShiftExpression
    //     RelationalExpression `>=` ShiftExpression
    //     RelationalExpression `instanceof` ShiftExpression
    //     RelationalExpression `in` ShiftExpression
    //     [+TypeScript] RelationalExpression `as` Type
    Relational,

    // ShiftExpression:
    //     AdditiveExpression
    //     ShiftExpression `<<` AdditiveExpression
    //     ShiftExpression `>>` AdditiveExpression
    //     ShiftExpression `>>>` AdditiveExpression
    Shift,

    // AdditiveExpression:
    //     MultiplicativeExpression
    //     AdditiveExpression `+` MultiplicativeExpression
    //     AdditiveExpression `-` MultiplicativeExpression
    Additive,

    // MultiplicativeExpression:
    //     ExponentiationExpression
    //     MultiplicativeExpression MultiplicativeOperator ExponentiationExpression
    // MultiplicativeOperator: one of `*`, `/`, `%`
    Multiplicative,

    // ExponentiationExpression:
    //     UnaryExpression
    //     UpdateExpression `**` ExponentiationExpression
    Exponentiation,

    // UnaryExpression:
    //     UpdateExpression
    //     `delete` UnaryExpression
    //     `void` UnaryExpression
    //     `typeof` UnaryExpression
    //     `+` UnaryExpression
    //     `-` UnaryExpression
    //     `~` UnaryExpression
    //     `!` UnaryExpression
    //     AwaitExpression
    // UpdateExpression:            // TODO: Do we need to investigate the precedence here?
    //     `++` UnaryExpression
    //     `--` UnaryExpression
    Unary,


    // UpdateExpression:
    //     LeftHandSideExpression
    //     LeftHandSideExpression `++`
    //     LeftHandSideExpression `--`
    Update,

    // LeftHandSideExpression:
    //     NewExpression
    //     CallExpression
    // NewExpression:
    //     MemberExpression
    //     `new` NewExpression
    LeftHandSide,

    // CallExpression:
    //     CoverCallExpressionAndAsyncArrowHead
    //     SuperCall
    //     ImportCall
    //     CallExpression Arguments
    //     CallExpression `[` Expression `]`
    //     CallExpression `.` IdentifierName
    //     CallExpression TemplateLiteral
    // MemberExpression:
    //     PrimaryExpression
    //     MemberExpression `[` Expression `]`
    //     MemberExpression `.` IdentifierName
    //     MemberExpression TemplateLiteral
    //     SuperProperty
    //     MetaProperty
    //     `new` MemberExpression Arguments
    Member,

    // TODO: JSXElement?
    // PrimaryExpression:
    //     `this`
    //     IdentifierReference
    //     Literal
    //     ArrayLiteral
    //     ObjectLiteral
    //     FunctionExpression
    //     ClassExpression
    //     GeneratorExpression
    //     AsyncFunctionExpression
    //     AsyncGeneratorExpression
    //     RegularExpressionLiteral
    //     TemplateLiteral
    //     CoverParenthesizedExpressionAndArrowParameterList
    Primary,

    Highest = Primary,
    Lowest = Comma,
    // -1 is lower than all other precedences. Returning it will cause binary expression
    // parsing to stop.
    Invalid = -1,
};

static NodeFlags operator |(NodeFlags lhs, NodeFlags rhs)
{
    return (NodeFlags) ((number) lhs | (number) rhs);
}

enum class ParsingContext {
    SourceElements,            // Elements in source file
    BlockStatements,           // Statements in block
    SwitchClauses,             // Clauses in switch statement
    SwitchClauseStatements,    // Statements in switch clause
    TypeMembers,               // Members in interface or type literal
    ClassMembers,              // Members in class declaration
    EnumMembers,               // Members in enum declaration
    HeritageClauseElement,     // Elements in a heritage clause
    VariableDeclarations,      // Variable declarations in variable statement
    ObjectBindingElements,     // Binding elements in object binding list
    ArrayBindingElements,      // Binding elements in array binding list
    ArgumentExpressions,       // Expressions in argument list
    ObjectLiteralMembers,      // Members in object literal
    JsxAttributes,             // Attributes in jsx element
    JsxChildren,               // Things between opening and closing JSX tags
    ArrayLiteralMembers,       // Members in array literal
    Parameters,                // Parameters in parameter list
    JSDocParameters,           // JSDoc parameters in parameter list of JSDoc function type
    RestProperties,            // Property names in a rest type list
    TypeParameters,            // Type parameters in type parameter list
    TypeArguments,             // Type arguments in type argument list
    TupleElementTypes,         // Element types in tuple element type list
    HeritageClauses,           // Heritage clauses for a class or interface declaration.
    ImportOrExportSpecifiers,  // Named import clause's import specifier list
    Count                      // Number of parsing contexts
};

enum class Tristate {
    False,
    True,
    Unknown
};

struct TextSpan {
    number start;
    number length;
};

struct FileReference : TextSpan {
    string fileName;
};

struct AmdDependency {
    string path;
    string name;
};

struct TextChangeRange {
    TextSpan span;
    number newLength;
};

struct DiagnosticRelatedInformation {
    DiagnosticCategory category;
    string fileName;
    number code;
    number start;
    number length;
    string messageText;
};

struct Diagnostic : DiagnosticRelatedInformation {
    std::vector<string> reportsUnnecessary;
    std::vector<DiagnosticRelatedInformation> relatedInformation;
};

struct DiagnosticWithDetachedLocation : Diagnostic {
};

struct Node
{
    SyntaxKind kind;
    NodeFlags flags;
    NodeArray<Node> decorators;
    ModifiersArray modifiers;

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

    auto push(Node node) -> void
    {
        isArray = true;
        children.push_back(node);
    }
};

static auto isArray(Node &node) -> boolean
{
    return node.isArray;
}

typedef Node Identifier, PropertyName, PrivateIdentifier, ThisTypeNode, LiteralLikeNode, LiteralExpression, EntityName, Expression, IndexSignatureDeclaration,
    TypeElement, BinaryOperatorToken, UnaryExpression, UpdateExpression, LeftHandSideExpression, MemberExpression, JsxText, JsxChild, JsxTagNameExpression,
    JsxClosingFragment, QuestionDotToken, PrimaryExpression, ObjectLiteralElementLike, FunctionExpression, Statement, CaseOrDefaultClause, ArrayBindingElement,
    ObjectBindingPattern, ArrayBindingPattern, FunctionDeclaration, ConstructorDeclaration, MethodDeclaration, AccessorDeclaration, ClassElement, ClassExpression,
    ModuleBlock, EndOfFileToken, BooleanLiteral, NullLiteral;

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

struct SourceFile
{
    Node root;

    Node statements;
    Node endOfFileToken;
    Node externalModuleIndicator;
    Node commonJsModuleIndicator;

    // extra fields
    std::vector<FileReference> referencedFiles;
    std::vector<FileReference> typeReferenceDirectives;
    std::vector<FileReference> libReferenceDirectives;
    std::vector<FileReference> languageVariant;
    std::vector<AmdDependency> amdDependencies;
    boolean isDeclarationFile;

    std::map<string, string> renamedDependencies;
    boolean hasNoDefaultLib;
    ScriptTarget languageVersion;

    std::map<string, string> pragmas;

    // stats
    string fileName;
    string text;
    number nodeCount;
    number identifierCount;
    std::map<string, string> identifiers;

    std::vector<DiagnosticWithDetachedLocation> parseDiagnostics;
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

struct DiagnosticWithLocation : Diagnostic {
    SourceFile file;
};

struct NodeWithDiagnostics
{ 
    Node node;
    std::vector<Diagnostic> diagnostics;
};

template <typename T, typename U>
auto forEach(std::vector<T> array, std::function<U(T, number)> callback = nullptr) -> U {
    if (!array.empty()) {
        for (let i = 0; i < array.size(); i++) {
            auto result = callback(array[i], i);
            if (result) {
                return result;
            }
        }
    }

    return U();
}

static auto getScriptKindFromFileName(string fileName) -> ScriptKind
{
    auto ext = fileName.substr(fileName.find(S('.')));
    std::transform(ext.begin(), ext.end(), ext.begin(), [](char_t c){ return std::tolower(c); });
    if (ext == S("js"))
        return ScriptKind::JS;
    if (ext == S("jsx"))
        return ScriptKind::JSX;
    if (ext == S("ts"))
        return ScriptKind::TS;
    if (ext == S("tsx"))
        return ScriptKind::TSX;
    if (ext == S("json"))
        return ScriptKind::JSON;
    return ScriptKind::Unknown;
}

static auto ensureScriptKind(string fileName, ScriptKind scriptKind = ScriptKind::Unknown) -> ScriptKind {
    // Using scriptKind as a condition handles both:
    // - 'scriptKind' is unspecified and thus it is `undefined`
    // - 'scriptKind' is set and it is `Unknown` (0)
    // If the 'scriptKind' is 'undefined' or 'Unknown' then we attempt
    // to get the ScriptKind from the file name. If it cannot be resolved
    // from the file name then the default 'TS' script kind is returned.
    return scriptKind != ScriptKind::Unknown ? scriptKind : scriptKind = getScriptKindFromFileName(fileName), scriptKind != ScriptKind::Unknown ? scriptKind : ScriptKind::TS;
}

static auto isDiagnosticWithDetachedLocation(DiagnosticRelatedInformation diagnostic) -> boolean {
    return diagnostic.start != -1
        && diagnostic.length != -1
        && diagnostic.fileName != S("");
}

template <typename T>
auto attachFileToDiagnostic(T diagnostic, SourceFile file) -> DiagnosticWithLocation {
    auto fileName = file.fileName;
    auto length = file.text.length();
    Debug::assertEqual(diagnostic.fileName, fileName);
    Debug::assertLessThanOrEqual(diagnostic.start, length);
    Debug::assertLessThanOrEqual(diagnostic.start + diagnostic.length, length);
    DiagnosticWithLocation diagnosticWithLocation;
    diagnosticWithLocation.file = file;
    diagnosticWithLocation.start = diagnostic.start;
    diagnosticWithLocation.length = diagnostic.length;
    diagnosticWithLocation.messageText = diagnostic.messageText;
    diagnosticWithLocation.category = diagnostic.category;
    diagnosticWithLocation.code = diagnostic.code;
    diagnosticWithLocation.reportsUnnecessary = diagnostic.reportsUnnecessary;

    if (!diagnostic.relatedInformation.empty()) {
        for (auto &related : diagnostic.relatedInformation) {
            if (isDiagnosticWithDetachedLocation(related) && related.fileName == fileName) {
                Debug::assertLessThanOrEqual(related.start, length);
                Debug::assertLessThanOrEqual(related.start + related.length, length);
                diagnosticWithLocation.relatedInformation.push_back(attachFileToDiagnostic(related, file));
            }
            else {
                diagnosticWithLocation.relatedInformation.push_back(related);
            }
        }
    }

    return diagnosticWithLocation;
}

static auto attachFileToDiagnostics(std::vector<DiagnosticWithDetachedLocation> diagnostics, SourceFile file) -> std::vector<DiagnosticWithLocation> {
    std::vector<DiagnosticWithLocation> diagnosticsWithLocation;
    for (auto &diagnostic : diagnostics) {
        diagnosticsWithLocation.push_back(attachFileToDiagnostic(diagnostic, file));
    }
    return diagnosticsWithLocation;
}


#endif // PARSER_H