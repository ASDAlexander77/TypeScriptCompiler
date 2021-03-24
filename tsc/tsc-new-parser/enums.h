#ifndef ENUMS_H
#define ENUMS_H

#include "config.h"

#define ENUM_OPS(x) \
inline bool operator !(x lhs)   \
{   \
    return (number)lhs == 0;    \
}   \
    \
inline x operator |(x lhs, x rhs)   \
{   \
    return (x) ((number) lhs | (number) rhs);   \
}   \
    \
inline x operator &(x lhs, x rhs)   \
{   \
    return (x) ((number) lhs & (number) rhs);   \
}   \
    \
inline x operator ~(x rhs)  \
{   \
    return (x) (~(number) rhs); \
}   \
    \
inline x& operator|=(x& lhv, x rhv) \
{   \
    lhv = (x) ((number) lhv | (number)rhv); \
    return lhv; \
}   \
    \
inline x& operator&=(x& lhv, x rhv) \
{   \
    lhv = (x) ((number) lhv & (number)rhv); \
    return lhv; \
}

enum class ScriptTarget : number {
    ES3 = 0,
    ES5 = 1,
    ES2015 = 2,
    ES2016 = 3,
    ES2017 = 4,
    ES2018 = 5,
    ES2019 = 6,
    ES2020 = 7,
    ES2021 = 8,
    ESNext = 99,
    JSON = 100,
    Latest = ESNext,
};

enum class CommentDirectiveType : number {
    Undefined,
    ExpectError,
    Ignore
};

enum class LanguageVariant : number {
    Standard,
    JSX
};

enum class DiagnosticCategory : int
{
    Undefined,
    Warning,
    Error,
    Suggestion,
    Message
};

enum class SignatureFlags : number {
    None = 0,
    Yield = 1 << 0,
    Await = 1 << 1,
    Type  = 1 << 2,
    IgnoreMissingOpenBrace = 1 << 4,
    JSDoc = 1 << 5,
};

ENUM_OPS(SignatureFlags)

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

enum class NodeFlags : number {
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

ENUM_OPS(NodeFlags)

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

enum class ParsingContext : number {
    Unknown,
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

ENUM_OPS(ParsingContext)

enum class TransformFlags : number {
    None = 0,

    // Facts
    // - Flags used to indicate that a node or subtree contains syntax that requires transformation.
    ContainsTypeScript = 1 << 0,
    ContainsJsx = 1 << 1,
    ContainsESNext = 1 << 2,
    ContainsES2021 = 1 << 3,
    ContainsES2020 = 1 << 4,
    ContainsES2019 = 1 << 5,
    ContainsES2018 = 1 << 6,
    ContainsES2017 = 1 << 7,
    ContainsES2016 = 1 << 8,
    ContainsES2015 = 1 << 9,
    ContainsGenerator = 1 << 10,
    ContainsDestructuringAssignment = 1 << 11,

    // Markers
    // - Flags used to indicate that a subtree contains a specific transformation.
    ContainsTypeScriptClassSyntax = 1 << 12, // Decorators, Property Initializers, Parameter Property Initializers
    ContainsLexicalThis = 1 << 13,
    ContainsRestOrSpread = 1 << 14,
    ContainsObjectRestOrSpread = 1 << 15,
    ContainsComputedPropertyName = 1 << 16,
    ContainsBlockScopedBinding = 1 << 17,
    ContainsBindingPattern = 1 << 18,
    ContainsYield = 1 << 19,
    ContainsAwait = 1 << 20,
    ContainsHoistedDeclarationOrCompletion = 1 << 21,
    ContainsDynamicImport = 1 << 22,
    ContainsClassFields = 1 << 23,
    ContainsPossibleTopLevelAwait = 1 << 24,

    // Please leave this as 1 << 29.
    // It is the maximum bit we can set before we outgrow the size of a v8 small integer (SMI) on an x86 system.
    // It is a good reminder of how much room we have left
    HasComputedFlags = 1 << 29, // Transform flags have been computed.

    // Assertions
    // - Bitmasks that are used to assert facts about the syntax of a node and its subtree.
    AssertTypeScript = ContainsTypeScript,
    AssertJsx = ContainsJsx,
    AssertESNext = ContainsESNext,
    AssertES2021 = ContainsES2021,
    AssertES2020 = ContainsES2020,
    AssertES2019 = ContainsES2019,
    AssertES2018 = ContainsES2018,
    AssertES2017 = ContainsES2017,
    AssertES2016 = ContainsES2016,
    AssertES2015 = ContainsES2015,
    AssertGenerator = ContainsGenerator,
    AssertDestructuringAssignment = ContainsDestructuringAssignment,

    // Scope Exclusions
    // - Bitmasks that exclude flags from propagating out of a specific context
    //   into the subtree flags of their container.
    OuterExpressionExcludes = HasComputedFlags,
    PropertyAccessExcludes = OuterExpressionExcludes,
    NodeExcludes = PropertyAccessExcludes,
    ArrowFunctionExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsBlockScopedBinding | ContainsYield | ContainsAwait | ContainsHoistedDeclarationOrCompletion | ContainsBindingPattern | ContainsObjectRestOrSpread | ContainsPossibleTopLevelAwait,
    FunctionExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsLexicalThis | ContainsBlockScopedBinding | ContainsYield | ContainsAwait | ContainsHoistedDeclarationOrCompletion | ContainsBindingPattern | ContainsObjectRestOrSpread | ContainsPossibleTopLevelAwait,
    ConstructorExcludes = NodeExcludes | ContainsLexicalThis | ContainsBlockScopedBinding | ContainsYield | ContainsAwait | ContainsHoistedDeclarationOrCompletion | ContainsBindingPattern | ContainsObjectRestOrSpread | ContainsPossibleTopLevelAwait,
    MethodOrAccessorExcludes = NodeExcludes | ContainsLexicalThis | ContainsBlockScopedBinding | ContainsYield | ContainsAwait | ContainsHoistedDeclarationOrCompletion | ContainsBindingPattern | ContainsObjectRestOrSpread,
    PropertyExcludes = NodeExcludes | ContainsLexicalThis,
    ClassExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsComputedPropertyName,
    ModuleExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsLexicalThis | ContainsBlockScopedBinding | ContainsHoistedDeclarationOrCompletion | ContainsPossibleTopLevelAwait,
    TypeExcludes = ~ContainsTypeScript,
    ObjectLiteralExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsComputedPropertyName | ContainsObjectRestOrSpread,
    ArrayLiteralOrCallOrNewExcludes = NodeExcludes | ContainsRestOrSpread,
    VariableDeclarationListExcludes = NodeExcludes | ContainsBindingPattern | ContainsObjectRestOrSpread,
    ParameterExcludes = NodeExcludes,
    CatchClauseExcludes = NodeExcludes | ContainsObjectRestOrSpread,
    BindingPatternExcludes = NodeExcludes | ContainsRestOrSpread,

    // Propagating flags
    // - Bitmasks for flags that should propagate from a child
    PropertyNamePropagatingFlags = ContainsLexicalThis,

    // Masks
    // - Additional bitmasks
};

ENUM_OPS(TransformFlags)

enum class GeneratedIdentifierFlags : number {
    // Kinds
    None = 0,                           // Not automatically generated.
    /*@internal*/ Auto = 1,             // Automatically generated identifier.
    /*@internal*/ Loop = 2,             // Automatically generated identifier with a preference for '_i'.
    /*@internal*/ Unique = 3,           // Unique name based on the 'text' property.
    /*@internal*/ Node = 4,             // Unique name based on the node in the 'original' property.
    /*@internal*/ KindMask = 7,         // Mask to extract the kind of identifier from its flags.

    // Flags
    ReservedInNestedScopes = 1 << 3,    // Reserve the generated name in nested scopes
    Optimistic = 1 << 4,                // First instance won't use '_#' if there's no conflict
    FileLevel = 1 << 5,                 // Use only the file identifiers list and not generated names to search for conflicts
    AllowNameSubstitution = 1 << 6, // Used by `module.ts` to indicate generated nodes which can have substitutions performed upon them (as they were generated by an earlier transform phase)
};

enum class ModifierFlags : number {
    None =               0,
    Export =             1 << 0,  // Declarations
    Ambient =            1 << 1,  // Declarations
    Public =             1 << 2,  // Property/Method
    Private =            1 << 3,  // Property/Method
    Protected =          1 << 4,  // Property/Method
    Static =             1 << 5,  // Property/Method
    Readonly =           1 << 6,  // Property/Method
    Abstract =           1 << 7,  // Class/Method/ConstructSignature
    Async =              1 << 8,  // Property/Method/Function
    Default =            1 << 9,  // Function/Class (export default declaration)
    Const =              1 << 11, // Const enum
    HasComputedJSDocModifiers = 1 << 12, // Indicates the computed modifier flags include modifiers from JSDoc.

    Deprecated =         1 << 13, // Deprecated tag.
    HasComputedFlags =   1 << 29, // Modifier flags have been computed

    AccessibilityModifier = Public | Private | Protected,
    // Accessibility modifiers and 'readonly' can be attached to a parameter in a constructor to make it a property.
    ParameterPropertyModifier = AccessibilityModifier | Readonly,
    NonPublicAccessibilityModifier = Private | Protected,

    TypeScriptModifier = Ambient | Public | Private | Protected | Readonly | Abstract | Const,
    ExportDefault = Export | Default,
    All = Export | Ambient | Public | Private | Protected | Static | Readonly | Abstract | Async | Default | Const | Deprecated
};

ENUM_OPS(ModifierFlags)

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

ENUM_OPS(NodeFactoryFlags)

enum class OuterExpressionKinds : number {
    None = 0,
    Parentheses = 1 << 0,
    TypeAssertions = 1 << 1,
    NonNullAssertions = 1 << 2,
    PartiallyEmittedExpressions = 1 << 3,

    Assertions = TypeAssertions | NonNullAssertions,
    All = Parentheses | Assertions | PartiallyEmittedExpressions
};

ENUM_OPS(OuterExpressionKinds)

enum class JSDocState : number {
    BeginningOfLine,
    SawAsterisk,
    SavingComments,
    SavingBackticks, // Only NOTE used when parsing tag comments
};

enum class PropertyLikeParse : number {
    Property = 1 << 0,
    Parameter = 1 << 1,
    CallbackParameter = 1 << 2,
};

enum class InvalidPosition : number
{
    Value = -1
};

enum class Tristate : number {
    False,
    True,
    Unknown
};

enum class Comparison : number {
    LessThan    = -1,
    EqualTo     = 0,
    GreaterThan = 1
};

#endif // ENUMS_H