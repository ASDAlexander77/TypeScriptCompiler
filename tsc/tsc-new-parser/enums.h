#ifndef ENUMS_H
#define ENUMS_H

#include "config.h"

#define ENUM_OPS(x)                                                                                                                        \
    inline bool operator!(x lhs)                                                                                                           \
    {                                                                                                                                      \
        return (number)lhs == 0;                                                                                                           \
    }                                                                                                                                      \
                                                                                                                                           \
    inline x operator|(x lhs, x rhs)                                                                                                       \
    {                                                                                                                                      \
        return (x)((number)lhs | (number)rhs);                                                                                             \
    }                                                                                                                                      \
                                                                                                                                           \
    inline x operator&(x lhs, x rhs)                                                                                                       \
    {                                                                                                                                      \
        return (x)((number)lhs & (number)rhs);                                                                                             \
    }                                                                                                                                      \
                                                                                                                                           \
    inline x operator~(x rhs)                                                                                                              \
    {                                                                                                                                      \
        return (x)(~(number)rhs);                                                                                                          \
    }                                                                                                                                      \
                                                                                                                                           \
    inline x &operator|=(x &lhv, x rhv)                                                                                                    \
    {                                                                                                                                      \
        lhv = (x)((number)lhv | (number)rhv);                                                                                              \
        return lhv;                                                                                                                        \
    }                                                                                                                                      \
                                                                                                                                           \
    inline x &operator&=(x &lhv, x rhv)                                                                                                    \
    {                                                                                                                                      \
        lhv = (x)((number)lhv & (number)rhv);                                                                                              \
        return lhv;                                                                                                                        \
    }

namespace ts
{
enum class ScriptTarget : number
{
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

enum class CommentDirectiveType : number
{
    Undefined,
    ExpectError,
    Ignore
};

enum class LanguageVariant : number
{
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

enum class SignatureFlags : number
{
    None = 0,
    Yield = 1 << 0,
    Await = 1 << 1,
    Type = 1 << 2,
    IgnoreMissingOpenBrace = 1 << 4,
    JSDoc = 1 << 5,
};

ENUM_OPS(SignatureFlags)

enum class SpeculationKind : number
{
    TryParse,
    Lookahead,
    Reparse
};

enum class ScriptKind : number
{
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

enum class NodeFlags : number
{
    None               = 0,
    Let                = 1 << 0,  // Variable declaration
    Const              = 1 << 1,  // Variable declaration
    Using              = 1 << 2,  // Variable declaration
    AwaitUsing         = Const | Using, // Variable declaration (NOTE: on a single node these flags would otherwise be mutually exclusive)
    NestedNamespace    = 1 << 3,  // Namespace declaration
    Synthesized        = 1 << 4,  // Node was synthesized during transformation
    Namespace          = 1 << 5,  // Namespace declaration
    OptionalChain      = 1 << 6,  // Chained MemberExpression rooted to a pseudo-OptionalExpression
    ExportContext      = 1 << 7,  // Export context (initialized by binding)
    ContainsThis       = 1 << 8,  // Interface contains references to "this"
    HasImplicitReturn  = 1 << 9,  // If function implicitly returns on one of codepaths (initialized by binding)
    HasExplicitReturn  = 1 << 10,  // If function has explicit reachable return on one of codepaths (initialized by binding)
    GlobalAugmentation = 1 << 11,  // Set if module declaration is an augmentation for the global scope
    HasAsyncFunctions  = 1 << 12, // If the file has async functions (initialized by binding)
    DisallowInContext  = 1 << 13, // If node was parsed in a context where 'in-expressions' are not allowed
    YieldContext       = 1 << 14, // If node was parsed in the 'yield' context created when parsing a generator
    DecoratorContext   = 1 << 15, // If node was parsed as part of a decorator
    AwaitContext       = 1 << 16, // If node was parsed in the 'await' context created when parsing an async function
    DisallowConditionalTypesContext = 1 << 17, // If node was parsed in a context where conditional types are not allowed
    ThisNodeHasError   = 1 << 18, // If the parser encountered an error when parsing the code that created this node
    JavaScriptFile     = 1 << 19, // If node was parsed in a JavaScript
    ThisNodeOrAnySubNodesHasError = 1 << 20, // If this node or any of its children had an error
    HasAggregatedChildData = 1 << 21, // If we've computed data from children and cached it in this node

    // These flags will be set when the parser encounters a dynamic import expression or 'import.meta' to avoid
    // walking the tree if the flags are not set. However, these flags are just a approximation
    // (hence why it's named "PossiblyContainsDynamicImport") because once set, the flags never get cleared.
    // During editing, if a dynamic import is removed, incremental parsing will *NOT* clear this flag.
    // This means that the tree will always be traversed during module resolution, or when looking for external module indicators.
    // However, the removal operation should not occur often and in the case of the
    // removal, it is likely that users will add the import anyway.
    // The advantage of this approach is its simplicity. For the case of batch compilation,
    // we guarantee that users won't have to pay the price of walking the tree if a dynamic import isn't used.
    /** @internal */ PossiblyContainsDynamicImport = 1 << 22,
    /** @internal */ PossiblyContainsImportMeta    = 1 << 23,

    JSDoc                                          = 1 << 24, // If node was parsed inside jsdoc
    /** @internal */ Ambient                       = 1 << 25, // If node was inside an ambient context -- a declaration file, or inside something with the `declare` modifier.
    /** @internal */ InWithStatement               = 1 << 26, // If any ancestor of node was the `statement` of a WithStatement (not the `expression`)
    JsonFile                                       = 1 << 27, // If node was parsed in a Json
    /** @internal */ TypeCached                    = 1 << 28, // If a type was cached for node at any point
    /** @internal */ Deprecated                    = 1 << 29, // If has '@deprecated' JSDoc tag

    BlockScoped = Let | Const | Using,
    Constant = Const | Using,

    ReachabilityCheckFlags = HasImplicitReturn | HasExplicitReturn,
    ReachabilityAndEmitFlags = ReachabilityCheckFlags | HasAsyncFunctions,

    // Parsing context flags
    ContextFlags = DisallowInContext | DisallowConditionalTypesContext | YieldContext | DecoratorContext | AwaitContext | JavaScriptFile | InWithStatement | Ambient,

    // Exclude these flags when parsing a Type
    TypeExcludesFlags = YieldContext | AwaitContext,

    // Represents all flags that are potentially set once and
    // never cleared on SourceFiles which get re-used in between incremental parses.
    // See the comment above on `PossiblyContainsDynamicImport` and `PossiblyContainsImportMeta`.
    /** @internal */ PermanentlySetIncrementalFlags = PossiblyContainsDynamicImport | PossiblyContainsImportMeta,

    // The following flags repurpose other NodeFlags as different meanings for Identifier nodes
    /** @internal */ IdentifierHasExtendedUnicodeEscape = ContainsThis, // Indicates whether the identifier contains an extended unicode escape sequence
    /** @internal */ IdentifierIsInJSDocNamespace = HasAsyncFunctions, // Indicates whether the identifier is part of a JSDoc namespace
};

ENUM_OPS(NodeFlags)

enum class TokenFlags : number
{
    None = 0,
    /** @internal */
    PrecedingLineBreak = 1 << 0,
    /** @internal */
    PrecedingJSDocComment = 1 << 1,
    /** @internal */
    Unterminated = 1 << 2,
    /** @internal */
    ExtendedUnicodeEscape = 1 << 3,     // e.g. `\u{10ffff}`
    Scientific = 1 << 4,                // e.g. `10e2`
    Octal = 1 << 5,                     // e.g. `0777`
    HexSpecifier = 1 << 6,              // e.g. `0x00000000`
    BinarySpecifier = 1 << 7,           // e.g. `0b0110010000000000`
    OctalSpecifier = 1 << 8,            // e.g. `0o777`
    /** @internal */
    ContainsSeparator = 1 << 9,         // e.g. `0b1100_0101`
    /** @internal */
    UnicodeEscape = 1 << 10,            // e.g. `\u00a0`
    /** @internal */
    ContainsInvalidEscape = 1 << 11,    // e.g. `\uhello`
    /** @internal */
    HexEscape = 1 << 12,                // e.g. `\xa0`
    /** @internal */
    ContainsLeadingZero = 1 << 13,      // e.g. `0888`
    /** @internal */
    ContainsInvalidSeparator = 1 << 14, // e.g. `0_1`
    /** @internal */
    BinaryOrOctalSpecifier = BinarySpecifier | OctalSpecifier,
    /** @internal */
    WithSpecifier = HexSpecifier | BinaryOrOctalSpecifier,
    /** @internal */
    StringLiteralFlags = HexEscape | UnicodeEscape | ExtendedUnicodeEscape | ContainsInvalidEscape,
    /** @internal */
    NumericLiteralFlags = Scientific | Octal | ContainsLeadingZero | WithSpecifier | ContainsSeparator | ContainsInvalidSeparator,
    /** @internal */
    TemplateLiteralLikeFlags = HexEscape | UnicodeEscape | ExtendedUnicodeEscape | ContainsInvalidEscape,
    /** @internal */
    IsInvalid = Octal | ContainsLeadingZero | ContainsInvalidSeparator | ContainsInvalidEscape,
};

ENUM_OPS(TokenFlags)

enum class OperatorPrecedence : number
{
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

enum class ParsingContext : number
{
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
    ImportOrExportSpecifiers,  // Named import clause's import specifier list,
    ImportAttributes,          // Import attributes
    JSDocComment,              // Parsing via JSDocParser
    Count,                     // Number of parsing contexts
};

ENUM_OPS(ParsingContext)

enum class TransformFlags : number
{
    None = 0,

    // Facts
    // - Flags used to indicate that a node or subtree contains syntax that requires transformation.
    ContainsTypeScript = 1 << 0,
    ContainsJsx = 1 << 1,
    ContainsESNext = 1 << 2,
    ContainsES2022 = 1 << 3,
    ContainsES2021 = 1 << 4,
    ContainsES2020 = 1 << 5,
    ContainsES2019 = 1 << 6,
    ContainsES2018 = 1 << 7,
    ContainsES2017 = 1 << 8,
    ContainsES2016 = 1 << 9,
    ContainsES2015 = 1 << 10,
    ContainsGenerator = 1 << 11,
    ContainsDestructuringAssignment = 1 << 12,

    // Markers
    // - Flags used to indicate that a subtree contains a specific transformation.
    ContainsTypeScriptClassSyntax = 1 << 13, // Property Initializers, Parameter Property Initializers
    ContainsLexicalThis = 1 << 14,
    ContainsRestOrSpread = 1 << 15,
    ContainsObjectRestOrSpread = 1 << 16,
    ContainsComputedPropertyName = 1 << 17,
    ContainsBlockScopedBinding = 1 << 18,
    ContainsBindingPattern = 1 << 19,
    ContainsYield = 1 << 20,
    ContainsAwait = 1 << 21,
    ContainsHoistedDeclarationOrCompletion = 1 << 22,
    ContainsDynamicImport = 1 << 23,
    ContainsClassFields = 1 << 24,
    ContainsDecorators = 1 << 25,
    ContainsPossibleTopLevelAwait = 1 << 26,
    ContainsLexicalSuper = 1 << 27,
    ContainsUpdateExpressionForIdentifier = 1 << 28,
    ContainsPrivateIdentifierInExpression = 1 << 29,

    HasComputedFlags = 1 << 31, // Transform flags have been computed.

    // Assertions
    // - Bitmasks that are used to assert facts about the syntax of a node and its subtree.
    AssertTypeScript = ContainsTypeScript,
    AssertJsx = ContainsJsx,
    AssertESNext = ContainsESNext,
    AssertES2022 = ContainsES2022,
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
    FunctionExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsLexicalThis | ContainsLexicalSuper | ContainsBlockScopedBinding | ContainsYield | ContainsAwait | ContainsHoistedDeclarationOrCompletion | ContainsBindingPattern | ContainsObjectRestOrSpread | ContainsPossibleTopLevelAwait,
    ConstructorExcludes = NodeExcludes | ContainsLexicalThis | ContainsLexicalSuper | ContainsBlockScopedBinding | ContainsYield | ContainsAwait | ContainsHoistedDeclarationOrCompletion | ContainsBindingPattern | ContainsObjectRestOrSpread | ContainsPossibleTopLevelAwait,
    MethodOrAccessorExcludes = NodeExcludes | ContainsLexicalThis | ContainsLexicalSuper | ContainsBlockScopedBinding | ContainsYield | ContainsAwait | ContainsHoistedDeclarationOrCompletion | ContainsBindingPattern | ContainsObjectRestOrSpread,
    PropertyExcludes = NodeExcludes | ContainsLexicalThis | ContainsLexicalSuper,
    ClassExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsComputedPropertyName,
    ModuleExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsLexicalThis | ContainsLexicalSuper | ContainsBlockScopedBinding | ContainsHoistedDeclarationOrCompletion | ContainsPossibleTopLevelAwait,
    TypeExcludes = ~ContainsTypeScript,
    ObjectLiteralExcludes = NodeExcludes | ContainsTypeScriptClassSyntax | ContainsComputedPropertyName | ContainsObjectRestOrSpread,
    ArrayLiteralOrCallOrNewExcludes = NodeExcludes | ContainsRestOrSpread,
    VariableDeclarationListExcludes = NodeExcludes | ContainsBindingPattern | ContainsObjectRestOrSpread,
    ParameterExcludes = NodeExcludes,
    CatchClauseExcludes = NodeExcludes | ContainsObjectRestOrSpread,
    BindingPatternExcludes = NodeExcludes | ContainsRestOrSpread,
    ContainsLexicalThisOrSuper = ContainsLexicalThis | ContainsLexicalSuper,

    // Propagating flags
    // - Bitmasks for flags that should propagate from a child
    PropertyNamePropagatingFlags = ContainsLexicalThis | ContainsLexicalSuper,
    // Masks
    // - Additional bitmasks
};

ENUM_OPS(TransformFlags)

enum class InternalFlags : number
{
    None = 0,

    // Internal
    ForceConst = 1 << 0,
    ForceConstRef = 1 << 1,
    ForceVirtual = 1 << 2,
    VarsInObjectContext = 1 << 3,
    ForAwait = 1 << 4,
    SuppressConstructorCall = 1 << 5,
    ThisArgAlias = 1 << 6,
    DllExport = 1 << 7,
    DllImport = 1 << 8,
    IsPublic = 1 << 9,
    GenerationProcessed = 1 << 10
};

ENUM_OPS(InternalFlags)

enum class GeneratedIdentifierFlags : number
{
    // Kinds
    None = 0,                   // Not automatically generated.
    /*@internal*/ Auto = 1,     // Automatically generated identifier.
    /*@internal*/ Loop = 2,     // Automatically generated identifier with a preference for '_i'.
    /*@internal*/ Unique = 3,   // Unique name based on the 'text' property.
    /*@internal*/ Node = 4,     // Unique name based on the node in the 'original' property.
    /*@internal*/ KindMask = 7, // Mask to extract the kind of identifier from its flags.

    // Flags
    ReservedInNestedScopes = 1 << 3, // Reserve the generated name in nested scopes
    Optimistic = 1 << 4,             // First instance won't use '_#' if there's no conflict
    FileLevel = 1 << 5,              // Use only the file identifiers list and not generated names to search for conflicts
    AllowNameSubstitution = 1 << 6,  // Used by `module.ts` to indicate generated nodes which can have substitutions performed upon them (as
                                     // they were generated by an earlier transform phase)
};

enum class ModifierFlags : number
{
    None = 0,
    Export = 1 << 0,                     // Declarations
    Ambient = 1 << 1,                    // Declarations
    Public = 1 << 2,                     // Property/Method
    Private = 1 << 3,                    // Property/Method
    Protected = 1 << 4,                  // Property/Method
    Static = 1 << 5,                     // Property/Method
    Readonly = 1 << 6,                   // Property/Method
    Abstract = 1 << 7,                   // Class/Method/ConstructSignature
    Async = 1 << 8,                      // Property/Method/Function
    Default = 1 << 9,                    // Function/Class (export default declaration)
    Const = 1 << 11,                     // Const enum
    HasComputedJSDocModifiers = 1 << 12, // Indicates the computed modifier flags include modifiers from JSDoc.

    Deprecated = 1 << 13,       // Deprecated tag.
    HasComputedFlags = 1 << 29, // Modifier flags have been computed

    AccessibilityModifier = Public | Private | Protected,
    // Accessibility modifiers and 'readonly' can be attached to a parameter in a constructor to make it a property.
    ParameterPropertyModifier = AccessibilityModifier | Readonly,
    NonPublicAccessibilityModifier = Private | Protected,

    TypeScriptModifier = Ambient | Public | Private | Protected | Readonly | Abstract | Const,
    ExportDefault = Export | Default,
    All = Export | Ambient | Public | Private | Protected | Static | Readonly | Abstract | Async | Default | Const | Deprecated
};

ENUM_OPS(ModifierFlags)

enum class NodeFactoryFlags : number
{
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

enum class OuterExpressionKinds : number
{
    None = 0,
    Parentheses = 1 << 0,
    TypeAssertions = 1 << 1,
    NonNullAssertions = 1 << 2,
    PartiallyEmittedExpressions = 1 << 3,

    Assertions = TypeAssertions | NonNullAssertions,
    All = Parentheses | Assertions | PartiallyEmittedExpressions
};

ENUM_OPS(OuterExpressionKinds)

enum class JSDocState : number
{
    BeginningOfLine,
    SawAsterisk,
    SavingComments,
    SavingBackticks, // Only NOTE used when parsing tag comments
};

enum class PropertyLikeParse : number
{
    Property = 1 << 0,
    Parameter = 1 << 1,
    CallbackParameter = 1 << 2,
};

enum class InvalidPosition : number
{
    Value = -1
};

enum class Tristate : number
{
    False,
    True,
    Unknown
};

inline bool operator!(Tristate rhs)
{
    return rhs != Tristate::True;
}

enum class Comparison : number
{
    LessThan = -1,
    EqualTo = 0,
    GreaterThan = 1
};

enum class SymbolFlags : number
{
    None = 0,
    FunctionScopedVariable = 1 << 0, // Variable (var) or parameter
    BlockScopedVariable = 1 << 1,    // A block-scoped variable (let or const)
    Property = 1 << 2,               // Property or enum member
    EnumMember = 1 << 3,             // Enum member
    Function = 1 << 4,               // Function
    Class = 1 << 5,                  // Class
    Interface = 1 << 6,              // Interface
    ConstEnum = 1 << 7,              // Const enum
    RegularEnum = 1 << 8,            // Enum
    ValueModule = 1 << 9,            // Instantiated module
    NamespaceModule = 1 << 10,       // Uninstantiated module
    TypeLiteral = 1 << 11,           // Type Literal or mapped type
    ObjectLiteral = 1 << 12,         // Object Literal
    Method = 1 << 13,                // Method
    Constructor = 1 << 14,           // Constructor
    GetAccessor = 1 << 15,           // Get accessor
    SetAccessor = 1 << 16,           // Set accessor
    Signature = 1 << 17,             // Call, construct, or index signature
    TypeParameter = 1 << 18,         // Type parameter
    TypeAlias = 1 << 19,             // Type alias
    ExportValue = 1 << 20,           // Exported value marker (see comment in declareModuleMember in binder)
    Alias = 1 << 21,                 // An alias for another symbol (see comment in isAliasSymbolDeclaration in checker)
    Prototype = 1 << 22,             // Prototype property (no source representation)
    ExportStar = 1 << 23,            // Export * declaration
    Optional = 1 << 24,              // Optional property
    Transient = 1 << 25,             // Transient symbol (created during type check)
    Assignment = 1 << 26,            // Assignment treated as declaration (eg `this.prop = 1`)
    ModuleExports = 1 << 27,         // Symbol for CommonJS `module` of `module.exports`
    /* @internal */
    All = FunctionScopedVariable | BlockScopedVariable | Property | EnumMember | Function | Class | Interface | ConstEnum | RegularEnum |
          ValueModule | NamespaceModule | TypeLiteral | ObjectLiteral | Method | Constructor | GetAccessor | SetAccessor | Signature |
          TypeParameter | TypeAlias | ExportValue | Alias | Prototype | ExportStar | Optional | Transient,

    Enum = RegularEnum | ConstEnum,
    Variable = FunctionScopedVariable | BlockScopedVariable,
    Value = Variable | Property | EnumMember | ObjectLiteral | Function | Class | Enum | ValueModule | Method | GetAccessor | SetAccessor,
    Type = Class | Interface | Enum | EnumMember | TypeLiteral | TypeParameter | TypeAlias,
    Namespace = ValueModule | NamespaceModule | Enum,
    Module = ValueModule | NamespaceModule,
    Accessor = GetAccessor | SetAccessor,

    // Variables can be redeclared, but can not redeclare a block-scoped declaration with the
    // same name, or any other value that is not a variable, e.g. ValueModule or Class
    FunctionScopedVariableExcludes = Value & ~FunctionScopedVariable,

    // Block-scoped declarations are not allowed to be re-declared
    // they can not merge with anything in the value space
    BlockScopedVariableExcludes = Value,

    ParameterExcludes = Value,
    PropertyExcludes = None,
    EnumMemberExcludes = Value | Type,
    FunctionExcludes = Value & ~(Function | ValueModule | Class),
    ClassExcludes = (Value | Type) & ~(ValueModule | Interface | Function), // class-interface mergability done in checker.ts
    InterfaceExcludes = Type & ~(Interface | Class),
    RegularEnumExcludes = (Value | Type) & ~(RegularEnum | ValueModule), // regular enums merge only with regular enums and modules
    ConstEnumExcludes = (Value | Type) & ~ConstEnum,                     // const enums merge only with const enums
    ValueModuleExcludes = Value & ~(Function | Class | RegularEnum | ValueModule),
    NamespaceModuleExcludes = 0,
    MethodExcludes = Value & ~Method,
    GetAccessorExcludes = Value & ~SetAccessor,
    SetAccessorExcludes = Value & ~GetAccessor,
    TypeParameterExcludes = Type & ~TypeParameter,
    TypeAliasExcludes = Type,
    AliasExcludes = Alias,

    ModuleMember = Variable | Function | Class | Interface | Enum | Module | TypeAlias | Alias,

    ExportHasLocal = Function | Class | Enum | ValueModule,

    BlockScoped = BlockScopedVariable | Class | Enum,

    PropertyOrAccessor = Property | Accessor,

    ClassMember = Method | Accessor | Property,

    /* @internal */
    ExportSupportsDefaultModifier = Class | Function | Interface,

    /* @internal */
    ExportDoesNotSupportDefaultModifier = ~ExportSupportsDefaultModifier,

    /* @internal */
    // The set of things we consider semantically classifiable.  Used to speed up the LS during
    // classification.
    Classifiable = Class | Enum | TypeAlias | Interface | TypeParameter | Module | Alias,

    /* @internal */
    LateBindingContainer = Class | Interface | TypeLiteral | ObjectLiteral | Function,
};

enum class TypeFlags : number
{
    Any = 1 << 0,
    Unknown = 1 << 1,
    String = 1 << 2,
    Number = 1 << 3,
    Boolean = 1 << 4,
    Enum = 1 << 5,
    BigInt = 1 << 6,
    StringLiteral = 1 << 7,
    NumberLiteral = 1 << 8,
    BooleanLiteral = 1 << 9,
    EnumLiteral = 1 << 10, // Always combined with StringLiteral, NumberLiteral, or Union
    BigIntLiteral = 1 << 11,
    ESSymbol = 1 << 12,       // Type of symbol primitive introduced in ES6
    UniqueESSymbol = 1 << 13, // unique symbol
    Void = 1 << 14,
    Undefined = 1 << 15,
    Null = 1 << 16,
    Never = 1 << 17,           // Never type
    TypeParameter = 1 << 18,   // Type parameter
    Object = 1 << 19,          // Object type
    Union = 1 << 20,           // Union (T | U)
    Intersection = 1 << 21,    // Intersection (T & U)
    Index = 1 << 22,           // keyof T
    IndexedAccess = 1 << 23,   // T[K]
    Conditional = 1 << 24,     // T extends U ? X : Y
    Substitution = 1 << 25,    // Type parameter substitution
    NonPrimitive = 1 << 26,    // intrinsic object type
    TemplateLiteral = 1 << 27, // Template literal type
    StringMapping = 1 << 28,   // Uppercase/Lowercase type

    /* @internal */
    AnyOrUnknown = Any | Unknown,
    /* @internal */
    Nullable = Undefined | Null,
    Literal = StringLiteral | NumberLiteral | BigIntLiteral | BooleanLiteral,
    Unit = Literal | UniqueESSymbol | Nullable,
    StringOrNumberLiteral = StringLiteral | NumberLiteral,
    /* @internal */
    StringOrNumberLiteralOrUnique = StringLiteral | NumberLiteral | UniqueESSymbol,
    /* @internal */
    DefinitelyFalsy = StringLiteral | NumberLiteral | BigIntLiteral | BooleanLiteral | Void | Undefined | Null,
    PossiblyFalsy = DefinitelyFalsy | String | Number | BigInt | Boolean,
    /* @internal */
    Intrinsic =
        Any | Unknown | String | Number | BigInt | Boolean | BooleanLiteral | ESSymbol | Void | Undefined | Null | Never | NonPrimitive,
    /* @internal */
    Primitive = String | Number | BigInt | Boolean | Enum | EnumLiteral | ESSymbol | Void | Undefined | Null | Literal | UniqueESSymbol,
    StringLike = String | StringLiteral | TemplateLiteral | StringMapping,
    NumberLike = Number | NumberLiteral | Enum,
    BigIntLike = BigInt | BigIntLiteral,
    BooleanLike = Boolean | BooleanLiteral,
    EnumLike = Enum | EnumLiteral,
    ESSymbolLike = ESSymbol | UniqueESSymbol,
    VoidLike = Void | Undefined,
    /* @internal */
    DisjointDomains = NonPrimitive | StringLike | NumberLike | BigIntLike | BooleanLike | ESSymbolLike | VoidLike | Null,
    UnionOrIntersection = Union | Intersection,
    StructuredType = Object | Union | Intersection,
    TypeVariable = TypeParameter | IndexedAccess,
    InstantiableNonPrimitive = TypeVariable | Conditional | Substitution,
    InstantiablePrimitive = Index | TemplateLiteral | StringMapping,
    Instantiable = InstantiableNonPrimitive | InstantiablePrimitive,
    StructuredOrInstantiable = StructuredType | Instantiable,
    /* @internal */
    ObjectFlagsType = Any | Nullable | Never | Object | Union | Intersection,
    /* @internal */
    Simplifiable = IndexedAccess | Conditional,
    /* @internal */
    Substructure = Object | Union | Intersection | Index | IndexedAccess | Conditional | Substitution | TemplateLiteral | StringMapping,
    // 'Narrowable' types are types where narrowing actually narrows.
    // This *should* be every type other than null, undefined, void, and never
    Narrowable = Any | Unknown | StructuredOrInstantiable | StringLike | NumberLike | BigIntLike | BooleanLike | ESSymbol | UniqueESSymbol |
                 NonPrimitive,
    /* @internal */
    NotPrimitiveUnion = Any | Unknown | Enum | Void | Never | Object | Intersection | Instantiable,
    // The following flags are aggregated during union and intersection type construction
    /* @internal */
    IncludesMask = Any | Unknown | Primitive | Never | Object | Union | Intersection | NonPrimitive | TemplateLiteral,
    // The following flags are used for different purposes during union and intersection type construction
    /* @internal */
    IncludesStructuredOrInstantiable = TypeParameter,
    /* @internal */
    IncludesNonWideningType = Index,
    /* @internal */
    IncludesWildcard = IndexedAccess,
    /* @internal */
    IncludesEmptyObject = Conditional,
};

enum class Associativity : number
{
    Left,
    Right
};

enum class JSDocParsingMode : number {
    /**
     * Always parse JSDoc comments and include them in the AST.
     *
     * This is the default if no mode is provided.
     */
    ParseAll,
    /**
     * Never parse JSDoc comments, mo matter the file type.
     */
    ParseNone,
    /**
     * Parse only JSDoc comments which are needed to provide correct type errors.
     *
     * This will always parse JSDoc in non-TS files, but only parse JSDoc comments
     * containing `@see` and `@link` in TS files.
     */
    ParseForTypeErrors,
    /**
     * Parse only JSDoc comments which are needed to provide correct type info.
     *
     * This will always parse JSDoc in non-TS files, but never in TS files.
     *
     * Note: Do not use this mode if you require accurate type errors; use {@link ParseForTypeErrors} instead.
     */
    ParseForTypeInfo,
};


enum class EmitFlags : number {
    None = 0,
    SingleLine = 1 << 0,                    // The contents of this node should be emitted on a single line.
    MultiLine = 1 << 1,
    AdviseOnEmitNode = 1 << 2,              // The printer should invoke the onEmitNode callback when printing this node.
    NoSubstitution = 1 << 3,                // Disables further substitution of an expression.
    CapturesThis = 1 << 4,                  // The function captures a lexical `this`
    NoLeadingSourceMap = 1 << 5,            // Do not emit a leading source map location for this node.
    NoTrailingSourceMap = 1 << 6,           // Do not emit a trailing source map location for this node.
    NoSourceMap = NoLeadingSourceMap | NoTrailingSourceMap, // Do not emit a source map location for this node.
    NoNestedSourceMaps = 1 << 7,            // Do not emit source map locations for children of this node.
    NoTokenLeadingSourceMaps = 1 << 8,      // Do not emit leading source map location for token nodes.
    NoTokenTrailingSourceMaps = 1 << 9,     // Do not emit trailing source map location for token nodes.
    NoTokenSourceMaps = NoTokenLeadingSourceMaps | NoTokenTrailingSourceMaps, // Do not emit source map locations for tokens of this node.
    NoLeadingComments = 1 << 10,            // Do not emit leading comments for this node.
    NoTrailingComments = 1 << 11,           // Do not emit trailing comments for this node.
    NoComments = NoLeadingComments | NoTrailingComments, // Do not emit comments for this node.
    NoNestedComments = 1 << 12,
    HelperName = 1 << 13,                   // The Identifier refers to an *unscoped* emit helper (one that is emitted at the top of the file)
    ExportName = 1 << 14,                   // Ensure an export prefix is added for an identifier that points to an exported declaration with a local name (see SymbolFlags.ExportHasLocal).
    LocalName = 1 << 15,                    // Ensure an export prefix is not added for an identifier that points to an exported declaration.
    InternalName = 1 << 16,                 // The name is internal to an ES5 class body function.
    Indented = 1 << 17,                     // Adds an explicit extra indentation level for class and function bodies when printing (used to match old emitter).
    NoIndentation = 1 << 18,                // Do not indent the node.
    AsyncFunctionBody = 1 << 19,
    ReuseTempVariableScope = 1 << 20,       // Reuse the existing temp variable scope during emit.
    CustomPrologue = 1 << 21,               // Treat the statement as if it were a prologue directive (NOTE: Prologue directives are *not* transformed).
    NoHoisting = 1 << 22,                   // Do not hoist this declaration in --module system
    Iterator = 1 << 23,                     // The expression to a `yield*` should be treated as an Iterator when down-leveling, not an Iterable.
    NoAsciiEscaping = 1 << 24,              // When synthesizing nodes that lack an original node or textSourceNode, we want to write the text on the node with ASCII escaping substitutions.
};

} // namespace ts

#endif // ENUMS_H