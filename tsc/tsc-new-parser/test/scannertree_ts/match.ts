import * as fs from 'fs';
import { execSync } from 'child_process';
import * as ts from 'typescript';

enum SyntaxKindMapped {
    Unknown,
    EndOfFileToken,
    SingleLineCommentTrivia,
    MultiLineCommentTrivia,
    NewLineTrivia,
    WhitespaceTrivia,
    // We detect and preserve #! on the first line
    ShebangTrivia,
    // We detect and provide better error recovery when we encounter a git merge marker.  This
    // allows us to edit files with git-conflict markers in them in a much more pleasant manner.
    ConflictMarkerTrivia,
    // Literals
    NumericLiteral,
    BigIntLiteral,
    StringLiteral,
    JsxText,
    JsxTextAllWhiteSpaces,
    RegularExpressionLiteral,
    NoSubstitutionTemplateLiteral,
    // Pseudo-literals
    TemplateHead,
    TemplateMiddle,
    TemplateTail,
    // Punctuation
    OpenBraceToken,
    CloseBraceToken,
    OpenParenToken,
    CloseParenToken,
    OpenBracketToken,
    CloseBracketToken,
    DotToken,
    DotDotDotToken,
    SemicolonToken,
    CommaToken,
    QuestionDotToken,
    LessThanToken,
    LessThanSlashToken,
    GreaterThanToken,
    LessThanEqualsToken,
    GreaterThanEqualsToken,
    EqualsEqualsToken,
    ExclamationEqualsToken,
    EqualsEqualsEqualsToken,
    ExclamationEqualsEqualsToken,
    EqualsGreaterThanToken,
    PlusToken,
    MinusToken,
    AsteriskToken,
    AsteriskAsteriskToken,
    SlashToken,
    PercentToken,
    PlusPlusToken,
    MinusMinusToken,
    LessThanLessThanToken,
    GreaterThanGreaterThanToken,
    GreaterThanGreaterThanGreaterThanToken,
    AmpersandToken,
    BarToken,
    CaretToken,
    ExclamationToken,
    TildeToken,
    AmpersandAmpersandToken,
    BarBarToken,
    QuestionToken,
    ColonToken,
    AtToken,
    QuestionQuestionToken,
    /** Only the JSDoc scanner produces BacktickToken. The normal scanner produces NoSubstitutionTemplateLiteral and related kinds. */
    BacktickToken,
    // Assignments
    EqualsToken,
    PlusEqualsToken,
    MinusEqualsToken,
    AsteriskEqualsToken,
    AsteriskAsteriskEqualsToken,
    SlashEqualsToken,
    PercentEqualsToken,
    LessThanLessThanEqualsToken,
    GreaterThanGreaterThanEqualsToken,
    GreaterThanGreaterThanGreaterThanEqualsToken,
    AmpersandEqualsToken,
    BarEqualsToken,
    BarBarEqualsToken,
    AmpersandAmpersandEqualsToken,
    QuestionQuestionEqualsToken,
    CaretEqualsToken,
    // Identifiers and PrivateIdentifiers
    Identifier,
    PrivateIdentifier,
    // Reserved words
    BreakKeyword,
    CaseKeyword,
    CatchKeyword,
    ClassKeyword,
    ConstKeyword,
    ContinueKeyword,
    DebuggerKeyword,
    DefaultKeyword,
    DeleteKeyword,
    DoKeyword,
    ElseKeyword,
    EnumKeyword,
    ExportKeyword,
    ExtendsKeyword,
    FalseKeyword,
    FinallyKeyword,
    ForKeyword,
    FunctionKeyword,
    IfKeyword,
    ImportKeyword,
    InKeyword,
    InstanceOfKeyword,
    NewKeyword,
    NullKeyword,
    ReturnKeyword,
    SuperKeyword,
    SwitchKeyword,
    ThisKeyword,
    ThrowKeyword,
    TrueKeyword,
    TryKeyword,
    TypeOfKeyword,
    VarKeyword,
    VoidKeyword,
    WhileKeyword,
    WithKeyword,
    // Strict mode reserved words
    ImplementsKeyword,
    InterfaceKeyword,
    LetKeyword,
    PackageKeyword,
    PrivateKeyword,
    ProtectedKeyword,
    PublicKeyword,
    StaticKeyword,
    YieldKeyword,
    // Contextual keywords
    AbstractKeyword,
    AsKeyword,
    AssertsKeyword,
    AnyKeyword,
    AsyncKeyword,
    AwaitKeyword,
    BooleanKeyword,
    ConstructorKeyword,
    DeclareKeyword,
    GetKeyword,
    InferKeyword,
    IntrinsicKeyword,
    IsKeyword,
    KeyOfKeyword,
    ModuleKeyword,
    NamespaceKeyword,
    NeverKeyword,
    ReadonlyKeyword,
    RequireKeyword,
    NumberKeyword,
    ObjectKeyword,
    SetKeyword,
    StringKeyword,
    SymbolKeyword,
    TypeKeyword,
    UndefinedKeyword,
    UniqueKeyword,
    UnknownKeyword,
    FromKeyword,
    GlobalKeyword,
    BigIntKeyword,
    OfKeyword, // LastKeyword and LastToken and LastContextualKeyword

    // Parse tree nodes

    // Names
    QualifiedName,
    ComputedPropertyName,
    // Signature elements
    TypeParameter,
    Parameter,
    Decorator,
    // TypeMember
    PropertySignature,
    PropertyDeclaration,
    MethodSignature,
    MethodDeclaration,
    Constructor,
    GetAccessor,
    SetAccessor,
    CallSignature,
    ConstructSignature,
    IndexSignature,
    // Type
    TypePredicate,
    TypeReference,
    FunctionType,
    ConstructorType,
    TypeQuery,
    TypeLiteral,
    ArrayType,
    TupleType,
    OptionalType,
    RestType,
    UnionType,
    IntersectionType,
    ConditionalType,
    InferType,
    ParenthesizedType,
    ThisType,
    TypeOperator,
    IndexedAccessType,
    MappedType,
    LiteralType,
    NamedTupleMember,
    TemplateLiteralType,
    TemplateLiteralTypeSpan,
    ImportType,
    // Binding patterns
    ObjectBindingPattern,
    ArrayBindingPattern,
    BindingElement,
    // Expression
    ArrayLiteralExpression,
    ObjectLiteralExpression,
    PropertyAccessExpression,
    ElementAccessExpression,
    CallExpression,
    NewExpression,
    TaggedTemplateExpression,
    TypeAssertionExpression,
    ParenthesizedExpression,
    FunctionExpression,
    ArrowFunction,
    DeleteExpression,
    TypeOfExpression,
    VoidExpression,
    AwaitExpression,
    PrefixUnaryExpression,
    PostfixUnaryExpression,
    BinaryExpression,
    ConditionalExpression,
    TemplateExpression,
    YieldExpression,
    SpreadElement,
    ClassExpression,
    OmittedExpression,
    ExpressionWithTypeArguments,
    AsExpression,
    NonNullExpression,
    MetaProperty,
    SyntheticExpression,

    // Misc
    TemplateSpan,
    SemicolonClassElement,
    // Element
    Block,
    EmptyStatement,
    VariableStatement,
    ExpressionStatement,
    IfStatement,
    DoStatement,
    WhileStatement,
    ForStatement,
    ForInStatement,
    ForOfStatement,
    ContinueStatement,
    BreakStatement,
    ReturnStatement,
    WithStatement,
    SwitchStatement,
    LabeledStatement,
    ThrowStatement,
    TryStatement,
    DebuggerStatement,
    VariableDeclaration,
    VariableDeclarationList,
    FunctionDeclaration,
    ClassDeclaration,
    InterfaceDeclaration,
    TypeAliasDeclaration,
    EnumDeclaration,
    ModuleDeclaration,
    ModuleBlock,
    CaseBlock,
    NamespaceExportDeclaration,
    ImportEqualsDeclaration,
    ImportDeclaration,
    ImportClause,
    NamespaceImport,
    NamedImports,
    ImportSpecifier,
    ExportAssignment,
    ExportDeclaration,
    NamedExports,
    NamespaceExport,
    ExportSpecifier,
    MissingDeclaration,

    // Module references
    ExternalModuleReference,

    // JSX
    JsxElement,
    JsxSelfClosingElement,
    JsxOpeningElement,
    JsxClosingElement,
    JsxFragment,
    JsxOpeningFragment,
    JsxClosingFragment,
    JsxAttribute,
    JsxAttributes,
    JsxSpreadAttribute,
    JsxExpression,

    // Clauses
    CaseClause,
    DefaultClause,
    HeritageClause,
    CatchClause,

    // Property assignments
    PropertyAssignment,
    ShorthandPropertyAssignment,
    SpreadAssignment,

    // Enum
    EnumMember,
    // Unparsed
    UnparsedPrologue,
    UnparsedPrepend,
    UnparsedText,
    UnparsedInternalText,
    UnparsedSyntheticReference,

    // Top-level nodes
    SourceFile,
    Bundle,
    UnparsedSource,
    InputFiles,

    // JSDoc nodes
    JSDocTypeExpression,
    JSDocNameReference,
    JSDocAllType, // The * type
    JSDocUnknownType, // The ? type
    JSDocNullableType,
    JSDocNonNullableType,
    JSDocOptionalType,
    JSDocFunctionType,
    JSDocVariadicType,
    JSDocNamepathType, // https://jsdoc.app/about-namepaths.html
    JSDocComment,
    JSDocTypeLiteral,
    JSDocSignature,
    JSDocTag,
    JSDocAugmentsTag,
    JSDocImplementsTag,
    JSDocAuthorTag,
    JSDocDeprecatedTag,
    JSDocClassTag,
    JSDocPublicTag,
    JSDocPrivateTag,
    JSDocProtectedTag,
    JSDocReadonlyTag,
    JSDocCallbackTag,
    JSDocEnumTag,
    JSDocParameterTag,
    JSDocReturnTag,
    JSDocThisTag,
    JSDocTypeTag,
    JSDocTemplateTag,
    JSDocTypedefTag,
    JSDocSeeTag,
    JSDocPropertyTag,

    // Synthesized list
    SyntaxList,

    // Transformation nodes
    NotEmittedStatement,
    PartiallyEmittedExpression,
    CommaListExpression,
    MergeDeclarationMarker,
    EndOfDeclarationMarker,
    SyntheticReferenceExpression,

    // Enum value count
    Count
}

enum SyntaxKindMapped2 {
    FirstAssignment = SyntaxKindMapped.EqualsToken,
    LastAssignment = SyntaxKindMapped.CaretEqualsToken,
    FirstCompoundAssignment = SyntaxKindMapped.PlusEqualsToken,
    LastCompoundAssignment = SyntaxKindMapped.CaretEqualsToken,
    FirstReservedWord = SyntaxKindMapped.BreakKeyword,
    LastReservedWord = SyntaxKindMapped.WithKeyword,
    FirstKeyword = SyntaxKindMapped.BreakKeyword,
    LastKeyword = SyntaxKindMapped.OfKeyword,
    FirstFutureReservedWord = SyntaxKindMapped.ImplementsKeyword,
    LastFutureReservedWord = SyntaxKindMapped.YieldKeyword,
    FirstTypeNode = SyntaxKindMapped.TypePredicate,
    LastTypeNode = SyntaxKindMapped.ImportType,
    FirstPunctuation = SyntaxKindMapped.OpenBraceToken,
    LastPunctuation = SyntaxKindMapped.CaretEqualsToken,
    FirstToken = SyntaxKindMapped.Unknown,
    LastToken = SyntaxKindMapped.OfKeyword,
    FirstTriviaToken = SyntaxKindMapped.SingleLineCommentTrivia,
    LastTriviaToken = SyntaxKindMapped.ConflictMarkerTrivia,
    FirstLiteralToken = SyntaxKindMapped.NumericLiteral,
    LastLiteralToken = SyntaxKindMapped.NoSubstitutionTemplateLiteral,
    FirstTemplateToken = SyntaxKindMapped.NoSubstitutionTemplateLiteral,
    LastTemplateToken = SyntaxKindMapped.TemplateTail,
    FirstBinaryOperator = SyntaxKindMapped.LessThanToken,
    LastBinaryOperator = SyntaxKindMapped.CaretEqualsToken,
    FirstStatement = SyntaxKindMapped.VariableStatement,
    LastStatement = SyntaxKindMapped.DebuggerStatement,
    FirstNode = SyntaxKindMapped.QualifiedName,
    FirstJSDocNode = SyntaxKindMapped.JSDocTypeExpression,
    LastJSDocNode = SyntaxKindMapped.JSDocPropertyTag,
    FirstJSDocTagNode = SyntaxKindMapped.JSDocTag,
    LastJSDocTagNode = SyntaxKindMapped.JSDocPropertyTag,
    FirstContextualKeyword = SyntaxKindMapped.AbstractKeyword,
    LastContextualKeyword = SyntaxKindMapped.OfKeyword
};

function printTree(filePath) {
    const dataStr = ts.sys.readFile(filePath);
    const scanner = ts.createScanner(ts.ScriptTarget.Latest, true, ts.LanguageVariant.Standard, dataStr);

    let result = "";

    let token = ts.SyntaxKind.Unknown;
    while (token != ts.SyntaxKind.EndOfFileToken) {
        token = scanner.scan();
        const strToken = ts.SyntaxKind[token];
        result += (SyntaxKindMapped[strToken] || SyntaxKindMapped2[strToken]) + " " + scanner.getTokenText() + "\n";
    }

    return result;
}

try {
    const fld = process.argv[2] || "G:/Dev/TypeScript/tests/cases/compiler";
    const files = fs.readdirSync(fld);
    for (const file of files) {
        //const data = await fs.readFile(fld + "/" + file);

        if (file == "collisionCodeGenModuleWithUnicodeNames.ts"
            || file == "constructorWithIncompleteTypeAnnotation.ts"
            || file == "extendedUnicodePlaneIdentifiers.ts"
            || file == "extendedUnicodePlaneIdentifiersJSDoc.ts"
            || file == "fileWithNextLine1.ts"
            || file == "parseErrorInHeritageClause1.ts"
            || file == "sourceMap-LineBreaks.ts"
            || file == "unicodeIdentifierName2.ts"
            || file == "unicodeIdentifierNames.ts"
            || file == "unicodeStringLiteral.ts") {
            continue;
        }

        //console.log("... file data: " + data);
        console.log("printing file TS ... read file: " + file);
        const output1 = printTree(fld + "/" + file);
        console.log("executing file C++ ... read file: " + file);
        const output2 = execSync("C:/dev/TypeScriptCompiler/__build/tsc/tsc-new-parser/Debug/tsc-new-scanner.exe " + fld + "/" + file);
        console.log("testing file ... file: " + file);

        const output1_str = output1.toString().split("\n");
        const output2_str = output2.toString().split("\n");

        for (let i = 0; i < output1_str.length; i++) {
            const o1 = output1_str[i].trim();
            const o2 = output2_str[i].trim();

            if (o1 != o2) {
                console.log("Output TS:", o1);
                console.log("Output c++ scanner:", o2);
                throw "File mismatched " + file;
            }
        }
    }
}
catch (err) {
    console.error(err);
}
