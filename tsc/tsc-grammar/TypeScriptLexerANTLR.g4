lexer grammar TypeScriptLexerANTLR;

tokens 
{
    // extra tokens
}

channels 
{
    WHITESPACE_CHANNEL,
    COMMENTS_CHANNEL
}

/** Keywords */
ABSTRACT_KEYWORD: 'abstract' ;
ANY_KEYWORD: 'any' ;
AS_KEYWORD: 'as' ;
ASSERTS_KEYWORD: 'asserts' ;
BIGINT_KEYWORD: 'bigint' ;
BOOLEAN_KEYWORD: 'boolean' ;
BREAK_KEYWORD: 'break' ;
CASE_KEYWORD: 'case' ;
CATCH_KEYWORD: 'catch' ;
CLASS_KEYWORD: 'class' ;
CONTINUE_KEYWORD: 'continue' ;
CONST_KEYWORD: 'const' ;
DEBUGGER_KEYWORD: 'debugger' ;
DECLARE_KEYWORD: 'declare' ;
DEFAULT_KEYWORD: 'default' ;
DELETE_KEYWORD: 'delete' ;
DO_KEYWORD: 'do' ;
ELSE_KEYWORD: 'else' ;
ENUM_KEYWORD: 'enum' ;
EXPORT_KEYWORD: 'export' ;
EXTENDS_KEYWORD: 'extends' ;
FALSE_KEYWORD: 'false' ;
FINALLY_KEYWORD: 'finally' ;
FOR_KEYWORD: 'for' ;
FROM_KEYWORD: 'from' ;
FUNCTION_KEYWORD: 'function' ;
GET_KEYWORD: 'get' ;
IF_KEYWORD: 'if' ;
IMPLEMENTS_KEYWORD: 'implements' ;
IMPORT_KEYWORD: 'import' ;
IN_KEYWORD: 'in' ;
INFER_KEYWORD: 'infer' ;
INSTANCEOF_KEYWORD: 'instanceof' ;
INTERFACE_KEYWORD: 'interface' ;
INTRINSIC_KEYWORD: 'intrinsic' ;
IS_KEYWORD: 'is' ;
KEYOF_KEYWORD: 'keyof' ;
LET_KEYWORD: 'let' ;
MODULE_KEYWORD: 'module' ;
NAMESPACE_KEYWORD: 'namespace' ;
NEVER_KEYWORD: 'never' ;
NEW_KEYWORD: 'new' ;
NULL_KEYWORD: 'null' ;
NUMBER_KEYWORD: 'number' ;
OBJECT_KEYWORD: 'object' ;
PACKAGE_KEYWORD: 'package' ;
PRIVATE_KEYWORD: 'private' ;
PROTECTED_KEYWORD: 'protected' ;
PUBLIC_KEYWORD: 'public' ;
READONLY_KEYWORD: 'readonly' ;
REQUIRE_KEYWORD: 'require' ;
GLOBAL_KEYWORD: 'global' ;
RETURN_KEYWORD: 'return' ;
SET_KEYWORD: 'set' ;
STATIC_KEYWORD: 'static' ;
STRING_KEYWORD: 'string' ;
SUPER_KEYWORD: 'super' ;
SWITCH_KEYWORD: 'switch' ;
SYMBOL_KEYWORD: 'symbol' ;
THIS_KEYWORD: 'this' ;
THROW_KEYWORD: 'throw' ;
TRUE_KEYWORD: 'true' ;
TRY_KEYWORD: 'try' ;
TYPE_KEYWORD: 'type' ;
TYPEOF_KEYWORD: 'typeof' ;
UNDEFINED_KEYWORD: 'undefined' ;
UNIQUE_KEYWORD: 'unique' ;
UNKNOWN_KEYWORD: 'unknown' ;
VAR_KEYWORD: 'var' ;
VOID_KEYWORD: 'void' ;
WHILE_KEYWORD: 'while' ;
WITH_KEYWORD: 'with' ;
YIELD_KEYWORD: 'yield' ;
ASYNC_KEYWORD: 'async' ;
AWAIT_KEYWORD: 'await' ;
OF_KEYWORD: 'of' ;

/** punctuators */
OPENBRACE_TOKEN: '{' ;
CLOSEBRACE_TOKEN: '}' ;
OPENPAREN_TOKEN: '(' ;
CLOSEPAREN_TOKEN: ')' ;
OPENBRACKET_TOKEN: '[' ;
CLOSEBRACKET_TOKEN: ']' ;
DOT_TOKEN: '.' ;
DOTDOTDOT_TOKEN: '...' ;
SEMICOLON_TOKEN: ';' ;
COMMA_TOKEN: ',' ;
LESSTHAN_TOKEN: '<' ;
GREATERTHAN_TOKEN: '>' ;
LESSTHANEQUALS_TOKEN: '<=' ;
GREATERTHANEQUALS_TOKEN: '>=' ;
EQUALSEQUALS_TOKEN: '==' ;
EXCLAMATIONEQUALS_TOKEN: '!=' ;
EQUALSEQUALSEQUALS_TOKEN: '===' ;
EXCLAMATIONEQUALSEQUALS_TOKEN: '!==' ;
EQUALSGREATERTHAN_TOKEN: '=>' ;
PLUS_TOKEN: '+' ;
MINUS_TOKEN: '-' ;
ASTERISKASTERISK_TOKEN: '**' ;
ASTERISK_TOKEN: '*' ;
SLASH_TOKEN: '/' ;
PERCENT_TOKEN: '%' ;
PLUSPLUS_TOKEN: '++' ;
MINUSMINUS_TOKEN: '--' ;
LESSTHANLESSTHAN_TOKEN: '<<' ;
LESSTHANSLASH_TOKEN: '</' ;
GREATERTHANGREATERTHAN_TOKEN: '>>' ;
GREATERTHANGREATERTHANGREATERTHAN_TOKEN: '>>>' ;
AMPERSAND_TOKEN: '&' ;
BAR_TOKEN: '|' ;
CARET_TOKEN: '^' ;
EXCLAMATION_TOKEN: '!' ;
TILDE_TOKEN: '~' ;
AMPERSANDAMPERSAND_TOKEN: '&&' ;
BARBAR_TOKEN: '||' ;
QUESTION_TOKEN: '?' ;
QUESTIONQUESTION_TOKEN: '??' ;
QUESTIONDOT_TOKEN: '?.' ;
COLON_TOKEN: ':' ;
EQUALS_TOKEN: '=' ;
PLUSEQUALS_TOKEN: '+=' ;
MINUSEQUALS_TOKEN: '-=' ;
ASTERISKEQUALS_TOKEN: '*=' ;
ASTERISKASTERISKEQUALS_TOKEN: '**=' ;
SLASHEQUALS_TOKEN: '/=' ;
PERCENTEQUALS_TOKEN: '%=' ;
LESSTHANLESSTHANEQUALS_TOKEN: '<<=' ;
GREATERTHANGREATERTHANEQUALS_TOKEN: '>>=' ;
GREATERTHANGREATERTHANGREATERTHANEQUALS_TOKEN: '>>>=' ;
AMPERSANDEQUALS_TOKEN: '&=' ;
BAREQUALS_TOKEN: '|=' ;
CARETEQUALS_TOKEN: '^=' ;
BARBAREQUALS_TOKEN: '||=' ;
AMPERSANDAMPERSANDEQUALS_TOKEN: '&&=' ;
QUESTIONQUESTIONEQUALS_TOKEN: '??=' ;
AT_TOKEN: '@' ;
BACKTICK_TOKEN: '`' ;

/** Rules */
/** White space */
/**
TAB : [\t] ;
VT : [\u000B] ;
FF : [\f] ;
SP : [ ] ;
NBSP : [\u00A0] ;
ZWNBSP : [\uFEFF] ;
USP : [\p{Space_Separator}] ;
 */

WhiteSpace
    : [\t\u000B\f \u00A0\uFEFF\p{Space_Separator}] -> channel(WHITESPACE_CHANNEL) ;

/** Line Terminators */
/**
LF : [\n] ;
CR : [\r] ;
LS : [\u2028] ;
PS : [\u2029] ;
*/

fragment LineTerminator
    : [\r\n\u2028\u2029] ;

fragment NotLineTerminator
    : ~[\r\n\u2028\u2029] ;

LineTerminatorSequence
    : ('\r\n' | LineTerminator) ;

/** Comment */
MultiLineComment
    : '/*' .*? '*/' -> channel(COMMENTS_CHANNEL) ; 

SingleLineComment
    : '//' NotLineTerminator* -> channel(COMMENTS_CHANNEL) ;

IdentifierName
    : IdentifierStart IdentifierPart* ;

fragment IdentifierStart 
    : [\p{ID_Start}$_]
    | UnicodeEscapeSequence ;

/**
ZWNJ : [\u200C] ;
ZWJ : [\uu200D] ;
*/ 
fragment IdentifierPart 
    : [\p{ID_Continue}$\u200C\u200D]
    | UnicodeEscapeSequence
    ;

fragment Punctuator
    : OPENBRACE_TOKEN | CLOSEBRACE_TOKEN | OPENPAREN_TOKEN | CLOSEPAREN_TOKEN | OPENBRACKET_TOKEN | CLOSEBRACKET_TOKEN | DOT_TOKEN | DOTDOTDOT_TOKEN
    | SEMICOLON_TOKEN | COMMA_TOKEN | LESSTHAN_TOKEN | GREATERTHAN_TOKEN | LESSTHANEQUALS_TOKEN | GREATERTHANEQUALS_TOKEN | EQUALSEQUALS_TOKEN | EXCLAMATIONEQUALS_TOKEN
    | EQUALSEQUALSEQUALS_TOKEN | EXCLAMATIONEQUALSEQUALS_TOKEN | EQUALSGREATERTHAN_TOKEN | PLUS_TOKEN | MINUS_TOKEN | ASTERISKASTERISK_TOKEN | ASTERISK_TOKEN
    | SLASH_TOKEN | PERCENT_TOKEN | PLUSPLUS_TOKEN | MINUSMINUS_TOKEN | LESSTHANLESSTHAN_TOKEN | LESSTHANSLASH_TOKEN | GREATERTHANGREATERTHAN_TOKEN | GREATERTHANGREATERTHANGREATERTHAN_TOKEN
    | AMPERSAND_TOKEN | BAR_TOKEN | CARET_TOKEN | EXCLAMATION_TOKEN | TILDE_TOKEN | AMPERSANDAMPERSAND_TOKEN | BARBAR_TOKEN | QUESTION_TOKEN | QUESTIONQUESTION_TOKEN | QUESTIONDOT_TOKEN
    | COLON_TOKEN | EQUALS_TOKEN | PLUSEQUALS_TOKEN | MINUSEQUALS_TOKEN | ASTERISKEQUALS_TOKEN | ASTERISKASTERISKEQUALS_TOKEN | SLASHEQUALS_TOKEN
    | PERCENTEQUALS_TOKEN | LESSTHANLESSTHANEQUALS_TOKEN | GREATERTHANGREATERTHANEQUALS_TOKEN | GREATERTHANGREATERTHANGREATERTHANEQUALS_TOKEN | AMPERSANDEQUALS_TOKEN
    | BAREQUALS_TOKEN | CARETEQUALS_TOKEN | BARBAREQUALS_TOKEN | AMPERSANDAMPERSANDEQUALS_TOKEN | QUESTIONQUESTIONEQUALS_TOKEN | AT_TOKEN | BACKTICK_TOKEN ;    

BinaryIntegerLiteral
    : '0' [bB] BinaryDigits ;

OctalIntegerLiteral
    : '0' [oO]? OctalDigits ;

HexIntegerLiteral
    : '0' [xX] HexDigits ;

DecimalIntegerLiteral 
    : DecimalDigits ;

DecimalLiteral
    : DecimalIntegerLiteral? '.' DecimalDigits ExponentPart?
    | DecimalIntegerLiteral ExponentPart ;

DecimalBigIntegerLiteral 
    : '0' BigIntLiteralSuffix
    | NonZeroDigit NumericLiteralSeparator? DecimalDigits? BigIntLiteralSuffix ;

fragment NumericLiteralSeparator
    : '_' ;

BinaryBigIntegerLiteral
    : BinaryIntegerLiteral BigIntLiteralSuffix ;

OctalBigIntegerLiteral
    : OctalIntegerLiteral BigIntLiteralSuffix ;

HexBigIntegerLiteral
    : HexIntegerLiteral BigIntLiteralSuffix ;

fragment BigIntLiteralSuffix 
    : 'n' ;

fragment DecimalDigits
    : DecimalDigit (NumericLiteralSeparator DecimalDigit | DecimalDigit)* ;

fragment DecimalDigit 
    : [0-9] ; 

fragment NonZeroDigit
    : [1-9] ;

fragment ExponentPart
    : [eE] SignedInteger ;

fragment SignedInteger
    : [+-]? DecimalDigits ;

fragment BinaryDigits
    : BinaryDigit (NumericLiteralSeparator BinaryDigit | BinaryDigit)* ;

fragment BinaryDigit 
    : [01] ;    

fragment OctalDigits
    : OctalDigit (NumericLiteralSeparator OctalDigit | OctalDigit)* ;

fragment OctalDigit
    : [0-7] ;

fragment HexDigits
    : HexDigit (NumericLiteralSeparator HexDigit | HexDigit)* ;    

fragment HexDigit
    : [0-9a-fA-F] ;    

/**
LS : [\u2028] ;
PS : [\u2029] ;
*/ 
StringLiteral
    : '"' (DoubleStringCharacter*) '"'
    | '\'' (SingleStringCharacter*) '\''
    | NoSubstitutionTemplate ;

fragment DoubleStringCharacter 
    : ~["\\\n\r\u2028\u2029]
    | [\u2028\u2029]
    | '\\' EscapeSequence
    | LineContinuation ;

fragment SingleStringCharacter 
    : ~['\\\n\r\u2028\u2029]
    | [\u2028\u2029]
    | '\\' EscapeSequence
    | LineContinuation ;    

fragment LineContinuation
    : '\\' LineTerminatorSequence ;
    
fragment EscapeSequence
    : CharacterEscapeSequence
    | '0'
    | HexEscapeSequence
    | UnicodeEscapeSequence ;

fragment CharacterEscapeSequence
    :  ~[\u2028\u2029] ;

HexEscapeSequence
    : 'x' HexDigit HexDigit ;

fragment UnicodeEscapeSequence
    : '\\u' (Hex4Digits | '{' CodePoint '}' ) ;

fragment Hex4Digits
    : HexDigit HexDigit HexDigit HexDigit ;        

fragment CodePoint 
    : HexDigits ;

RegularExpressionLiteral 
    : '/' RegularExpressionBody '/' RegularExpressionFlags ;

fragment RegularExpressionBody
    : RegularExpressionFirstChar RegularExpressionChar* ;

fragment RegularExpressionFirstChar 
    : ~[*\r\n\u2028\u2029\\/[] // LineTerminator and not one of * or \ or / or [
    | '\\' NotLineTerminator ;

fragment RegularExpressionChar 
    : ~[\r\n\u2028\u2029\\/[] // LineTerminator and not one of \ or / or [
    | '\\' NotLineTerminator ;    

fragment RegularExpressionFlags
    : IdentifierPart* ;

/*
fragment Template
    : NoSubstitutionTemplate
    | TemplateHead ;
*/    

fragment NoSubstitutionTemplate
    : '`' TemplateCharacter* '`' ;

TemplateHead
    : '`' TemplateCharacter* '${' ;

/*
fragment TemplateSubstitutionTail
    : TemplateMiddle
    | TemplateTail ;
 */

TemplateMiddle
    : '}' TemplateCharacter* '${' ;

TemplateTail
    : '}' TemplateCharacter* '`' ;

fragment TemplateCharacter
    : '$' {_input->LA(1) != '{'}?
    | '\\' EscapeSequence
    | '\\' NotEscapeSequence
    | LineContinuation
    | LineTerminatorSequence
    | ~[`\\$\r\n\u2028\u2029] ; // but not one of ` or \ or $ or LineTerminator

fragment NotEscapeSequence
    : '0' DecimalDigit
    | NonZeroDigit
    | 'x' HexDigit?
    | 'u' HexDigit? HexDigit? HexDigit?
    | 'u' '{' CodePoint? ;
    