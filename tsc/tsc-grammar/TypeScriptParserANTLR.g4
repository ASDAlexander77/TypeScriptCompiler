parser grammar TypeScriptParserANTLR;

options {
	tokenVocab = TypeScriptLexerANTLR;
}

// Actual grammar start.
main
    : statement* EOF ;

statement
    : expressionStatement ;

expressionStatement
    : expression SEMICOLON_TOKEN ;

expression
    : primaryExpression
    | leftHandSideExpression ;

nullLiteral
    : NULL_KEYWORD ;

booleanLiteral
    : TRUE_KEYWORD
    | FALSE_KEYWORD ;

literal
    : nullLiteral
    | booleanLiteral
    | numericLiteral
    | StringLiteral ;

numericLiteral 
    : DecimalLiteral
    | DecimalBigIntegerLiteral
    | BinaryBigIntegerLiteral
    | OctalBigIntegerLiteral
    | HexBigIntegerLiteral ;    

identifierReference
    : IdentifierName ;

leftHandSideExpression     
    : callExpression ;

arguments
    :  OPENPAREN_TOKEN expression (COMMA_TOKEN expression)* CLOSEPAREN_TOKEN ;

callExpression
    : memberExpression arguments
    | callExpression arguments
    ;

memberExpression    
    : primaryExpression
    | memberExpression DOT_TOKEN IdentifierName ;

primaryExpression
    : literal
    | identifierReference ;

