parser grammar TypeScriptParserANTLR;

options {
	tokenVocab = TypeScriptLexerANTLR;
}

// Actual grammar start.
main
    : declaration* EOF ;

declaration
    : functionDeclaration ;    

functionDeclaration
    : FUNCTION_KEYWORD IdentifierName? OPENPAREN_TOKEN formalParameters? CLOSEPAREN_TOKEN typeParameter? OPENBRACE_TOKEN functionBody CLOSEBRACE_TOKEN ;

formalParameters
    : functionRestParameter
    | formalParameter (COMMA_TOKEN formalParameter)* (COMMA_TOKEN functionRestParameter)? ;    

formalParameter
    : IdentifierName QUESTION_TOKEN? typeParameter? initializer? ;    

typeParameter
    : COLON_TOKEN typeDeclaration ;    

initializer
    : EQUALS_TOKEN assignmentExpression ;  

assignmentExpression
    : leftHandSideExpression ;   

typeDeclaration
    : ANY_KEYWORD 
    | NUMBER_KEYWORD
    | BOOLEAN_KEYWORD 
    | STRING_KEYWORD
    | BIGINT_KEYWORD ;    

functionRestParameter
    : DOTDOTDOT_TOKEN formalParameter ;

functionBody
    : statementListItem* ;    

statementListItem
    : statement 
    | declaration ;

statement
    : (expression 
       | returnStatement) SEMICOLON_TOKEN ;

returnStatement
    : RETURN_KEYWORD expression? ;

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
    | DecimalIntegerLiteral
    | DecimalBigIntegerLiteral
    | BinaryBigIntegerLiteral
    | OctalBigIntegerLiteral
    | HexBigIntegerLiteral ;    

identifierReference
    : IdentifierName ;

leftHandSideExpression
    : memberExpression     
    | callExpression ;

arguments
    :  OPENPAREN_TOKEN (expression (COMMA_TOKEN expression)*)? CLOSEPAREN_TOKEN ;

callExpression
    : memberExpression arguments
    | callExpression arguments ;

memberExpression    
    : primaryExpression
    | memberExpression DOT_TOKEN IdentifierName ;

primaryExpression
    : literal
    | identifierReference ;
