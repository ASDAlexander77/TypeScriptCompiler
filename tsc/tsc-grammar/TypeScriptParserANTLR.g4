parser grammar TypeScriptParserANTLR;

options {
	tokenVocab = TypeScriptLexerANTLR;
}

// Actual grammar start.
main
    : moduleBody EOF ;

moduleBody 
    : moduleItem* ;

moduleItem
    : statementListItem
    ;

statementListItem 
    : statement
    | declaration
    ;

declaration 
    : hoistableDeclaration
    ;

hoistableDeclaration
    : functionDeclaration
    ;    

functionDeclaration
    : FUNCTION_KEYWORD bindingIdentifier? OPENPAREN_TOKEN formalParameters? CLOSEPAREN_TOKEN typeParameter? OPENBRACE_TOKEN functionBody CLOSEBRACE_TOKEN ;

formalParameters
    : functionRestParameter 
    | formalParameter (COMMA_TOKEN formalParameter)* (COMMA_TOKEN functionRestParameter)? ;    

formalParameter
    : IdentifierName QUESTION_TOKEN? typeParameter? initializer? ;    

typeParameter
    : COLON_TOKEN typeDeclaration ;    

initializer
    : EQUALS_TOKEN assignmentExpression ;  

typeDeclaration
    : ANY_KEYWORD
    | NUMBER_KEYWORD
    | BOOLEAN_KEYWORD
    | STRING_KEYWORD
    | BIGINT_KEYWORD ;    

functionRestParameter
    : DOTDOTDOT_TOKEN formalParameter ;

functionBody
    : functionStatementList ;    

functionStatementList
    : statementList ;

statementList
    : statementListItem* ;    

statement
    : emptyStatement
    | expressionStatement
    | returnStatement
    ;    

emptyStatement
    : SEMICOLON_TOKEN ;

expressionStatement
    : expression SEMICOLON_TOKEN ;

returnStatement
    : RETURN_KEYWORD expression? SEMICOLON_TOKEN ;

expression
    : assignmentExpression
    | assignmentExpression (COMMA_TOKEN assignmentExpression)* ;
    
assignmentExpression
    : conditionalExpression ;   

conditionalExpression
    : shortCircuitExpression
    | shortCircuitExpression QUESTION_TOKEN assignmentExpression COLON_TOKEN assignmentExpression ;

shortCircuitExpression    
    : logicalORExpression ;

logicalORExpression
    : logicalANDExpression ;

logicalANDExpression
    : bitwiseORExpression ;

bitwiseORExpression
    : bitwiseXORExpression ;    

bitwiseXORExpression
    : bitwiseANDExpression ;    

bitwiseANDExpression
    : equalityExpression ;

equalityExpression
    : relationalExpression 
    | equalityExpression EQUALSEQUALS_TOKEN equalityExpression
    ;    

relationalExpression
    : shiftExpression ;

shiftExpression
    : additiveExpression
    ;    

additiveExpression
    : multiplicativeExpression
    ;    

multiplicativeExpression
    : exponentiationExpression
    ;  

exponentiationExpression
    : unaryExpression
    ;      

unaryExpression
    : updateExpression
    ;

updateExpression
    : leftHandSideExpression
    ;

leftHandSideExpression    
    : newExpression
    | callExpression
    | optionalExpression
    ;

newExpression
    : memberExpression
    | NEW_KEYWORD newExpression ;

callExpression
    : coverCallExpressionAndAsyncArrowHead
    | callExpression arguments ;

coverCallExpressionAndAsyncArrowHead
    : memberExpression arguments ;    

memberExpression    
    : primaryExpression
    | memberExpression DOT_TOKEN IdentifierName ;

primaryExpression
    : literal
    | identifierReference ;

optionalExpression
    : memberExpression optionalChain
    ;    

optionalChain
    : QUESTIONDOT_TOKEN IdentifierName
    ;    

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
    : identifier ;

bindingIdentifier    
    : identifier ;

identifier
    : IdentifierName ; // but not ReservedWord 

arguments
    : OPENPAREN_TOKEN (expression (COMMA_TOKEN expression)*)? CLOSEPAREN_TOKEN ;

