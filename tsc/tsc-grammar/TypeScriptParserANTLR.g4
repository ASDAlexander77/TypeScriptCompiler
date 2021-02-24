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
    | ifStatement
    | returnStatement
    | block
    ;    

block
    : OPENBRACE_TOKEN statementList CLOSEBRACE_TOKEN ;

emptyStatement
    : SEMICOLON_TOKEN ;

expressionStatement
    : expression SEMICOLON_TOKEN? ;

ifStatement
    : IF_KEYWORD OPENPAREN_TOKEN expression CLOSEPAREN_TOKEN statement (ELSE_KEYWORD statement)? SEMICOLON_TOKEN? ;    

returnStatement
    : RETURN_KEYWORD expression? SEMICOLON_TOKEN? ;

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
    | equalityExpression EQUALSEQUALS_TOKEN relationalExpression
    | equalityExpression EXCLAMATIONEQUALS_TOKEN relationalExpression
    | equalityExpression EQUALSEQUALSEQUALS_TOKEN relationalExpression
    | equalityExpression EXCLAMATIONEQUALSEQUALS_TOKEN relationalExpression
    ;    

relationalExpression
    : shiftExpression ;

shiftExpression
    : additiveExpression
    ;    

additiveExpression
    : multiplicativeExpression
    | additiveExpression PLUS_TOKEN multiplicativeExpression
    | additiveExpression MINUS_TOKEN multiplicativeExpression
    ;    

multiplicativeExpression
    : exponentiationExpression
    | multiplicativeExpression ASTERISK_TOKEN exponentiationExpression
    | multiplicativeExpression SLASH_TOKEN exponentiationExpression
    | multiplicativeExpression PERCENT_TOKEN exponentiationExpression
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

undefinedLiteral
    : UNDEFINED_KEYWORD ;

booleanLiteral
    : TRUE_KEYWORD
    | FALSE_KEYWORD ;

literal
    : nullLiteral
    | undefinedLiteral
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

