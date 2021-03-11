parser grammar TypeScriptParserANTLR;

options {
	tokenVocab = TypeScriptLexerANTLR;
}

@parser::members 
{
    bool channelTokenEquals(size_t tokenType)
    {
        auto nextToken = _input->getTokenSource()->nextToken();
        return nextToken && nextToken->getType() == tokenType;
    }
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
    | lexicalDeclaration
    ;

hoistableDeclaration
    : functionDeclaration
    ;    

lexicalDeclaration
    : (CONST_KEYWORD | LET_KEYWORD | VAR_KEYWORD) bindingList
    ;

bindingList
    : lexicalBinding (COMMA_TOKEN lexicalBinding)*
    ;

lexicalBinding
    : bindingIdentifier typeParameter? initializer?
    | bindingPattern initializer
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
    | BIGINT_KEYWORD 
    | VOID_KEYWORD ;    

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

statementTerminator
    : SEMICOLON_TOKEN
    | {channelTokenEquals(LineTerminatorSequence)}?
    ;

expressionStatement
    : expression statementTerminator ;

ifStatement
    : IF_KEYWORD OPENPAREN_TOKEN expression CLOSEPAREN_TOKEN statement (ELSE_KEYWORD statement)? ; // No need for statementTerminator as statement is teminated

returnStatement
    : RETURN_KEYWORD expression? statementTerminator ;

expression
    : assignmentExpression
    | assignmentExpression (COMMA_TOKEN assignmentExpression)* ;
    
assignmentExpression
    : conditionalExpression
    | leftHandSideExpression EQUALS_TOKEN assignmentExpression
    ;   

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
    | DELETE_KEYWORD unaryExpression
    | VOID_KEYWORD unaryExpression
    | TYPEOF_KEYWORD unaryExpression
    | PLUS_TOKEN unaryExpression
    | MINUS_TOKEN unaryExpression
    | TILDE_TOKEN unaryExpression
    | EXCLAMATION_TOKEN unaryExpression
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
    | NEW_KEYWORD newExpression 
    ;

callExpression
    : coverCallExpressionAndAsyncArrowHead
    | callExpression arguments 
    ;

coverCallExpressionAndAsyncArrowHead
    : memberExpression arguments 
    ;    

memberExpression    
    : primaryExpression
    | memberExpression DOT_TOKEN IdentifierName 
    ;

primaryExpression
    : THIS_KEYWORD
    | literal
    | identifierReference 
    | coverParenthesizedExpressionAndArrowParameterList 
    ;

coverParenthesizedExpressionAndArrowParameterList
    : OPENPAREN_TOKEN expression CLOSEPAREN_TOKEN
    | OPENPAREN_TOKEN expression COMMA_TOKEN CLOSEPAREN_TOKEN
    | OPENPAREN_TOKEN CLOSEPAREN_TOKEN
    | OPENPAREN_TOKEN DOTDOTDOT_TOKEN bindingIdentifier CLOSEPAREN_TOKEN
    | OPENPAREN_TOKEN DOTDOTDOT_TOKEN bindingPattern CLOSEPAREN_TOKEN
    | OPENPAREN_TOKEN expression COMMA_TOKEN DOTDOTDOT_TOKEN bindingIdentifier CLOSEPAREN_TOKEN
    | OPENPAREN_TOKEN expression COMMA_TOKEN DOTDOTDOT_TOKEN bindingPattern CLOSEPAREN_TOKEN
    ;

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
    : identifier
    | YIELD_KEYWORD
    | AWAIT_KEYWORD 
    ;

bindingPattern
    : objectBindingPattern
    | arrayBindingPattern
    ;

objectBindingPattern
    : OPENBRACE_TOKEN CLOSEBRACE_TOKEN
// TODO    
    ;    

arrayBindingPattern
    : OPENBRACKET_TOKEN CLOSEBRACKET_TOKEN
// TODO    
    ;    

identifier
    : IdentifierName ; // but not ReservedWord 

arguments
    : OPENPAREN_TOKEN CLOSEPAREN_TOKEN  
    | OPENPAREN_TOKEN argumentList CLOSEPAREN_TOKEN 
    | OPENPAREN_TOKEN argumentList COMMA_TOKEN CLOSEPAREN_TOKEN 
    ;

argumentList
    : argumentListItem (COMMA_TOKEN argumentListItem)*
    ;

argumentListItem
    : DOTDOTDOT_TOKEN? assignmentExpression ;
