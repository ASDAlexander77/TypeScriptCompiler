parser grammar TypeScriptParserANTLR;

options {
	tokenVocab = TypeScriptLexerANTLR;
}

@parser::postinclude {
#include "typescript/AST.h"
}

@parser::context {
#define PUSH(x, ...) push_node<x>(_localctx, __VA_ARGS__)
}

@parser::members {

/* public parser declarations/members section */
std::unique_ptr<ModuleAST> moduleAST;
const std::unique_ptr<ModuleAST> &getModuleAST() { return moduleAST; }

std::stack<std::unique_ptr<NodeAST>> stack;

template <typename NodeTy, typename... Args>
std::unique_ptr<NodeTy> make_node(antlr4::tree::ParseTree *tree, Args &&... args) 
{
}

template <typename NodeTy, typename... Args>
void push_node(antlr4::tree::ParseTree *tree, Args &&... args) 
{ 
    const antlr4::misc::Interval &loc = tree->getSourceInterval();
    stack.push(
        std::make_unique<NodeTy>(
            TextRange({static_cast<int>(loc.a), static_cast<int>(loc.b)}), 
            std::forward<Args>(args)...)); 
};

} // @parser::members

// Actual grammar start.
main
    : module EOF { PUSH(ModuleAST); } ;

module
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
    : FUNCTION_KEYWORD bindingIdentifier? OPENPAREN_TOKEN formalParameters? CLOSEPAREN_TOKEN typeParameter? OPENBRACE_TOKEN functionBody CLOSEBRACE_TOKEN 
        { PUSH(FunctionDeclarationAST); } ;

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
    : assignmentExpression (COMMA_TOKEN assignmentExpression)* ;
    
assignmentExpression
    : conditionalExpression ;   

conditionalExpression
    : shortCircuitExpression (QUESTION_TOKEN assignmentExpression COLON_TOKEN assignmentExpression)? ;

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
    : memberExpression arguments
    | callExpression arguments ;

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
    : IdentifierName ;

bindingIdentifier    
    : identifier ;

identifier
    : IdentifierName ; // but not ReservedWord 

arguments
    :  OPENPAREN_TOKEN (expression (COMMA_TOKEN expression)*)? CLOSEPAREN_TOKEN ;

