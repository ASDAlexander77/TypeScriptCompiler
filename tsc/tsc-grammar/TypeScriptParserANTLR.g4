parser grammar TypeScriptParserANTLR;

options {
	tokenVocab = TypeScriptLexerANTLR;
}

@parser::postinclude {
#include "typescript/AST.h"
#include "typescript/Helper.h"
#include <map>
}

@parser::context {
#define NODE(x, ...) assign_new<x>(_localctx, __VA_ARGS__)
#define GET(x) get(_localctx->x())
#define GET_AS(x, y) getAsType<x>(_localctx->y())
#define MOVE_DOWN(x) move_down(_localctx, _localctx->x())
#define COLLECTION(x) collection(_localctx, _localctx->x())

template<typename V>
class ParseTreeAssoc {
public:
    V get(antlr4::tree::ParseTree *node) {
        return _annotations[node];
    }

    void put(antlr4::tree::ParseTree *node, V value) {
        _annotations[node] = value;
    }

    V removeFrom(antlr4::tree::ParseTree *node) {
        auto value = _annotations[node];
        _annotations.erase(node);
        return value;
    }

protected:
    std::map<antlr4::tree::ParseTree *, V> _annotations;
};

}

@parser::members {

/* public parser declarations/members section */
ModuleAST m;
ModuleAST &getModuleAST() 
{ 
    //auto moduleASTPtr = getAsType<ModuleAST>(main());
    //return *moduleASTPtr.get(); 
    return m;
}

ParseTreeAssoc<std::shared_ptr<NodeAST>> assoc;

template <typename NodeTy, typename... Args>
void assign_new(antlr4::tree::ParseTree *tree, Args &&... args) 
{ 
    const antlr4::misc::Interval &loc = tree->getSourceInterval();
    assoc.put(
        tree, 
        std::make_shared<NodeTy>(
            TextRange({static_cast<int>(loc.a), static_cast<int>(loc.b)}), 
            std::forward<Args>(args)...)); 
};

template <typename NodeTy>
typename NodeTy::TypePtr getAsType(antlr4::tree::ParseTree *tree) 
{ 
    return std::dynamic_pointer_cast<NodeTy>(assoc.get(tree));
};

std::shared_ptr<NodeAST> get(antlr4::tree::ParseTree *tree) 
{ 
    return assoc.get(tree); 
};

void move_down(antlr4::tree::ParseTree *to, antlr4::tree::ParseTree *from) 
{ 
    assoc.put(to, assoc.get(from)); 
};

template <typename CtxTy>
std::vector<std::shared_ptr<NodeAST>> collection(antlr4::tree::ParseTree *tree, std::vector<CtxTy> items)
{
    std::vector<std::shared_ptr<NodeAST>> nodes;
    for (auto &item : items)
    {
        nodes.push_back(get(item));
    }

    return nodes;
};

} // @parser::members

// Actual grammar start.
main
    : moduleBody EOF { NODE(ModuleAST, GET_AS(ModuleBlockAST, moduleBody)); } ;

moduleBody 
    : moduleItem* { /*NODE(ModuleBlockAST, COLLECTION(moduleItem));*/ } ;

moduleItem
    : statementListItem { MOVE_DOWN(statementListItem); }
    ;

statementListItem 
    : statement      { MOVE_DOWN(statement); }
    | declaration    { MOVE_DOWN(declaration); }
    ;

declaration 
    : hoistableDeclaration { MOVE_DOWN(hoistableDeclaration); }
    ;

hoistableDeclaration
    : functionDeclaration { MOVE_DOWN(functionDeclaration); }
    ;    

functionDeclaration
    : FUNCTION_KEYWORD bindingIdentifier? OPENPAREN_TOKEN formalParameters? CLOSEPAREN_TOKEN typeParameter? OPENBRACE_TOKEN functionBody CLOSEBRACE_TOKEN 
        { NODE(FunctionDeclarationAST, GET(bindingIdentifier)); } ;

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

