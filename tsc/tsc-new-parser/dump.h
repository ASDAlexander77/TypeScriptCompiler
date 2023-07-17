#ifndef DUMP_H
#define DUMP_H

#include <functional>
#include <iostream>
#include <regex>
#include <stack>
#include <string>

#include "core.h"
#include "enums.h"
#include "node_test.h"
#include "parser.h"
#include "scanner.h"
#include "scanner_enums.h"
#include "types.h"

namespace ts
{

template <typename OUT> class Printer
{
    OUT &out;
    int ident;
    bool declarationMode;

    // temp var
    bool isLastStatementBlock;

  public:
    Printer(OUT &out) : out(out), ident(0), declarationMode(false), isLastStatementBlock(false)
    {
    }

    void setDeclarationMode(bool declarationMode_)
    {
        declarationMode = declarationMode_;
    }

    void printNode(ts::Node node)
    {
        forEachChildPrint(node);
    }

    template <typename T>
    void printNodes(NodeArray<T> nodes, const char *open = nullptr, const char *separator = nullptr,
                              const char *end = nullptr, bool ifAny = false)
    {
        forEachChildrenPrint(nodes, open, separator, end, ifAny);
    }

  protected:

    void printText(const char* text)
    {
        if (text)
        {
            std::string s(text);
            auto end = s.length() - 1;
            if (s.at(end) == '\n')
            {
                out << s.substr(0, end).c_str();
                newLine();
            }
            else
            {
                out << text;
            }
        }
    }

    void newLine()
    {
        out << std::endl;
    }    

    void  newLineWithIntent()
    {
        newLine();
        printIntent();
    }

    void printIntent()
    {
        for (auto i = 0; i < ident; i++)
        {
            out << "\t";
        }
    }

    void incIndent()
    {
        ident++;
    }

    void decIndent()
    {
        ident--;
    }

    inline bool isBlock(ts::Node node)
    {
        return node == SyntaxKind::Block 
            || node == SyntaxKind::ModuleBlock;
    }    

    inline bool isTypeWithParams(ts::TypeNode typeNode)
    {
        if (typeNode == SyntaxKind::FunctionType)
        {
            auto signatureBase = typeNode.as<SignatureDeclarationBase>();
            if (!signatureBase->typeParameters.empty())
            {
                return true;
            }
        }

        return false;
    }

    inline bool isBlockOrStatementWithBlock(ts::Node node)
    {
        if (node == SyntaxKind::IfStatement)
        {
            auto ifStat = node.as<IfStatement>();
            return ifStat->elseStatement && isBlock(ifStat->elseStatement) || isBlock(ifStat->thenStatement);
        }

        if (node == SyntaxKind::WhileStatement)
        {
            auto whileStatement = node.as<WhileStatement>();
            return isBlock(whileStatement->statement);
        }

        if (node == SyntaxKind::ForStatement)
        {
            auto forStatement = node.as<ForStatement>();
            return isBlock(forStatement->statement);
        }

        if (node == SyntaxKind::ForInStatement)
        {
            auto forInStatement = node.as<ForInStatement>();
            return isBlock(forInStatement->statement);
        }        

        if (node == SyntaxKind::ForOfStatement)
        {
            auto forOfStatement = node.as<ForOfStatement>();
            return isBlock(forOfStatement->statement);
        }        

        if (node == SyntaxKind::WithStatement)
        {
            auto withStatement = node.as<WithStatement>();
            return isBlock(withStatement->statement);
        }    

        if (node == SyntaxKind::TryStatement)
        {
            auto tryStatement = node.as<TryStatement>();
            if (tryStatement->finallyBlock)
            {
                return isBlock(tryStatement->finallyBlock);
            }

            auto catchClause = tryStatement->catchClause.as<CatchClause>();
            return isBlock(catchClause->block);
        }        

        return node == SyntaxKind::Block 
            || node == SyntaxKind::ModuleBlock 
            || node == SyntaxKind::FunctionDeclaration 
            || node == SyntaxKind::MethodDeclaration
            || node == SyntaxKind::GetAccessor
            || node == SyntaxKind::SetAccessor
            || node == SyntaxKind::Constructor
            || node == SyntaxKind::ClassDeclaration
            || node == SyntaxKind::EnumDeclaration
            || node == SyntaxKind::ModuleDeclaration
            || node == SyntaxKind::InterfaceDeclaration
            || node == SyntaxKind::SwitchStatement;
    }

    void printDecorators(ts::Node node)
    {
        forEachChildrenPrint(node->decorators);
    }

    void printModifiersWithMode(ts::Node node)
    {
        if (declarationMode)
        {
            forEachChildrenPrintFilterWithAppend(node->modifiers, SyntaxKind::ExportKeyword, SyntaxKind::DeclareKeyword, nullptr, " ");
        }
        else
        {
            forEachChildrenPrint(node->modifiers, nullptr, nullptr, nullptr, false, " ");
        }
    }

    void printModifiers(ts::Node node)
    {
        forEachChildrenPrint(node->modifiers, nullptr, nullptr, nullptr, false, " ");
    }

    template <typename T>
    bool printStatementsLike(NodeArray<T> nodes, const char* separator = ";")
    {
        isLastStatementBlock = false;
        for (auto node : nodes)
        {
            isLastStatementBlock = false;

            printIntent();
            forEachChildPrint(node);
            if (!isBlockOrStatementWithBlock(node))
            {
                out << separator;
                newLine();
            }
            else
            {
                isLastStatementBlock = true;
            }
        }

        return isLastStatementBlock;
    }

    bool printStatements(NodeArray<ts::Statement> nodes)
    {
        return printStatementsLike(nodes);
    }

    void printBlockBase(std::function<void(void)> bodyFunc, SyntaxKind parent = SyntaxKind::Unknown)
    {
        newLineWithIntent();
        out << "{";
        incIndent();
        newLine();
        
        bodyFunc();
        
        decIndent();
        printIntent();
        out << "}";
        if (parent != SyntaxKind::ArrowFunction && parent != SyntaxKind::FunctionExpression && parent != SyntaxKind::ClassExpression)
            newLine();           
    }

    void printBlock(ts::Block block)
    {
        printBlockBase(
            [&]() { 
                printStatements(block->statements);
            },
            block->parent
        );
    }

    void printClauses(NodeArray<ts::CaseOrDefaultClause> nodes)
    {
        for (auto node : nodes)
        {
            printIntent();
            forEachChildPrint(node);
        }
    }    

    void printCaseBlock(ts::CaseBlock caseBlock)
    {
        printBlockBase(
            [&]() { 
                printClauses(caseBlock->clauses);
            }
        );
    }

    void printMembersBlock(ts::InterfaceDeclaration interfaceDeclaration)
    {
        printBlockBase(
            [&]() { 
                printStatementsLike(interfaceDeclaration->members);
            }
        );
    }

    void printMembersBlock(ts::ClassLikeDeclaration classDeclaration, SyntaxKind parent = SyntaxKind::Unknown)
    {
        printBlockBase(
            [&]() { 
                printStatementsLike(classDeclaration->members);
            },
            parent
        );
    }

    void printMembersBlock(ts::EnumDeclaration enumDeclaration)
    {
        printBlockBase(
            [&]() { 
                printStatementsLike(enumDeclaration->members, ",");
            }
        );
    }    

    template <typename T>
    void forEachChildrenPrint(NodeArray<T> nodes, const char *open = nullptr, const char *separator = nullptr,
                              const char *end = nullptr, bool ifAny = false, const char *afterChild = nullptr)
    {
        if (!ifAny && open)
        {
            printText(open);
        }

        auto hasAny = false;
        for (auto node : nodes)
        {
            if (!hasAny && ifAny && open)
            {
                printText(open);
            }

            if (hasAny && separator)
            {
                printText(separator);
            }

            hasAny = true;
            forEachChildPrint(node);

            if (afterChild)
            {
                printText(afterChild);    
            }
        }

        if ((!ifAny || ifAny && hasAny) && end)
        {
            printText(end);
        }
    }

    template <typename T>
    void forEachChildrenPrintFilterWithAppend(NodeArray<T> nodes, SyntaxKind filter, SyntaxKind withAppend, const char *open = nullptr, const char *separator = nullptr,
                              const char *end = nullptr, bool ifAny = false)
    {
        if (!ifAny && open)
        {
            printText(open);
        }

        auto hasAny = false;
        if (withAppend != SyntaxKind::Unknown)
        {
            if (!hasAny && ifAny && open)
            {
                printText(open);
            }

            out << Scanner::tokenStrings[withAppend];

            hasAny = true;
        }

        for (auto node : nodes)
        {
            if (node == filter || node == withAppend) continue;

            if (!hasAny && ifAny && open)
            {
                printText(open);
            }

            if (hasAny && separator)
            {
                printText(separator);
            }

            hasAny = true;
            forEachChildPrint(node);
        }

        if ((!ifAny || ifAny && hasAny) && end)
        {
            printText(end);
        }
    }

    // if return 'true' means stop
    void forEachChildPrint(Node node)
    {
        if (!node)
        {
            // empty node
            return;
        }

        // fake positive result to allow to run first command
        auto kind = (SyntaxKind)node;
        switch (kind)
        {
        case SyntaxKind::Identifier: {
            out << node.as<Identifier>()->escapedText.c_str();
            break;
        }
        case SyntaxKind::PrivateIdentifier: {
            out << "#" << node.as<PrivateIdentifier>()->escapedText.c_str();
            break;
        }
        case SyntaxKind::NumericLiteral: {
            out << node.as<NumericLiteral>()->text;
            break;
        }
        case SyntaxKind::BigIntLiteral: {
            out << node.as<BigIntLiteral>()->text;
            out << "n";
            break;
        }
        case SyntaxKind::StringLiteral: {
            auto stringLiteral = node.as<StringLiteral>();
            if (stringLiteral->singleQuote)
                out << "'";
            else 
                out << "\"";
            out << stringLiteral->text;
            if (stringLiteral->singleQuote)
                out << "'";
            else 
                out << "\"";
            break;
        }
        case SyntaxKind::RegularExpressionLiteral: {
            auto regularExpressionLiteral = node.as<RegularExpressionLiteral>();
            out << "/";
            out << regularExpressionLiteral->text;
            out << "/";
            break;
        }        
        case SyntaxKind::QualifiedName: {
            auto qualifiedName = node.as<QualifiedName>();
            forEachChildPrint(qualifiedName->left);
            out << ".";
            forEachChildPrint(qualifiedName->right);
            break;
        }
        case SyntaxKind::TypeParameter: {
            auto typeParameterDeclaration = node.as<TypeParameterDeclaration>();
            forEachChildPrint(typeParameterDeclaration->name);
            if (typeParameterDeclaration->constraint)
            {
                out << ((typeParameterDeclaration->parent == SyntaxKind::MappedType) ? " in " : " extends ");
                forEachChildPrint(typeParameterDeclaration->constraint);
            }

            if (typeParameterDeclaration->_default)
            {
                out << " = ";
                forEachChildPrint(typeParameterDeclaration->_default);
            }

            forEachChildPrint(typeParameterDeclaration->expression);
            break;
        }
        case SyntaxKind::ShorthandPropertyAssignment: {
            auto shorthandPropertyAssignment = node.as<ShorthandPropertyAssignment>();
            printDecorators(node);
            printModifiers(node);
            forEachChildPrint(shorthandPropertyAssignment->name);
            forEachChildPrint(shorthandPropertyAssignment->questionToken);
            forEachChildPrint(shorthandPropertyAssignment->exclamationToken);
            forEachChildPrint(shorthandPropertyAssignment->equalsToken);
            forEachChildPrint(shorthandPropertyAssignment->objectAssignmentInitializer);
            break;
        }
        case SyntaxKind::SpreadAssignment: {
            auto spreadAssignment = node.as<SpreadAssignment>();
            out << "...";
            forEachChildPrint(spreadAssignment->expression);
            break;
        }
        case SyntaxKind::Parameter: {
            printDecorators(node);
            printModifiers(node);
            auto parameterDeclaration = node.as<ParameterDeclaration>();
            forEachChildPrint(parameterDeclaration->dotDotDotToken);
            forEachChildPrint(parameterDeclaration->name);
            forEachChildPrint(parameterDeclaration->questionToken);
            if (parameterDeclaration->type)
                out << ": ";
            forEachChildPrint(parameterDeclaration->type);
            if (parameterDeclaration->initializer)
                out << " = ";
            forEachChildPrint(parameterDeclaration->initializer);
            break;
        }
        case SyntaxKind::PropertyDeclaration: {
            printDecorators(node);
            printModifiers(node);
            auto propertyDeclaration = node.as<PropertyDeclaration>();
            forEachChildPrint(propertyDeclaration->name);
            forEachChildPrint(propertyDeclaration->questionToken);
            forEachChildPrint(propertyDeclaration->exclamationToken);
            if (propertyDeclaration->type)
                out << ": ";
            forEachChildPrint(propertyDeclaration->type);
            if (propertyDeclaration->initializer)
                out << " = ";
            forEachChildPrint(propertyDeclaration->initializer);
            break;
        }
        case SyntaxKind::PropertySignature: {
            printDecorators(node);
            printModifiers(node);
            auto propertySignature = node.as<PropertySignature>();
            forEachChildPrint(propertySignature->name);
            forEachChildPrint(propertySignature->questionToken);
            if (propertySignature->type)
                out << ": ";
            forEachChildPrint(propertySignature->type);
            if (propertySignature->initializer)
                out << " = ";
            forEachChildPrint(propertySignature->initializer);
            break;
        }
        case SyntaxKind::PropertyAssignment: {
            printDecorators(node);
            printModifiers(node);
            auto propertyAssignment = node.as<PropertyAssignment>();
            forEachChildPrint(propertyAssignment->name);
            forEachChildPrint(propertyAssignment->questionToken);
            if (propertyAssignment->initializer)
                out << ": ";
            forEachChildPrint(propertyAssignment->initializer);
            break;
        }
        case SyntaxKind::VariableDeclaration: {
            auto variableDeclaration = node.as<VariableDeclaration>();
            printDecorators(node);
            printModifiers(node);
            forEachChildPrint(variableDeclaration->name);
            forEachChildPrint(variableDeclaration->exclamationToken);
            if (variableDeclaration->type)
                out << " : ";
            forEachChildPrint(variableDeclaration->type);
            if (variableDeclaration->initializer)
                out << " = ";
            forEachChildPrint(variableDeclaration->initializer);
            break;
        }
        case SyntaxKind::BindingElement: {
            printDecorators(node);
            printModifiers(node);
            auto bindingElement = node.as<BindingElement>();
            forEachChildPrint(bindingElement->dotDotDotToken);
            forEachChildPrint(bindingElement->propertyName);
            forEachChildPrint(bindingElement->name);
            forEachChildPrint(bindingElement->initializer);
            break;
        }
        case SyntaxKind::FunctionType:
        case SyntaxKind::ConstructorType:
        case SyntaxKind::CallSignature:
        case SyntaxKind::MethodSignature: {
            auto signatureDeclarationBase = node.as<SignatureDeclarationBase>();
            printDecorators(node);
            printModifiers(node);
            if (kind == SyntaxKind::ConstructorType)
                out << "new ";
            if (kind == SyntaxKind::MethodSignature)
                forEachChildPrint(signatureDeclarationBase->name);
            if (kind == SyntaxKind::MethodSignature)
                forEachChildPrint(signatureDeclarationBase->questionToken);
            forEachChildrenPrint(signatureDeclarationBase->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(signatureDeclarationBase->parameters, "(", ", ", ")");
            out << " => ";
            forEachChildPrint(signatureDeclarationBase->type);
            break;
        }
        case SyntaxKind::ConstructSignature: {
            auto signatureDeclarationBase = node.as<SignatureDeclarationBase>();
            printDecorators(node);
            printModifiers(node);
            out << "new ";
            forEachChildrenPrint(signatureDeclarationBase->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(signatureDeclarationBase->parameters, "(", ", ", ")");
            out << ": ";
            forEachChildPrint(signatureDeclarationBase->type);
            break;
        }
        case SyntaxKind::IndexSignature: {
            auto signatureDeclarationBase = node.as<SignatureDeclarationBase>();
            printDecorators(node);
            printModifiers(node);
            forEachChildrenPrint(signatureDeclarationBase->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(signatureDeclarationBase->parameters, "[", ", ", "]");
            out << ": ";
            forEachChildPrint(signatureDeclarationBase->type);
            break;
        }
        case SyntaxKind::MethodDeclaration:
        case SyntaxKind::Constructor:
        case SyntaxKind::GetAccessor:
        case SyntaxKind::SetAccessor:
        case SyntaxKind::FunctionExpression:
        case SyntaxKind::FunctionDeclaration:
        case SyntaxKind::ArrowFunction: {

            auto functionLikeDeclarationBase = node.as<FunctionLikeDeclarationBase>();
            functionLikeDeclarationBase->body->parent = functionLikeDeclarationBase;

            printDecorators(node);
            printModifiersWithMode(node);

            if (kind == SyntaxKind::FunctionExpression || kind == SyntaxKind::FunctionDeclaration)
                out << "function ";

            if (kind == SyntaxKind::Constructor)
            {
                out << "constructor";
            }

            if (kind == SyntaxKind::GetAccessor)
            {
                out << "get ";
            }

            if (kind == SyntaxKind::SetAccessor)
            {
                out << "set ";
            }

            forEachChildPrint(functionLikeDeclarationBase->asteriskToken);
            forEachChildPrint(functionLikeDeclarationBase->name);
            forEachChildPrint(functionLikeDeclarationBase->questionToken);
            forEachChildPrint(functionLikeDeclarationBase->exclamationToken);
            forEachChildrenPrint(functionLikeDeclarationBase->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(functionLikeDeclarationBase->parameters, "(", ", ", ")");
            if (functionLikeDeclarationBase->type)
                out << " : ";
            forEachChildPrint(functionLikeDeclarationBase->type);
            if (kind == SyntaxKind::ArrowFunction)
                forEachChildPrint(node.as<ArrowFunction>()->equalsGreaterThanToken);
            forEachChildPrint(functionLikeDeclarationBase->body);
            break;
        }
        case SyntaxKind::TypeReference: {
            auto typeReferenceNode = node.as<TypeReferenceNode>();
            forEachChildPrint(typeReferenceNode->typeName);
            forEachChildrenPrint(typeReferenceNode->typeArguments, "<", ", ", ">", true);
            break;
        }
        case SyntaxKind::TypePredicate: {
            auto typePredicateNode = node.as<TypePredicateNode>();
            forEachChildPrint(typePredicateNode->assertsModifier);
            forEachChildPrint(typePredicateNode->parameterName);
            forEachChildPrint(typePredicateNode->type);
            break;
        }
        case SyntaxKind::TypeQuery: {
            out << "typeof ";
            forEachChildPrint(node.as<TypeQueryNode>()->exprName);
            break;
        }
        case SyntaxKind::TypeLiteral: {
            out << "{";
            forEachChildrenPrint(node.as<TypeLiteralNode>()->members, " ", ", ", " ", true);
            out << "}";
            break;
        }
        case SyntaxKind::ArrayType: {
            forEachChildPrint(node.as<ArrayTypeNode>()->elementType);
            out << "[]";
            break;
        }
        case SyntaxKind::TupleType: {
            out << "[";
            forEachChildrenPrint(node.as<TupleTypeNode>()->elements, nullptr, ", ");
            out << "]";
            break;
        }
        case SyntaxKind::UnionType: {
            forEachChildrenPrint(node.as<UnionTypeNode>()->types, nullptr, " | ");
            break;
        }
        case SyntaxKind::IntersectionType: {
            forEachChildrenPrint(node.as<IntersectionTypeNode>()->types, nullptr, " & ");
            break;
        }
        case SyntaxKind::ConditionalType: {
            auto conditionalTypeNode = node.as<ConditionalTypeNode>();
            forEachChildPrint(conditionalTypeNode->checkType);
            out << " extends ";
            forEachChildPrint(conditionalTypeNode->extendsType);
            out << " ? ";
            forEachChildPrint(conditionalTypeNode->trueType);
            out << " : ";
            forEachChildPrint(conditionalTypeNode->falseType);
            break;
        }
        case SyntaxKind::InferType: {
            forEachChildPrint(node.as<InferTypeNode>()->typeParameter);
            break;
        }
        case SyntaxKind::ImportType: {
            auto importTypeNode = node.as<ImportTypeNode>();
            forEachChildPrint(importTypeNode->argument);
            forEachChildPrint(importTypeNode->qualifier);
            forEachChildrenPrint(importTypeNode->typeArguments, "<", ", ", ">", true);
            break;
        }
        case SyntaxKind::ParenthesizedType: {
            out << "(";
            forEachChildPrint(node.as<ParenthesizedTypeNode>()->type);
            out << ")";
            break;
        }
        case SyntaxKind::TypeOperator: {
            auto typeOperatorNode = node.as<TypeOperatorNode>();
            assert(Scanner::tokenStrings[typeOperatorNode->_operator].length() > 0);
            out << Scanner::tokenStrings[typeOperatorNode->_operator];            
            out << " ";
            forEachChildPrint(typeOperatorNode->type);
            break;
        }
        case SyntaxKind::IndexedAccessType: {
            auto indexedAccessTypeNode = node.as<IndexedAccessTypeNode>();
            forEachChildPrint(indexedAccessTypeNode->objectType);
            out << "[";
            forEachChildPrint(indexedAccessTypeNode->indexType);
            out << "]";
            break;
        }
        case SyntaxKind::MappedType: {
            auto mappedTypeNode = node.as<MappedTypeNode>();
            forEachChildPrint(mappedTypeNode->readonlyToken);
            out  << "[";
            mappedTypeNode->typeParameter->parent = mappedTypeNode;
            forEachChildPrint(mappedTypeNode->typeParameter);
            forEachChildPrint(mappedTypeNode->nameType);
            out << "]";
            forEachChildPrint(mappedTypeNode->questionToken);
            if (mappedTypeNode->type)
            {
                out << ": ";
                forEachChildPrint(mappedTypeNode->type);
            }

            break;
        }
        case SyntaxKind::LiteralType: {
            forEachChildPrint(node.as<LiteralTypeNode>()->literal);
            break;
        }
        case SyntaxKind::NamedTupleMember: {
            auto namedTupleMember = node.as<NamedTupleMember>();
            forEachChildPrint(namedTupleMember->dotDotDotToken);
            forEachChildPrint(namedTupleMember->name);
            forEachChildPrint(namedTupleMember->questionToken);
            forEachChildPrint(namedTupleMember->type);
            break;
        }
        case SyntaxKind::ObjectBindingPattern: {
            out << "{";
            forEachChildrenPrint(node.as<ObjectBindingPattern>()->elements, " ", ", ", " ", true);
            out << "}";
            break;
        }
        case SyntaxKind::ArrayBindingPattern: {
            out << "[";
            forEachChildrenPrint(node.as<ArrayBindingPattern>()->elements, " ", ", ", " ", true);
            out << "]";
            break;
        }
        case SyntaxKind::ArrayLiteralExpression: {
            out << "[";
            forEachChildrenPrint(node.as<ArrayLiteralExpression>()->elements, " ", ", ", " ", true);
            out << "]";
            break;
        }
        case SyntaxKind::ObjectLiteralExpression: {
            out << "{";
            forEachChildrenPrint(node.as<ObjectLiteralExpression>()->properties, " ", ", ", " ", true);
            out << "}";
            break;
        }
        case SyntaxKind::PropertyAccessExpression: {
            auto propertyAccessExpression = node.as<PropertyAccessExpression>();
            forEachChildPrint(propertyAccessExpression->expression);
            if (propertyAccessExpression->questionDotToken)
                forEachChildPrint(propertyAccessExpression->questionDotToken);
            else
                out << ".";
            forEachChildPrint(propertyAccessExpression->name);
            break;
        }
        case SyntaxKind::ElementAccessExpression: {
            auto elementAccessExpression = node.as<ElementAccessExpression>();
            forEachChildPrint(elementAccessExpression->expression);
            forEachChildPrint(elementAccessExpression->questionDotToken);
            out << "[";
            forEachChildPrint(elementAccessExpression->argumentExpression);
            out << "]";
            break;
        }
        case SyntaxKind::CallExpression: {
            auto callExpression = node.as<CallExpression>();
            forEachChildPrint(callExpression->expression);
            forEachChildPrint(callExpression->questionDotToken);
            forEachChildrenPrint(callExpression->typeArguments, "<", ", ", ">", true);
            out << "(";
            forEachChildrenPrint(callExpression->arguments, nullptr, ", ");
            out << ")";
            break;
        }
        case SyntaxKind::NewExpression: {
            auto newExpression = node.as<NewExpression>();
            out << "new ";
            forEachChildPrint(newExpression->expression);
            forEachChildrenPrint(newExpression->typeArguments, "<", ", ", ">", true);
            forEachChildrenPrint(newExpression->arguments, "(", ", ", ")");
            break;
        }
        case SyntaxKind::TaggedTemplateExpression: {
            auto taggedTemplateExpression = node.as<TaggedTemplateExpression>();
            forEachChildPrint(taggedTemplateExpression->tag);
            forEachChildPrint(taggedTemplateExpression->questionDotToken);
            forEachChildrenPrint(taggedTemplateExpression->typeArguments, "<", ", ", ">", true);
            forEachChildPrint(taggedTemplateExpression->_template);
            break;
        }
        case SyntaxKind::TypeAssertionExpression: {
            auto typeAssertion = node.as<TypeAssertion>();
            out << "<";
            if (isTypeWithParams(typeAssertion->type))
            {
                out << " ";
            }

            forEachChildPrint(typeAssertion->type);
            out << ">";
            forEachChildPrint(typeAssertion->expression);
            break;
        }
        case SyntaxKind::ParenthesizedExpression: {
            out << "(";
            forEachChildPrint(node.as<ParenthesizedExpression>()->expression);
            out << ")";
            break;
        }
        case SyntaxKind::DeleteExpression: {
            out << "delete ";
            forEachChildPrint(node.as<DeleteExpression>()->expression);
            break;
        }
        case SyntaxKind::TypeOfExpression: {
            out << "typeof ";
            forEachChildPrint(node.as<TypeOfExpression>()->expression);
            break;
        }
        case SyntaxKind::VoidExpression: {
            out << "void ";
            forEachChildPrint(node.as<VoidExpression>()->expression);
            break;
        }
        case SyntaxKind::PrefixUnaryExpression: {
            auto prefixUnaryExpression = node.as<PrefixUnaryExpression>();
            out << Scanner::tokenStrings[prefixUnaryExpression->_operator];
            forEachChildPrint(prefixUnaryExpression->operand);
            break;
        }
        case SyntaxKind::YieldExpression: {
            auto yieldExpression = node.as<YieldExpression>();
            out << "yield ";
            forEachChildPrint(yieldExpression->asteriskToken);
            forEachChildPrint(yieldExpression->expression);
            break;
        }
        case SyntaxKind::AwaitExpression: {
            out << "await ";
            forEachChildPrint(node.as<AwaitExpression>()->expression);
            break;
        }
        case SyntaxKind::PostfixUnaryExpression: {
            auto postfixUnaryExpression = node.as<PostfixUnaryExpression>();
            forEachChildPrint(postfixUnaryExpression->operand);
            out << Scanner::tokenStrings[postfixUnaryExpression->_operator];
            break;
        }
        case SyntaxKind::BinaryExpression: {
            auto binaryExpression = node.as<BinaryExpression>();
            forEachChildPrint(binaryExpression->left);
            forEachChildPrint(binaryExpression->operatorToken);
            forEachChildPrint(binaryExpression->right);
            break;
        }
        case SyntaxKind::AsExpression: {
            auto asExpression = node.as<AsExpression>();
            forEachChildPrint(asExpression->expression);
            out << " as ";
            forEachChildPrint(asExpression->type);
            break;
        }
        case SyntaxKind::NonNullExpression: {
            forEachChildPrint(node.as<NonNullExpression>()->expression);
            out << "!";
            break;
        }
        case SyntaxKind::MetaProperty: {
            forEachChildPrint(node.as<MetaProperty>()->name);
            break;
        }
        case SyntaxKind::ConditionalExpression: {
            auto conditionalExpression = node.as<ConditionalExpression>();
            forEachChildPrint(conditionalExpression->condition);
            out << " ";
            forEachChildPrint(conditionalExpression->questionToken);
            out << " ";
            forEachChildPrint(conditionalExpression->whenTrue);
            forEachChildPrint(conditionalExpression->colonToken);
            forEachChildPrint(conditionalExpression->whenFalse);
            break;
        }
        case SyntaxKind::SpreadElement: {
            out << "...";
            forEachChildPrint(node.as<SpreadElement>()->expression);
            break;
        }
        case SyntaxKind::Block:
        case SyntaxKind::ModuleBlock: {
            printBlock(node.as<Block>());
            break;
        }
        case SyntaxKind::SourceFile: {
            auto sourceFile = node.as<SourceFile>();
            printStatements(sourceFile->statements);
            forEachChildPrint(sourceFile->endOfFileToken);
            break;
        }
        case SyntaxKind::VariableStatement: {
            printDecorators(node);
            printModifiers(node);
            forEachChildPrint(node.as<VariableStatement>()->declarationList);
            break;
        }
        case SyntaxKind::VariableDeclarationList: {
            auto variableDeclarationList = node.as<VariableDeclarationList>();

            auto isLet = (variableDeclarationList->flags & NodeFlags::Let) == NodeFlags::Let;
            auto isConst = (variableDeclarationList->flags & NodeFlags::Const) == NodeFlags::Const;
            auto isVar = !isLet && !isConst;
            if (isLet)
                out << "let ";
            if (isConst)
                out << "const ";
            if (isVar)
                out << "var ";

            forEachChildrenPrint(variableDeclarationList->declarations, nullptr, ",");
            break;
        }
        case SyntaxKind::ExpressionStatement: {
            forEachChildPrint(node.as<ExpressionStatement>()->expression);
            break;
        }
        case SyntaxKind::IfStatement: {
            auto ifStatement = node.as<IfStatement>();
            out << "if (";
            forEachChildPrint(ifStatement->expression);
            out << ")";
            auto thenIsBlock = isBlockOrStatementWithBlock(ifStatement->thenStatement); 
            if (!thenIsBlock)
            {
                out << " ";
            }

            forEachChildPrint(ifStatement->thenStatement);
            if (ifStatement->elseStatement)
            {
                if (!thenIsBlock)
                {
                    out << " ";
                }
                else
                {
                    printIntent();
                }

                out << "else";
                if (!isBlock(ifStatement->elseStatement))
                {
                    out << " ";
                }

                forEachChildPrint(ifStatement->elseStatement);
            }

            break;
        }
        case SyntaxKind::DoStatement: {
            auto doStatement = node.as<DoStatement>();
            auto isBodyBlock = isBlock(doStatement->statement);
            out << "do";
            if (!isBodyBlock)
            {
                out << " ";
            }

            forEachChildPrint(doStatement->statement);
            if (isBodyBlock)
            {
                printIntent();
            }
            else
            {
                out << " ";
            }

            out << "while (";
            forEachChildPrint(doStatement->expression);
            out << ")";
            break;
        }
        case SyntaxKind::WhileStatement: {
            auto whileStatement = node.as<WhileStatement>();
            out << "while (";
            forEachChildPrint(whileStatement->expression);
            out << ")";
            forEachChildPrint(whileStatement->statement);
            break;
        }
        case SyntaxKind::ForStatement: {
            auto forStatement = node.as<ForStatement>();
            out << "for (";
            forEachChildPrint(forStatement->initializer);
            out << "; ";
            forEachChildPrint(forStatement->condition);
            out << "; ";
            forEachChildPrint(forStatement->incrementor);
            out << ") ";
            forEachChildPrint(forStatement->statement);
            break;
        }
        case SyntaxKind::ForInStatement: {
            auto forInStatement = node.as<ForInStatement>();
            out << "for (";
            forEachChildPrint(forInStatement->initializer);
            out << " in ";
            forEachChildPrint(forInStatement->expression);
            out << ") ";
            forEachChildPrint(forInStatement->statement);
            break;
        }
        case SyntaxKind::ForOfStatement: {
            auto forOfStatement = node.as<ForOfStatement>();
            out << "for ";
            forEachChildPrint(forOfStatement->awaitModifier);
            out << "(";
            forEachChildPrint(forOfStatement->initializer);
            out << " of ";
            forEachChildPrint(forOfStatement->expression);
            out << ") ";
            forEachChildPrint(forOfStatement->statement);
            break;
        }
        case SyntaxKind::ContinueStatement: {
            auto continueStatement = node.as<ContinueStatement>();            
            out << "continue";
            if (continueStatement->label)
            {
                out << " ";
                forEachChildPrint(continueStatement->label);
            }

            break;
        }
        case SyntaxKind::BreakStatement: {
            auto breakStatement = node.as<BreakStatement>();
            out << "break";
            if (breakStatement->label)
            {
                out << " ";
                forEachChildPrint(breakStatement->label);
            }

            break;
        }
        case SyntaxKind::ReturnStatement: {
            out << "return";
            auto returnStatement = node.as<ReturnStatement>();
            if (returnStatement->expression)
                out << " ";
            forEachChildPrint(returnStatement->expression);
            break;
        }
        case SyntaxKind::WithStatement: {
            auto withStatement = node.as<WithStatement>();
            out << "with (";
            forEachChildPrint(withStatement->expression);
            out << ")";
            forEachChildPrint(withStatement->statement);
            break;
        }
        case SyntaxKind::SwitchStatement: {
            auto switchStatement = node.as<SwitchStatement>();
            out << "switch (";
            forEachChildPrint(switchStatement->expression);
            out << ")";
            forEachChildPrint(switchStatement->caseBlock);
            break;
        }
        case SyntaxKind::CaseBlock: {
            printCaseBlock(node.as<CaseBlock>());
            break;
        }
        case SyntaxKind::CaseClause: {
            auto caseClause = node.as<CaseClause>();
            out << "case ";
            forEachChildPrint(caseClause->expression);
            out << ":";
            newLine();
            incIndent();
            printStatements(caseClause->statements);
            decIndent();
            break;
        }
        case SyntaxKind::DefaultClause: {
            out << "default:";
            newLine();
            incIndent();
            printStatements(node.as<DefaultClause>()->statements);
            decIndent();
            break;
        }
        case SyntaxKind::LabeledStatement: {
            auto labeledStatement = node.as<LabeledStatement>();
            forEachChildPrint(labeledStatement->label);
            out << ": ";
            forEachChildPrint(labeledStatement->statement);
            break;
        }
        case SyntaxKind::ThrowStatement: {
            out << "throw ";
            forEachChildPrint(node.as<ThrowStatement>()->expression);
            break;
        }
        case SyntaxKind::DebuggerStatement: {
            out << "debugger";
            break;
        }
        case SyntaxKind::TryStatement: {
            auto tryStatement = node.as<TryStatement>();
            out << "try";
            forEachChildPrint(tryStatement->tryBlock);
            forEachChildPrint(tryStatement->catchClause);
            if (tryStatement->finallyBlock)
            {
                printIntent();
                out << "finally";
                forEachChildPrint(tryStatement->finallyBlock);
            }

            break;
        }
        case SyntaxKind::CatchClause: {
            auto catchClause = node.as<CatchClause>();
            printIntent();
            out << "catch (";
            forEachChildPrint(catchClause->variableDeclaration);
            out << ")";
            forEachChildPrint(catchClause->block);
            break;
        }
        case SyntaxKind::Decorator: {
            forEachChildPrint(node.as<Decorator>()->expression);
            break;
        }
        case SyntaxKind::ClassDeclaration:
        case SyntaxKind::ClassExpression: {
            auto classLikeDeclaration = node.as<ClassLikeDeclaration>();
            printDecorators(node);
            printModifiersWithMode(node);
            out << "class ";
            forEachChildPrint(classLikeDeclaration->name);
            forEachChildrenPrint(classLikeDeclaration->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(classLikeDeclaration->heritageClauses);
            printMembersBlock(classLikeDeclaration, kind);
            break;
        }
        case SyntaxKind::InterfaceDeclaration: {
            auto interfaceDeclaration = node.as<InterfaceDeclaration>();
            printDecorators(node);
            printModifiersWithMode(node);
            out << "interface ";
            forEachChildPrint(interfaceDeclaration->name);
            forEachChildrenPrint(interfaceDeclaration->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(interfaceDeclaration->heritageClauses);
            printMembersBlock(interfaceDeclaration);
            break;
        }
        case SyntaxKind::TypeAliasDeclaration: {
            auto typeAliasDeclaration = node.as<TypeAliasDeclaration>();
            printDecorators(node);
            printModifiersWithMode(node);
            out << "type ";
            forEachChildPrint(typeAliasDeclaration->name);
            forEachChildrenPrint(typeAliasDeclaration->typeParameters, "<", ", ", ">", true);
            out << " = ";
            forEachChildPrint(typeAliasDeclaration->type);
            break;
        }
        case SyntaxKind::EnumDeclaration: {
            auto enumDeclaration = node.as<EnumDeclaration>();
            printDecorators(node);
            printModifiersWithMode(node);
            out << "enum ";
            forEachChildPrint(enumDeclaration->name);
            printMembersBlock(enumDeclaration);
            break;
        }
        case SyntaxKind::EnumMember: {
            auto enumMember = node.as<EnumMember>();
            forEachChildPrint(enumMember->name);
            if (enumMember->initializer)
                out << " = ";
            forEachChildPrint(enumMember->initializer);
            break;
        }
        case SyntaxKind::ModuleDeclaration: {
            auto moduleDeclaration = node.as<ModuleDeclaration>();
            printDecorators(node);
            printModifiers(node);
            out << "module ";
            forEachChildPrint(moduleDeclaration->name);
            forEachChildPrint(moduleDeclaration->body);
            break;
        }
        case SyntaxKind::ImportEqualsDeclaration: {
            auto importEqualsDeclaration = node.as<ImportEqualsDeclaration>();
            printDecorators(node);
            printModifiers(node);
            out << "import ";
            forEachChildPrint(importEqualsDeclaration->name);
            out << " = ";
            forEachChildPrint(importEqualsDeclaration->moduleReference);
            break;
        }
        case SyntaxKind::ImportDeclaration: {
            auto importDeclaration = node.as<ImportDeclaration>();
            printDecorators(node);
            printModifiers(node);
            out << "import ";
            if (importDeclaration->importClause)
            {
                forEachChildPrint(importDeclaration->importClause);
                out << " from ";
            }

            forEachChildPrint(importDeclaration->moduleSpecifier);
            break;
        }
        case SyntaxKind::ImportClause: {
            auto importClause = node.as<ImportClause>();
            
            if (importClause->name)
                forEachChildPrint(importClause->name);
            
            if (importClause->namedBindings)
            {
                out << "{";
                forEachChildPrint(importClause->namedBindings);
                out << "}";
            }

            break;
        }
        case SyntaxKind::NamespaceExportDeclaration: {
            out << "export as namespace ";
            forEachChildPrint(node.as<NamespaceExportDeclaration>()->name);
            break;
        }

        case SyntaxKind::NamespaceImport: {
            forEachChildPrint(node.as<NamespaceImport>()->name);
            break;
        }
        case SyntaxKind::NamespaceExport: {
            forEachChildPrint(node.as<NamespaceExport>()->name);
            break;
        }
        case SyntaxKind::NamedImports: {
            forEachChildrenPrint(node.as<NamedImports>()->elements, nullptr, ", ");
            break;
        }
        case SyntaxKind::NamedExports: {
            forEachChildrenPrint(node.as<NamedExports>()->elements);
            break;
        }
        case SyntaxKind::ExportDeclaration: {
            auto exportDeclaration = node.as<ExportDeclaration>();
            printDecorators(node);
            printModifiers(node);
            out << "export {";
            forEachChildPrint(exportDeclaration->exportClause);
            out << " as ";
            forEachChildPrint(exportDeclaration->moduleSpecifier);
            out << "}";
            break;
        }
        case SyntaxKind::ImportSpecifier: {
            auto importSpecifier = node.as<ImportSpecifier>();
            forEachChildPrint(importSpecifier->propertyName);
            forEachChildPrint(importSpecifier->name);
            break;
        }
        case SyntaxKind::ExportSpecifier: {
            auto exportSpecifier = node.as<ExportSpecifier>();
            forEachChildPrint(exportSpecifier->propertyName);
            forEachChildPrint(exportSpecifier->name);
            break;
        }
        case SyntaxKind::ExportAssignment: {
            printDecorators(node);
            printModifiers(node);
            out << "export = ";
            forEachChildPrint(node.as<ExportAssignment>()->expression);
            break;
        }
        case SyntaxKind::TemplateHead: {
            auto templateHead = node.as<TemplateHead>();
            out << templateHead->text;
            break;
        }
        case SyntaxKind::TemplateMiddle: {
            auto templateMiddle = node.as<TemplateMiddle>();
            out << templateMiddle->text;
            break;
        }
        case SyntaxKind::TemplateTail: {
            auto templateTail = node.as<TemplateTail>();
            out << templateTail->text;
            break;
        }
        case SyntaxKind::NoSubstitutionTemplateLiteral: {
            out << "`";
            auto noSubstitutionTemplateLiteral = node.as<NoSubstitutionTemplateLiteral>();
            out << noSubstitutionTemplateLiteral->rawText;
            out << "`";
            break;
        }
        case SyntaxKind::TemplateExpression: {
            out << "`";
            auto templateExpression = node.as<TemplateExpression>();
            forEachChildPrint(templateExpression->head);
            forEachChildrenPrint(templateExpression->templateSpans);
            out << "`";
            break;
        }
        case SyntaxKind::TemplateSpan: {
            auto templateSpan = node.as<TemplateSpan>();
            out << "${";
            forEachChildPrint(templateSpan->expression);
            out << "}";
            forEachChildPrint(templateSpan->literal);
            break;
        }
        case SyntaxKind::TemplateLiteralType: {
            auto templateLiteralTypeNode = node.as<TemplateLiteralTypeNode>();
            forEachChildPrint(templateLiteralTypeNode->head);
            forEachChildrenPrint(templateLiteralTypeNode->templateSpans);
            break;
        }
        case SyntaxKind::TemplateLiteralTypeSpan: {
            auto templateLiteralTypeSpan = node.as<TemplateLiteralTypeSpan>();
            forEachChildPrint(templateLiteralTypeSpan->type);
            forEachChildPrint(templateLiteralTypeSpan->literal);
            break;
        }
        case SyntaxKind::ComputedPropertyName: {
            out << "[";
            forEachChildPrint(node.as<ComputedPropertyName>()->expression);
            out << "]";
            break;
        }
        case SyntaxKind::HeritageClause: {
            out << " extends ";
            forEachChildrenPrint(node.as<HeritageClause>()->types);
            break;
        }
        case SyntaxKind::ExpressionWithTypeArguments: {
            auto expressionWithTypeArguments = node.as<ExpressionWithTypeArguments>();
            forEachChildPrint(expressionWithTypeArguments->expression);
            forEachChildrenPrint(expressionWithTypeArguments->typeArguments, "<", ", ", ">", true);
            break;
        }
        case SyntaxKind::ExternalModuleReference: {
            out << "require(";
            forEachChildPrint(node.as<ExternalModuleReference>()->expression);
            out << ")";
            break;
        }
        case SyntaxKind::MissingDeclaration: {
            printDecorators(node);
            break;
        }
        case SyntaxKind::CommaListExpression: {
            forEachChildrenPrint(node.as<CommaListExpression>()->elements, nullptr, ", ");
            break;
        }
        case SyntaxKind::OmittedExpression: {
            break;
        }
        case SyntaxKind::JsxElement: {
            auto jsxElement = node.as<JsxElement>();
            forEachChildPrint(jsxElement->openingElement);
            forEachChildrenPrint(jsxElement->children);
            forEachChildPrint(jsxElement->closingElement);
            break;
        }
        case SyntaxKind::JsxFragment: {
            auto jsxFragment = node.as<JsxFragment>();
            forEachChildPrint(jsxFragment->openingFragment);
            forEachChildrenPrint(jsxFragment->children);
            forEachChildPrint(jsxFragment->closingFragment);
            break;
        }
        case SyntaxKind::JsxSelfClosingElement: {
            auto jsxSelfClosingElement = node.as<JsxSelfClosingElement>();
            forEachChildPrint(jsxSelfClosingElement->tagName);
            forEachChildrenPrint(jsxSelfClosingElement->typeArguments, "<", ", ", ">", true);
            forEachChildPrint(jsxSelfClosingElement->attributes);
            break;
        }
        case SyntaxKind::JsxOpeningElement: {
            auto jsxOpeningElement = node.as<JsxOpeningElement>();
            forEachChildPrint(jsxOpeningElement->tagName);
            forEachChildrenPrint(jsxOpeningElement->typeArguments, "<", ", ", ">", true);
            forEachChildPrint(jsxOpeningElement->attributes);
            break;
        }
        case SyntaxKind::JsxAttributes: {
            forEachChildrenPrint(node.as<JsxAttributes>()->properties);
            break;
        }
        case SyntaxKind::JsxAttribute: {
            auto jsxAttribute = node.as<JsxAttribute>();
            forEachChildPrint(jsxAttribute->name);
            forEachChildPrint(jsxAttribute->initializer);
            break;
        }
        case SyntaxKind::JsxSpreadAttribute: {
            forEachChildPrint(node.as<JsxSpreadAttribute>()->expression);
            break;
        }
        case SyntaxKind::JsxExpression: {
            auto jsxExpression = node.as<JsxExpression>();
            forEachChildPrint(jsxExpression->dotDotDotToken);
            forEachChildPrint(jsxExpression->expression);
            break;
        }
        case SyntaxKind::JsxClosingElement: {
            forEachChildPrint(node.as<JsxClosingElement>()->tagName);
            break;
        }

        case SyntaxKind::OptionalType: {
            forEachChildPrint(node.as<OptionalTypeNode>()->type);
            break;
        }
        case SyntaxKind::RestType: {
            forEachChildPrint(node.as<RestTypeNode>()->type);
            break;
        }
        case SyntaxKind::ThisType: {
            out << "this";
            break;
        }
        case SyntaxKind::JSDocTypeExpression: {
            forEachChildPrint(node.as<JSDocTypeExpression>()->type);
            break;
        }
        case SyntaxKind::JSDocNonNullableType: {
            forEachChildPrint(node.as<JSDocNonNullableType>()->type);
            break;
        }
        case SyntaxKind::JSDocNullableType: {
            forEachChildPrint(node.as<JSDocNullableType>()->type);
            break;
        }
        case SyntaxKind::JSDocOptionalType: {
            forEachChildPrint(node.as<JSDocOptionalType>()->type);
            break;
        }
        case SyntaxKind::JSDocVariadicType: {
            forEachChildPrint(node.as<JSDocVariadicType>()->type);
            break;
        }
        case SyntaxKind::JSDocFunctionType: {
            auto jsDocFunctionType = node.as<JSDocFunctionType>();
            forEachChildrenPrint(jsDocFunctionType->parameters, "(", ", ", ")");
            forEachChildPrint(jsDocFunctionType->type);
            break;
        }
        case SyntaxKind::JSDocComment: {
            forEachChildrenPrint(node.as<JSDoc>()->tags);
            break;
        }
        case SyntaxKind::JSDocSeeTag: {
            auto jsDocSeeTag = node.as<JSDocSeeTag>();
            forEachChildPrint(jsDocSeeTag->tagName);
            forEachChildPrint(jsDocSeeTag->name);
            break;
        }
        case SyntaxKind::JSDocNameReference: {
            forEachChildPrint(node.as<JSDocNameReference>()->name);
            break;
        }
        case SyntaxKind::JSDocParameterTag:
        case SyntaxKind::JSDocPropertyTag: {
            auto jsDocPropertyLikeTag = node.as<JSDocPropertyLikeTag>();
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            if (jsDocPropertyLikeTag->isNameFirst)
            {
                forEachChildPrint(jsDocPropertyLikeTag->name);
                forEachChildPrint(jsDocPropertyLikeTag->typeExpression);
            }
            else
            {
                forEachChildPrint(jsDocPropertyLikeTag->typeExpression);
                forEachChildPrint(jsDocPropertyLikeTag->name);
            }
            break;
        }
        case SyntaxKind::JSDocAuthorTag: {
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            break;
        }
        case SyntaxKind::JSDocImplementsTag: {
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocImplementsTag>()->_class);
            break;
        }
        case SyntaxKind::JSDocAugmentsTag: {
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocAugmentsTag>()->_class);
            break;
        }
        case SyntaxKind::JSDocTemplateTag: {
            auto jsDocTemplateTag = node.as<JSDocTemplateTag>();
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(jsDocTemplateTag->constraint);
            forEachChildrenPrint(jsDocTemplateTag->typeParameters, "<", ", ", ">", true);
            break;
        }
        case SyntaxKind::JSDocTypedefTag: {
            auto jsDocTypedefTag = node.as<JSDocTypedefTag>();
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            if (jsDocTypedefTag->typeExpression && jsDocTypedefTag->typeExpression == SyntaxKind::JSDocTypeExpression)
            {
                forEachChildPrint(jsDocTypedefTag->typeExpression);
                forEachChildPrint(jsDocTypedefTag->fullName);
            }
            else
            {
                forEachChildPrint(jsDocTypedefTag->fullName);
                forEachChildPrint(jsDocTypedefTag->typeExpression);
            }
            break;
        }
        case SyntaxKind::JSDocCallbackTag: {
            auto jsDocCallbackTag = node.as<JSDocCallbackTag>();
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(jsDocCallbackTag->fullName);
            forEachChildPrint(jsDocCallbackTag->typeExpression);
            break;
        }
        case SyntaxKind::JSDocReturnTag: {
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocReturnTag>()->typeExpression);
            break;
        }
        case SyntaxKind::JSDocTypeTag: {
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocTypeTag>()->typeExpression);
            break;
        }
        case SyntaxKind::JSDocThisTag: {
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocThisTag>()->typeExpression);
            break;
        }
        case SyntaxKind::JSDocEnumTag: {
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocEnumTag>()->typeExpression);
            break;
        }
        case SyntaxKind::JSDocSignature: {
            auto jsDocSignature = node.as<JSDocSignature>();
            forEachChildrenPrint(jsDocSignature->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(jsDocSignature->parameters, "(", ", ", ")");
            forEachChildPrint(jsDocSignature->type);
            break;
        }
        case SyntaxKind::JSDocTypeLiteral: {
            forEachChildrenPrint(node.as<JSDocTypeLiteral>()->jsDocPropertyTags);
            break;
        }
        case SyntaxKind::JSDocTag:
        case SyntaxKind::JSDocClassTag:
        case SyntaxKind::JSDocPublicTag:
        case SyntaxKind::JSDocPrivateTag:
        case SyntaxKind::JSDocProtectedTag:
        case SyntaxKind::JSDocReadonlyTag: {
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            break;
        }
        case SyntaxKind::PartiallyEmittedExpression: {
            forEachChildPrint(node.as<PartiallyEmittedExpression>()->expression);
            break;
        }
        case SyntaxKind::SemicolonClassElement: {
            out << ";";
            break;
        }
        case SyntaxKind::TrueKeyword:
        case SyntaxKind::FalseKeyword:
        case SyntaxKind::NullKeyword:
        case SyntaxKind::StringKeyword:
        case SyntaxKind::NumberKeyword:
        case SyntaxKind::ThisKeyword:
        case SyntaxKind::ConstKeyword:
        case SyntaxKind::UndefinedKeyword:
        case SyntaxKind::BooleanKeyword:
        case SyntaxKind::AnyKeyword:
        case SyntaxKind::VoidKeyword:
        case SyntaxKind::DeclareKeyword:
        case SyntaxKind::ReadonlyKeyword:
        case SyntaxKind::ObjectKeyword:
        case SyntaxKind::NeverKeyword:
        case SyntaxKind::InKeyword:
        case SyntaxKind::UnknownKeyword:
        case SyntaxKind::AbstractKeyword:
        case SyntaxKind::StaticKeyword:
        case SyntaxKind::PublicKeyword:
        case SyntaxKind::ProtectedKeyword:
        case SyntaxKind::PrivateKeyword:
        case SyntaxKind::SuperKeyword:
        case SyntaxKind::DefaultKeyword:
        case SyntaxKind::AsyncKeyword:
        case SyntaxKind::AwaitKeyword:
        case SyntaxKind::ImportKeyword:
        case SyntaxKind::BigIntKeyword:
        case SyntaxKind::SymbolKeyword:
        case SyntaxKind::InstanceOfKeyword:
        case SyntaxKind::AssertsKeyword:
        case SyntaxKind::ExportKeyword: {
            assert(Scanner::tokenStrings[node->_kind].length() > 0);
            out << Scanner::tokenStrings[node->_kind];
            break;
        }
        case SyntaxKind::ColonToken:
        case SyntaxKind::CommaToken:
        case SyntaxKind::EqualsToken:
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
        case SyntaxKind::EqualsGreaterThanToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::LessThanEqualsToken:
        case SyntaxKind::LessThanLessThanToken:
        case SyntaxKind::LessThanLessThanEqualsToken:
        case SyntaxKind::GreaterThanToken:
        case SyntaxKind::GreaterThanEqualsToken:
        case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
        case SyntaxKind::GreaterThanGreaterThanToken:
        case SyntaxKind::GreaterThanGreaterThanEqualsToken:
        case SyntaxKind::GreaterThanGreaterThanGreaterThanEqualsToken:
        case SyntaxKind::PlusToken:
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::PlusEqualsToken:
        case SyntaxKind::AsteriskToken:
        case SyntaxKind::AsteriskEqualsToken:
        case SyntaxKind::AsteriskAsteriskToken:
        case SyntaxKind::AsteriskAsteriskEqualsToken:
        case SyntaxKind::AmpersandToken:
        case SyntaxKind::AmpersandEqualsToken:
        case SyntaxKind::AmpersandAmpersandToken:
        case SyntaxKind::AmpersandAmpersandEqualsToken:
        case SyntaxKind::BarToken:
        case SyntaxKind::BarEqualsToken:
        case SyntaxKind::BarBarToken:
        case SyntaxKind::BarBarEqualsToken:
        case SyntaxKind::QuestionQuestionToken:
        case SyntaxKind::QuestionQuestionEqualsToken:
        case SyntaxKind::MinusToken:
        case SyntaxKind::MinusMinusToken:
        case SyntaxKind::MinusEqualsToken:
        case SyntaxKind::SlashToken:
        case SyntaxKind::SlashEqualsToken:
        case SyntaxKind::CaretToken:
        case SyntaxKind::CaretEqualsToken:
        case SyntaxKind::PercentToken:
        case SyntaxKind::PercentEqualsToken: {
            assert(Scanner::tokenStrings[node->_kind].length() > 0);
            out << " " << Scanner::tokenStrings[node->_kind] << " ";
            break;
        }
        case SyntaxKind::ExclamationToken:        
        case SyntaxKind::DotDotDotToken:
        case SyntaxKind::QuestionToken:
        case SyntaxKind::QuestionDotToken: {
            assert(Scanner::tokenStrings[node->_kind].length() > 0);
            out << Scanner::tokenStrings[node->_kind];
            break;
        }
        case SyntaxKind::EmptyStatement:
            break;
        case SyntaxKind::EndOfFileToken:
            break;
        default:
            out << "[MISSING " << Scanner::tokenToText[node->_kind] << "]";
            break;
        }
    }
};

void print(ts::Node node)
{
    Printer<std::wostream> printer(std::wcout);
    printer.printNode(node);
}

} // namespace ts

#endif // DUMP_H