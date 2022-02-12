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

  public:
    Printer(OUT &out) : out(out)
    {
    }

    void printNode(ts::Node node)
    {        
        forEachChildPrint(node);
    }

  protected:
    template <typename T> 
    void forEachChildrenPrint(NodeArray<T> nodes, const char* open = nullptr, const char* separator = nullptr, const char* end = nullptr, bool ifAny = false)
    {        
        if (!ifAny && open)
        {
            out << open;
        }
        
        auto hasAny = false;
        for (auto node : nodes)
        {            
            if (!hasAny && ifAny && open)
            {
                out << open;                    
            }

            if (hasAny && separator)
            {
                out << separator;                    
            }

            hasAny = true;
            forEachChildPrint(node);
        }

        if ((!ifAny || ifAny && hasAny) && end)
        {
            out << end;
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
        case SyntaxKind::Identifier:
            out << node.as<Identifier>()->escapedText.c_str();
            break;
        case SyntaxKind::QualifiedName:            
            forEachChildPrint(node.as<QualifiedName>()->left);            
            forEachChildPrint(node.as<QualifiedName>()->right);
            break;
        case SyntaxKind::TypeParameter:            
            forEachChildPrint(node.as<TypeParameterDeclaration>()->name);
            forEachChildPrint(node.as<TypeParameterDeclaration>()->constraint);
            forEachChildPrint(node.as<TypeParameterDeclaration>()->_default);
            forEachChildPrint(node.as<TypeParameterDeclaration>()->expression);
            break;
        case SyntaxKind::ShorthandPropertyAssignment:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<ShorthandPropertyAssignment>()->name);
            forEachChildPrint(node.as<ShorthandPropertyAssignment>()->questionToken);
            forEachChildPrint(node.as<ShorthandPropertyAssignment>()->exclamationToken);
            forEachChildPrint(node.as<ShorthandPropertyAssignment>()->equalsToken);
            forEachChildPrint(node.as<ShorthandPropertyAssignment>()->objectAssignmentInitializer);
            break;
        case SyntaxKind::SpreadAssignment:
            forEachChildPrint(node.as<SpreadAssignment>()->expression);
            break;
        case SyntaxKind::Parameter:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<ParameterDeclaration>()->dotDotDotToken);
            forEachChildPrint(node.as<ParameterDeclaration>()->name);
            forEachChildPrint(node.as<ParameterDeclaration>()->questionToken);
            forEachChildPrint(node.as<ParameterDeclaration>()->type);
            forEachChildPrint(node.as<ParameterDeclaration>()->initializer);
            break;
        case SyntaxKind::PropertyDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<PropertyDeclaration>()->name);
            forEachChildPrint(node.as<PropertyDeclaration>()->questionToken);
            forEachChildPrint(node.as<PropertyDeclaration>()->exclamationToken);
            forEachChildPrint(node.as<PropertyDeclaration>()->type);
            forEachChildPrint(node.as<PropertyDeclaration>()->initializer);
            break;
        case SyntaxKind::PropertySignature:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<PropertySignature>()->name);
            forEachChildPrint(node.as<PropertySignature>()->questionToken);
            forEachChildPrint(node.as<PropertySignature>()->type);
            forEachChildPrint(node.as<PropertySignature>()->initializer);
            break;
        case SyntaxKind::PropertyAssignment:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<PropertyAssignment>()->name);
            forEachChildPrint(node.as<PropertyAssignment>()->questionToken);
            forEachChildPrint(node.as<PropertyAssignment>()->initializer);
            break;
        case SyntaxKind::VariableDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<VariableDeclaration>()->name);
            forEachChildPrint(node.as<VariableDeclaration>()->exclamationToken);
            forEachChildPrint(node.as<VariableDeclaration>()->type);
            forEachChildPrint(node.as<VariableDeclaration>()->initializer);
            break;
        case SyntaxKind::BindingElement:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<BindingElement>()->dotDotDotToken);
            forEachChildPrint(node.as<BindingElement>()->propertyName);
            forEachChildPrint(node.as<BindingElement>()->name);
            forEachChildPrint(node.as<BindingElement>()->initializer);
            break;
        case SyntaxKind::FunctionType:
        case SyntaxKind::ConstructorType:
        case SyntaxKind::CallSignature:
        case SyntaxKind::ConstructSignature:
        case SyntaxKind::IndexSignature:
        case SyntaxKind::MethodSignature:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            if (kind == SyntaxKind::MethodSignature)                
                forEachChildPrint(node.as<SignatureDeclarationBase>()->name);
            if (kind == SyntaxKind::MethodSignature)                
                forEachChildPrint(node.as<SignatureDeclarationBase>()->questionToken);
            forEachChildrenPrint(node.as<SignatureDeclarationBase>()->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(node.as<SignatureDeclarationBase>()->parameters, "(", ", ", ")");
            forEachChildPrint(node.as<SignatureDeclarationBase>()->type);
            break;
        case SyntaxKind::MethodDeclaration:
        case SyntaxKind::Constructor:
        case SyntaxKind::GetAccessor:
        case SyntaxKind::SetAccessor:
        case SyntaxKind::FunctionExpression:
        case SyntaxKind::FunctionDeclaration:
        case SyntaxKind::ArrowFunction:

            if (kind == SyntaxKind::FunctionExpression
                || kind == SyntaxKind::FunctionDeclaration)                
                out << "function ";

            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<FunctionLikeDeclarationBase>()->asteriskToken);
            forEachChildPrint(node.as<FunctionLikeDeclarationBase>()->name);
            forEachChildPrint(node.as<FunctionLikeDeclarationBase>()->questionToken);
            forEachChildPrint(node.as<FunctionLikeDeclarationBase>()->exclamationToken);
            forEachChildrenPrint(node.as<FunctionLikeDeclarationBase>()->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(node.as<FunctionLikeDeclarationBase>()->parameters, "(", ", ", ")");
            forEachChildPrint(node.as<FunctionLikeDeclarationBase>()->type);
            if (kind == SyntaxKind::ArrowFunction)                
                forEachChildPrint(node.as<ArrowFunction>()->equalsGreaterThanToken);
            forEachChildPrint(node.as<FunctionLikeDeclarationBase>()->body);
            break;
        case SyntaxKind::TypeReference:
            forEachChildPrint(node.as<TypeReferenceNode>()->typeName);
            forEachChildrenPrint(node.as<TypeReferenceNode>()->typeArguments);
            break;
        case SyntaxKind::TypePredicate:
            forEachChildPrint(node.as<TypePredicateNode>()->assertsModifier);
            forEachChildPrint(node.as<TypePredicateNode>()->parameterName);
            forEachChildPrint(node.as<TypePredicateNode>()->type);
            break;
        case SyntaxKind::TypeQuery:
            forEachChildPrint(node.as<TypeQueryNode>()->exprName);
            break;
        case SyntaxKind::TypeLiteral:
            forEachChildrenPrint(node.as<TypeLiteralNode>()->members);
            break;
        case SyntaxKind::ArrayType:
            forEachChildPrint(node.as<ArrayTypeNode>()->elementType);
            break;
        case SyntaxKind::TupleType:
            forEachChildrenPrint(node.as<TupleTypeNode>()->elements);
            break;
        case SyntaxKind::UnionType:
            forEachChildrenPrint(node.as<UnionTypeNode>()->types);
            break;
        case SyntaxKind::IntersectionType:
            forEachChildrenPrint(node.as<IntersectionTypeNode>()->types);
            break;
        case SyntaxKind::ConditionalType:
            forEachChildPrint(node.as<ConditionalTypeNode>()->checkType);
            forEachChildPrint(node.as<ConditionalTypeNode>()->extendsType);
            forEachChildPrint(node.as<ConditionalTypeNode>()->trueType);
            forEachChildPrint(node.as<ConditionalTypeNode>()->falseType);
            break;
        case SyntaxKind::InferType:
            forEachChildPrint(node.as<InferTypeNode>()->typeParameter);
            break;
        case SyntaxKind::ImportType:
            forEachChildPrint(node.as<ImportTypeNode>()->argument);
            forEachChildPrint(node.as<ImportTypeNode>()->qualifier);
            forEachChildrenPrint(node.as<ImportTypeNode>()->typeArguments);
            break;
        case SyntaxKind::ParenthesizedType:
            forEachChildPrint(node.as<ParenthesizedTypeNode>()->type);
            break;
        case SyntaxKind::TypeOperator:
            forEachChildPrint(node.as<TypeOperatorNode>()->type);
            break;
        case SyntaxKind::IndexedAccessType:
            forEachChildPrint(node.as<IndexedAccessTypeNode>()->objectType);
            forEachChildPrint(node.as<IndexedAccessTypeNode>()->indexType);
            break;
        case SyntaxKind::MappedType:
            forEachChildPrint(node.as<MappedTypeNode>()->readonlyToken);
            forEachChildPrint(node.as<MappedTypeNode>()->typeParameter);
            forEachChildPrint(node.as<MappedTypeNode>()->nameType);
            forEachChildPrint(node.as<MappedTypeNode>()->questionToken);
            forEachChildPrint(node.as<MappedTypeNode>()->type);
            break;
        case SyntaxKind::LiteralType:
            forEachChildPrint(node.as<LiteralTypeNode>()->literal);
            break;
        case SyntaxKind::NamedTupleMember:
            forEachChildPrint(node.as<NamedTupleMember>()->dotDotDotToken);
            forEachChildPrint(node.as<NamedTupleMember>()->name);
            forEachChildPrint(node.as<NamedTupleMember>()->questionToken);
            forEachChildPrint(node.as<NamedTupleMember>()->type);
            break;
        case SyntaxKind::ObjectBindingPattern:
            forEachChildrenPrint(node.as<ObjectBindingPattern>()->elements);
            break;
        case SyntaxKind::ArrayBindingPattern:
            forEachChildrenPrint(node.as<ArrayBindingPattern>()->elements);
            break;
        case SyntaxKind::ArrayLiteralExpression:
            forEachChildrenPrint(node.as<ArrayLiteralExpression>()->elements);
            break;
        case SyntaxKind::ObjectLiteralExpression:
            forEachChildrenPrint(node.as<ObjectLiteralExpression>()->properties);
            break;
        case SyntaxKind::PropertyAccessExpression:
            forEachChildPrint(node.as<PropertyAccessExpression>()->expression);
            forEachChildPrint(node.as<PropertyAccessExpression>()->questionDotToken);
            forEachChildPrint(node.as<PropertyAccessExpression>()->name);
            break;
        case SyntaxKind::ElementAccessExpression:
            forEachChildPrint(node.as<ElementAccessExpression>()->expression);
            forEachChildPrint(node.as<ElementAccessExpression>()->questionDotToken);
            forEachChildPrint(node.as<ElementAccessExpression>()->argumentExpression);
            break;
        case SyntaxKind::CallExpression:
            forEachChildPrint(node.as<CallExpression>()->expression);
            forEachChildPrint(node.as<CallExpression>()->questionDotToken);
            forEachChildrenPrint(node.as<CallExpression>()->typeArguments);
            forEachChildrenPrint(node.as<CallExpression>()->arguments);
            break;
        case SyntaxKind::NewExpression:
            forEachChildPrint(node.as<NewExpression>()->expression);
            forEachChildrenPrint(node.as<NewExpression>()->typeArguments);
            forEachChildrenPrint(node.as<NewExpression>()->arguments);
            break;
        case SyntaxKind::TaggedTemplateExpression:
            forEachChildPrint(node.as<TaggedTemplateExpression>()->tag);
            forEachChildPrint(node.as<TaggedTemplateExpression>()->questionDotToken);
            forEachChildrenPrint(node.as<TaggedTemplateExpression>()->typeArguments);
            forEachChildPrint(node.as<TaggedTemplateExpression>()->_template);
            break;
        case SyntaxKind::TypeAssertionExpression:
            forEachChildPrint(node.as<TypeAssertion>()->type);
            forEachChildPrint(node.as<TypeAssertion>()->expression);
            break;
        case SyntaxKind::ParenthesizedExpression:
            forEachChildPrint(node.as<ParenthesizedExpression>()->expression);
            break;
        case SyntaxKind::DeleteExpression:
            forEachChildPrint(node.as<DeleteExpression>()->expression);
            break;
        case SyntaxKind::TypeOfExpression:
            forEachChildPrint(node.as<TypeOfExpression>()->expression);
            break;
        case SyntaxKind::VoidExpression:
            forEachChildPrint(node.as<VoidExpression>()->expression);
            break;
        case SyntaxKind::PrefixUnaryExpression:
            forEachChildPrint(node.as<PrefixUnaryExpression>()->operand);
            break;
        case SyntaxKind::YieldExpression:
            forEachChildPrint(node.as<YieldExpression>()->asteriskToken);
            forEachChildPrint(node.as<YieldExpression>()->expression);
            break;
        case SyntaxKind::AwaitExpression:
            forEachChildPrint(node.as<AwaitExpression>()->expression);
            break;
        case SyntaxKind::PostfixUnaryExpression:
            forEachChildPrint(node.as<PostfixUnaryExpression>()->operand);
            break;
        case SyntaxKind::BinaryExpression:
            forEachChildPrint(node.as<BinaryExpression>()->left);
            forEachChildPrint(node.as<BinaryExpression>()->operatorToken);
            forEachChildPrint(node.as<BinaryExpression>()->right);
            break;
        case SyntaxKind::AsExpression:
            forEachChildPrint(node.as<AsExpression>()->expression);
            forEachChildPrint(node.as<AsExpression>()->type);
            break;
        case SyntaxKind::NonNullExpression:
            forEachChildPrint(node.as<NonNullExpression>()->expression);
            break;
        case SyntaxKind::MetaProperty:
            forEachChildPrint(node.as<MetaProperty>()->name);
            break;
        case SyntaxKind::ConditionalExpression:
            forEachChildPrint(node.as<ConditionalExpression>()->condition);
            forEachChildPrint(node.as<ConditionalExpression>()->questionToken);
            forEachChildPrint(node.as<ConditionalExpression>()->whenTrue);
            forEachChildPrint(node.as<ConditionalExpression>()->colonToken);
            forEachChildPrint(node.as<ConditionalExpression>()->whenFalse);
            break;
        case SyntaxKind::SpreadElement:
            forEachChildPrint(node.as<SpreadElement>()->expression);
            break;
        case SyntaxKind::Block:
        case SyntaxKind::ModuleBlock:
            forEachChildrenPrint(node.as<Block>()->statements);
            break;
        case SyntaxKind::SourceFile:
            forEachChildrenPrint(node.as<SourceFile>()->statements);
            forEachChildPrint(node.as<SourceFile>()->endOfFileToken);
            break;
        case SyntaxKind::VariableStatement:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<VariableStatement>()->declarationList);
            break;
        case SyntaxKind::VariableDeclarationList:
            forEachChildrenPrint(node.as<VariableDeclarationList>()->declarations);
            break;
        case SyntaxKind::ExpressionStatement:
            forEachChildPrint(node.as<ExpressionStatement>()->expression);
            break;
        case SyntaxKind::IfStatement:
            forEachChildPrint(node.as<IfStatement>()->expression);
            forEachChildPrint(node.as<IfStatement>()->thenStatement);
            forEachChildPrint(node.as<IfStatement>()->elseStatement);
            break;
        case SyntaxKind::DoStatement:
            forEachChildPrint(node.as<DoStatement>()->statement);
            forEachChildPrint(node.as<DoStatement>()->expression);
            break;
        case SyntaxKind::WhileStatement:
            forEachChildPrint(node.as<WhileStatement>()->expression);
            forEachChildPrint(node.as<WhileStatement>()->statement);
            break;
        case SyntaxKind::ForStatement:
            forEachChildPrint(node.as<ForStatement>()->initializer);
            forEachChildPrint(node.as<ForStatement>()->condition);
            forEachChildPrint(node.as<ForStatement>()->incrementor);
            forEachChildPrint(node.as<ForStatement>()->statement);
            break;
        case SyntaxKind::ForInStatement:
            forEachChildPrint(node.as<ForInStatement>()->initializer);
            forEachChildPrint(node.as<ForInStatement>()->expression);
            forEachChildPrint(node.as<ForInStatement>()->statement);
            break;
        case SyntaxKind::ForOfStatement:
            forEachChildPrint(node.as<ForOfStatement>()->awaitModifier);
            forEachChildPrint(node.as<ForOfStatement>()->initializer);
            forEachChildPrint(node.as<ForOfStatement>()->expression);
            forEachChildPrint(node.as<ForOfStatement>()->statement);
            break;
        case SyntaxKind::ContinueStatement:
            forEachChildPrint(node.as<ContinueStatement>()->label);
            break;
        case SyntaxKind::BreakStatement:
            forEachChildPrint(node.as<BreakStatement>()->label);
            break;
        case SyntaxKind::ReturnStatement:
            forEachChildPrint(node.as<ReturnStatement>()->expression);
            break;
        case SyntaxKind::WithStatement:
            forEachChildPrint(node.as<WithStatement>()->expression);
            forEachChildPrint(node.as<WithStatement>()->statement);
            break;
        case SyntaxKind::SwitchStatement:
            forEachChildPrint(node.as<SwitchStatement>()->expression);
            forEachChildPrint(node.as<SwitchStatement>()->caseBlock);
            break;
        case SyntaxKind::CaseBlock:
            forEachChildrenPrint(node.as<CaseBlock>()->clauses);
            break;
        case SyntaxKind::CaseClause:
            forEachChildPrint(node.as<CaseClause>()->expression);
            forEachChildrenPrint(node.as<CaseClause>()->statements);
            break;
        case SyntaxKind::DefaultClause:
            forEachChildrenPrint(node.as<DefaultClause>()->statements);
            break;
        case SyntaxKind::LabeledStatement:
            forEachChildPrint(node.as<LabeledStatement>()->label);
            forEachChildPrint(node.as<LabeledStatement>()->statement);
            break;
        case SyntaxKind::ThrowStatement:
            forEachChildPrint(node.as<ThrowStatement>()->expression);
            break;
        case SyntaxKind::TryStatement:
            forEachChildPrint(node.as<TryStatement>()->tryBlock);
            forEachChildPrint(node.as<TryStatement>()->catchClause);
            forEachChildPrint(node.as<TryStatement>()->finallyBlock);
            break;
        case SyntaxKind::CatchClause:
            forEachChildPrint(node.as<CatchClause>()->variableDeclaration);
            forEachChildPrint(node.as<CatchClause>()->block);
            break;
        case SyntaxKind::Decorator:
            forEachChildPrint(node.as<Decorator>()->expression);
            break;
        case SyntaxKind::ClassDeclaration:
        case SyntaxKind::ClassExpression:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<ClassLikeDeclaration>()->name);
            forEachChildrenPrint(node.as<ClassLikeDeclaration>()->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(node.as<ClassLikeDeclaration>()->heritageClauses);
            forEachChildrenPrint(node.as<ClassLikeDeclaration>()->members);
            break;
        case SyntaxKind::InterfaceDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<InterfaceDeclaration>()->name);
            forEachChildrenPrint(node.as<InterfaceDeclaration>()->typeParameters, "<", ", ", ">", true);
            forEachChildrenPrint(node.as<InterfaceDeclaration>()->heritageClauses);
            forEachChildrenPrint(node.as<InterfaceDeclaration>()->members);
            break;
        case SyntaxKind::TypeAliasDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<TypeAliasDeclaration>()->name);
            forEachChildrenPrint(node.as<TypeAliasDeclaration>()->typeParameters, "<", ", ", ">", true);
            forEachChildPrint(node.as<TypeAliasDeclaration>()->type);
            break;
        case SyntaxKind::EnumDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<EnumDeclaration>()->name);
            forEachChildrenPrint(node.as<EnumDeclaration>()->members);
            break;
        case SyntaxKind::EnumMember:
            forEachChildPrint(node.as<EnumMember>()->name);
            forEachChildPrint(node.as<EnumMember>()->initializer);
            break;
        case SyntaxKind::ModuleDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<ModuleDeclaration>()->name);
            forEachChildPrint(node.as<ModuleDeclaration>()->body);
            break;
        case SyntaxKind::ImportEqualsDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<ImportEqualsDeclaration>()->name);
            forEachChildPrint(node.as<ImportEqualsDeclaration>()->moduleReference);
            break;
        case SyntaxKind::ImportDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<ImportDeclaration>()->importClause);
            forEachChildPrint(node.as<ImportDeclaration>()->moduleSpecifier);
            break;
        case SyntaxKind::ImportClause:
            forEachChildPrint(node.as<ImportClause>()->name);
            forEachChildPrint(node.as<ImportClause>()->namedBindings);
            break;
        case SyntaxKind::NamespaceExportDeclaration:
            forEachChildPrint(node.as<NamespaceExportDeclaration>()->name);
            break;

        case SyntaxKind::NamespaceImport:
            forEachChildPrint(node.as<NamespaceImport>()->name);
            break;
        case SyntaxKind::NamespaceExport:
            forEachChildPrint(node.as<NamespaceExport>()->name);
            break;
        case SyntaxKind::NamedImports:
            forEachChildrenPrint(node.as<NamedImports>()->elements);
            break;
        case SyntaxKind::NamedExports:
            forEachChildrenPrint(node.as<NamedExports>()->elements);
            break;
        case SyntaxKind::ExportDeclaration:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<ExportDeclaration>()->exportClause);
            forEachChildPrint(node.as<ExportDeclaration>()->moduleSpecifier);
            break;
        case SyntaxKind::ImportSpecifier:
            forEachChildPrint(node.as<ImportSpecifier>()->propertyName);
            forEachChildPrint(node.as<ImportSpecifier>()->name);
            break;
        case SyntaxKind::ExportSpecifier:
            forEachChildPrint(node.as<ExportSpecifier>()->propertyName);
            forEachChildPrint(node.as<ExportSpecifier>()->name);
            break;
        case SyntaxKind::ExportAssignment:
            forEachChildrenPrint(node->decorators);
            forEachChildrenPrint(node->modifiers);
            forEachChildPrint(node.as<ExportAssignment>()->expression);
            break;
        case SyntaxKind::TemplateExpression:
            forEachChildPrint(node.as<TemplateExpression>()->head);
            forEachChildrenPrint(node.as<TemplateExpression>()->templateSpans);
            break;
        case SyntaxKind::TemplateSpan:
            forEachChildPrint(node.as<TemplateSpan>()->expression);
            forEachChildPrint(node.as<TemplateSpan>()->literal);
            break;
        case SyntaxKind::TemplateLiteralType:
            forEachChildPrint(node.as<TemplateLiteralTypeNode>()->head);
            forEachChildrenPrint(node.as<TemplateLiteralTypeNode>()->templateSpans);
            break;
        case SyntaxKind::TemplateLiteralTypeSpan:
            forEachChildPrint(node.as<TemplateLiteralTypeSpan>()->type);
            forEachChildPrint(node.as<TemplateLiteralTypeSpan>()->literal);
            break;
        case SyntaxKind::ComputedPropertyName:
            forEachChildPrint(node.as<ComputedPropertyName>()->expression);
            break;
        case SyntaxKind::HeritageClause:
            forEachChildrenPrint(node.as<HeritageClause>()->types);
            break;
        case SyntaxKind::ExpressionWithTypeArguments:
            forEachChildPrint(node.as<ExpressionWithTypeArguments>()->expression);
            forEachChildrenPrint(node.as<ExpressionWithTypeArguments>()->typeArguments);
            break;
        case SyntaxKind::ExternalModuleReference:
            forEachChildPrint(node.as<ExternalModuleReference>()->expression);
            break;
        case SyntaxKind::MissingDeclaration:
            forEachChildrenPrint(node->decorators);
            break;
        case SyntaxKind::CommaListExpression:
            forEachChildrenPrint(node.as<CommaListExpression>()->elements);
            break;

        case SyntaxKind::JsxElement:
            forEachChildPrint(node.as<JsxElement>()->openingElement);
            forEachChildrenPrint(node.as<JsxElement>()->children);
            forEachChildPrint(node.as<JsxElement>()->closingElement);
            break;
        case SyntaxKind::JsxFragment:
            forEachChildPrint(node.as<JsxFragment>()->openingFragment);
            forEachChildrenPrint(node.as<JsxFragment>()->children);
            forEachChildPrint(node.as<JsxFragment>()->closingFragment);
            break;
        case SyntaxKind::JsxSelfClosingElement:
            forEachChildPrint(node.as<JsxSelfClosingElement>()->tagName);
            forEachChildrenPrint(node.as<JsxSelfClosingElement>()->typeArguments);
            forEachChildPrint(node.as<JsxSelfClosingElement>()->attributes);
            break;
        case SyntaxKind::JsxOpeningElement:
            forEachChildPrint(node.as<JsxOpeningElement>()->tagName);
            forEachChildrenPrint(node.as<JsxOpeningElement>()->typeArguments);
            forEachChildPrint(node.as<JsxOpeningElement>()->attributes);
            break;
        case SyntaxKind::JsxAttributes:
            forEachChildrenPrint(node.as<JsxAttributes>()->properties);
            break;
        case SyntaxKind::JsxAttribute:
            forEachChildPrint(node.as<JsxAttribute>()->name);
            forEachChildPrint(node.as<JsxAttribute>()->initializer);
            break;
        case SyntaxKind::JsxSpreadAttribute:
            forEachChildPrint(node.as<JsxSpreadAttribute>()->expression);
            break;
        case SyntaxKind::JsxExpression:
            forEachChildPrint(node.as<JsxExpression>()->dotDotDotToken);
            forEachChildPrint(node.as<JsxExpression>()->expression);
            break;
        case SyntaxKind::JsxClosingElement:
            forEachChildPrint(node.as<JsxClosingElement>()->tagName);
            break;

        case SyntaxKind::OptionalType:
            forEachChildPrint(node.as<OptionalTypeNode>()->type);
            break;
        case SyntaxKind::RestType:
            forEachChildPrint(node.as<RestTypeNode>()->type);
            break;
        case SyntaxKind::JSDocTypeExpression:
            forEachChildPrint(node.as<JSDocTypeExpression>()->type);
            break;
        case SyntaxKind::JSDocNonNullableType:
            forEachChildPrint(node.as<JSDocNonNullableType>()->type);
            break;
        case SyntaxKind::JSDocNullableType:
            forEachChildPrint(node.as<JSDocNullableType>()->type);
            break;
        case SyntaxKind::JSDocOptionalType:
            forEachChildPrint(node.as<JSDocOptionalType>()->type);
            break;
        case SyntaxKind::JSDocVariadicType:
            forEachChildPrint(node.as<JSDocVariadicType>()->type);
            break;
        case SyntaxKind::JSDocFunctionType:
            forEachChildrenPrint(node.as<JSDocFunctionType>()->parameters, "(", ", ", ")");
            forEachChildPrint(node.as<JSDocFunctionType>()->type);
            break;
        case SyntaxKind::JSDocComment:
            forEachChildrenPrint(node.as<JSDoc>()->tags);
            break;
        case SyntaxKind::JSDocSeeTag:
            forEachChildPrint(node.as<JSDocSeeTag>()->tagName);
            forEachChildPrint(node.as<JSDocSeeTag>()->name);
            break;
        case SyntaxKind::JSDocNameReference:
            forEachChildPrint(node.as<JSDocNameReference>()->name);
            break;
        case SyntaxKind::JSDocParameterTag:
        case SyntaxKind::JSDocPropertyTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            if (node.as<JSDocPropertyLikeTag>()->isNameFirst)
            {
                forEachChildPrint(node.as<JSDocPropertyLikeTag>()->name);
                forEachChildPrint(node.as<JSDocPropertyLikeTag>()->typeExpression);
            }
            else
            {
                forEachChildPrint(node.as<JSDocPropertyLikeTag>()->typeExpression);
                forEachChildPrint(node.as<JSDocPropertyLikeTag>()->name);
            }
            break;
        case SyntaxKind::JSDocAuthorTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            break;
        case SyntaxKind::JSDocImplementsTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocImplementsTag>()->_class);
            break;
        case SyntaxKind::JSDocAugmentsTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocAugmentsTag>()->_class);
            break;
        case SyntaxKind::JSDocTemplateTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocTemplateTag>()->constraint);
            forEachChildrenPrint(node.as<JSDocTemplateTag>()->typeParameters, "<", ", ", ">", true);
            break;
        case SyntaxKind::JSDocTypedefTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            if (node.as<JSDocTypedefTag>()->typeExpression &&
                node.as<JSDocTypedefTag>()->typeExpression == SyntaxKind::JSDocTypeExpression)
            {
                forEachChildPrint(node.as<JSDocTypedefTag>()->typeExpression);
                forEachChildPrint(node.as<JSDocTypedefTag>()->fullName);
            }
            else
            {
                forEachChildPrint(node.as<JSDocTypedefTag>()->fullName);
                forEachChildPrint(node.as<JSDocTypedefTag>()->typeExpression);
            }
            break;
        case SyntaxKind::JSDocCallbackTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocCallbackTag>()->fullName);
            forEachChildPrint(node.as<JSDocCallbackTag>()->typeExpression);
            break;
        case SyntaxKind::JSDocReturnTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocReturnTag>()->typeExpression);
            break;
        case SyntaxKind::JSDocTypeTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocTypeTag>()->typeExpression);
            break;
        case SyntaxKind::JSDocThisTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocThisTag>()->typeExpression);
            break;
        case SyntaxKind::JSDocEnumTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            forEachChildPrint(node.as<JSDocEnumTag>()->typeExpression);
            break;
        case SyntaxKind::JSDocSignature:            
            forEachChildrenPrint(node.as<JSDocSignature>()->typeParameters, "<", ", ", ">", true);            
            forEachChildrenPrint(node.as<JSDocSignature>()->parameters, "(", ", ", ")");
            forEachChildPrint(node.as<JSDocSignature>()->type);
            break;
        case SyntaxKind::JSDocTypeLiteral:            
            forEachChildrenPrint(node.as<JSDocTypeLiteral>()->jsDocPropertyTags);
            break;
        case SyntaxKind::JSDocTag:
        case SyntaxKind::JSDocClassTag:
        case SyntaxKind::JSDocPublicTag:
        case SyntaxKind::JSDocPrivateTag:
        case SyntaxKind::JSDocProtectedTag:
        case SyntaxKind::JSDocReadonlyTag:
            forEachChildPrint(node.as<JSDocTag>()->tagName);
            break;
        case SyntaxKind::PartiallyEmittedExpression:
            forEachChildPrint(node.as<PartiallyEmittedExpression>()->expression);
            break;
        default:
            out << "[MISSING " << Scanner::tokenToText[node->_kind] << "]";
        }
    }
};

void printNode(ts::Node node)
{
    Printer<std::wostream> printer(std::wcout);
    printer.printNode(node);
}

} // namespace ts

#endif // DUMP_H