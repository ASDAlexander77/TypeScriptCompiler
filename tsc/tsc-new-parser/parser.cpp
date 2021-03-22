#include "parser.h"
#include "nodeFactory.h"
#include "core.h"
#include "utilities.h"

namespace ts {

    namespace
    {
        template<typename T>
        auto visitNode(NodeFuncT<T> cbNode, Node node) -> T {
            return node ? cbNode(node) : T();
        }

        template<typename T>
        auto visitNodes(NodeFuncT<T> cbNode, NodeArrayFuncT<T> cbNodes, /*NodeArray*/Node nodes) -> T {
            if (nodes) {
                if (cbNodes) {
                    return cbNodes(nodes);
                }
                for (auto node : nodes) {
                    auto result = cbNode(node);
                    if (result) {
                        return result;
                    }
                }
            }
        }

        template<typename T>
        auto visitNodes(NodeFuncT<T> cbNode, NodeArrayFuncT<T> cbNodes, NodeArray<T> nodes) -> T {
            if (nodes) {
                if (cbNodes) {
                    return cbNodes(nodes);
                }
                for (auto node : nodes) {
                    auto result = cbNode(node);
                    if (result) {
                        return result;
                    }
                }
            }
        }    

        /*@internal*/
        auto isJSDocLikeText(safe_string text, number start) {
            return text[start + 1] == CharacterCodes::asterisk &&
                text[start + 2] == CharacterCodes::asterisk &&
                text[start + 3] != CharacterCodes::slash;
        }

        /**
         * Invokes a callback for each child of the given node. The 'cbNode' callback is invoked for all child nodes
         * stored in properties. If a 'cbNodes' callback is specified, it is invoked for embedded arrays; otherwise,
         * embedded arrays are flattened and the 'cbNode' callback is invoked for each element. If a callback returns
         * a truthy value, iteration stops and that value is returned. Otherwise, undefined is returned.
         *
         * @param node a given node to visit its children
         * @param cbNode a callback to be invoked for all child nodes
         * @param cbNodes a callback to be invoked for embedded array
         *
         * @remarks `forEachChild` must visit the children of a node in the order
         * that they appear in the source code. The language service depends on this property to locate nodes by position.
         */
        template<typename T>
        auto forEachChild(Node node, NodeFuncT<T> cbNode, NodeArrayFuncT<T> cbNodes = nullptr) -> T {
            if (!node || node.kind <= SyntaxKind::LastToken) {
                return T();
            }
            switch (node.kind) {
                case SyntaxKind::QualifiedName:
                    return visitNode(cbNode, node.as<QualifiedName>().left) ||
                        visitNode(cbNode, node.as<QualifiedName>().right);
                case SyntaxKind::TypeParameter:
                    return visitNode(cbNode, node.as<TypeParameterDeclaration>().name) ||
                        visitNode(cbNode, node.as<TypeParameterDeclaration>().constraint) ||
                        visitNode(cbNode, node.as<TypeParameterDeclaration>()._default) ||
                        visitNode(cbNode, node.as<TypeParameterDeclaration>().expression);
                case SyntaxKind::ShorthandPropertyAssignment:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<ShorthandPropertyAssignment>().name) ||
                        visitNode(cbNode, node.as<ShorthandPropertyAssignment>().questionToken) ||
                        visitNode(cbNode, node.as<ShorthandPropertyAssignment>().exclamationToken) ||
                        visitNode(cbNode, node.as<ShorthandPropertyAssignment>().equalsToken) ||
                        visitNode(cbNode, node.as<ShorthandPropertyAssignment>().objectAssignmentInitializer);
                case SyntaxKind::SpreadAssignment:
                    return visitNode(cbNode, node.as<SpreadAssignment>().expression);
                case SyntaxKind::Parameter:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<ParameterDeclaration>().dotDotDotToken) ||
                        visitNode(cbNode, node.as<ParameterDeclaration>().name) ||
                        visitNode(cbNode, node.as<ParameterDeclaration>().questionToken) ||
                        visitNode(cbNode, node.as<ParameterDeclaration>().type) ||
                        visitNode(cbNode, node.as<ParameterDeclaration>().initializer);
                case SyntaxKind::PropertyDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<PropertyDeclaration>().name) ||
                        visitNode(cbNode, node.as<PropertyDeclaration>().questionToken) ||
                        visitNode(cbNode, node.as<PropertyDeclaration>().exclamationToken) ||
                        visitNode(cbNode, node.as<PropertyDeclaration>().type) ||
                        visitNode(cbNode, node.as<PropertyDeclaration>().initializer);
                case SyntaxKind::PropertySignature:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<PropertySignature>().name) ||
                        visitNode(cbNode, node.as<PropertySignature>().questionToken) ||
                        visitNode(cbNode, node.as<PropertySignature>().type) ||
                        visitNode(cbNode, node.as<PropertySignature>().initializer);
                case SyntaxKind::PropertyAssignment:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<PropertyAssignment>().name) ||
                        visitNode(cbNode, node.as<PropertyAssignment>().questionToken) ||
                        visitNode(cbNode, node.as<PropertyAssignment>().initializer);
                case SyntaxKind::VariableDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<VariableDeclaration>().name) ||
                        visitNode(cbNode, node.as<VariableDeclaration>().exclamationToken) ||
                        visitNode(cbNode, node.as<VariableDeclaration>().type) ||
                        visitNode(cbNode, node.as<VariableDeclaration>().initializer);
                case SyntaxKind::BindingElement:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<BindingElement>().dotDotDotToken) ||
                        visitNode(cbNode, node.as<BindingElement>().propertyName) ||
                        visitNode(cbNode, node.as<BindingElement>().name) ||
                        visitNode(cbNode, node.as<BindingElement>().initializer);
                case SyntaxKind::FunctionType:
                case SyntaxKind::ConstructorType:
                case SyntaxKind::CallSignature:
                case SyntaxKind::ConstructSignature:
                case SyntaxKind::IndexSignature:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNodes(cbNode, cbNodes, node.as<SignatureDeclaration>().typeParameters) ||
                        visitNodes(cbNode, cbNodes, node.as<SignatureDeclaration>().parameters) ||
                        visitNode(cbNode, node.as<SignatureDeclaration>().type);
                case SyntaxKind::MethodDeclaration:
                case SyntaxKind::MethodSignature:
                case SyntaxKind::Constructor:
                case SyntaxKind::GetAccessor:
                case SyntaxKind::SetAccessor:
                case SyntaxKind::FunctionExpression:
                case SyntaxKind::FunctionDeclaration:
                case SyntaxKind::ArrowFunction:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<FunctionLikeDeclaration>().asteriskToken) ||
                        visitNode(cbNode, node.as<FunctionLikeDeclaration>().name) ||
                        visitNode(cbNode, node.as<FunctionLikeDeclaration>().questionToken) ||
                        visitNode(cbNode, node.as<FunctionLikeDeclaration>().exclamationToken) ||
                        visitNodes(cbNode, cbNodes, node.as<FunctionLikeDeclaration>().typeParameters) ||
                        visitNodes(cbNode, cbNodes, node.as<FunctionLikeDeclaration>().parameters) ||
                        visitNode(cbNode, node.as<FunctionLikeDeclaration>().type) ||
                        visitNode(cbNode, node.as<ArrowFunction>().equalsGreaterThanToken) ||
                        visitNode(cbNode, node.as<FunctionLikeDeclaration>().body);
                case SyntaxKind::TypeReference:
                    return visitNode(cbNode, node.as<TypeReferenceNode>().typeName) ||
                        visitNodes(cbNode, cbNodes, node.as<TypeReferenceNode>().typeArguments);
                case SyntaxKind::TypePredicate:
                    return visitNode(cbNode, node.as<TypePredicateNode>().assertsModifier) ||
                        visitNode(cbNode, node.as<TypePredicateNode>().parameterName) ||
                        visitNode(cbNode, node.as<TypePredicateNode>().type);
                case SyntaxKind::TypeQuery:
                    return visitNode(cbNode, node.as<TypeQueryNode>().exprName);
                case SyntaxKind::TypeLiteral:
                    return visitNodes(cbNode, cbNodes, node.as<TypeLiteralNode>().members);
                case SyntaxKind::ArrayType:
                    return visitNode(cbNode, node.as<ArrayTypeNode>().elementType);
                case SyntaxKind::TupleType:
                    return visitNodes(cbNode, cbNodes, node.as<TupleTypeNode>().elements);
                case SyntaxKind::UnionType:
                case SyntaxKind::IntersectionType:
                    return visitNodes(cbNode, cbNodes, node.as<UnionOrIntersectionTypeNode>().types);
                case SyntaxKind::ConditionalType:
                    return visitNode(cbNode, node.as<ConditionalTypeNode>().checkType) ||
                        visitNode(cbNode, node.as<ConditionalTypeNode>().extendsType) ||
                        visitNode(cbNode, node.as<ConditionalTypeNode>().trueType) ||
                        visitNode(cbNode, node.as<ConditionalTypeNode>().falseType);
                case SyntaxKind::InferType:
                    return visitNode(cbNode, node.as<InferTypeNode>().typeParameter);
                case SyntaxKind::ImportType:
                    return visitNode(cbNode, node.as<ImportTypeNode>().argument) ||
                        visitNode(cbNode, node.as<ImportTypeNode>().qualifier) ||
                        visitNodes(cbNode, cbNodes, node.as<ImportTypeNode>().typeArguments);
                case SyntaxKind::ParenthesizedType:
                    return visitNode(cbNode, node.as<ParenthesizedTypeNode>().type);
                case SyntaxKind::TypeOperator:
                    return visitNode(cbNode, node.as<TypeOperatorNode>().type);
                case SyntaxKind::IndexedAccessType:
                    return visitNode(cbNode, node.as<IndexedAccessTypeNode>().objectType) ||
                        visitNode(cbNode, node.as<IndexedAccessTypeNode>().indexType);
                case SyntaxKind::MappedType:
                    return visitNode(cbNode, node.as<MappedTypeNode>().readonlyToken) ||
                        visitNode(cbNode, node.as<MappedTypeNode>().typeParameter) ||
                        visitNode(cbNode, node.as<MappedTypeNode>().nameType) ||
                        visitNode(cbNode, node.as<MappedTypeNode>().questionToken) ||
                        visitNode(cbNode, node.as<MappedTypeNode>().type);
                case SyntaxKind::LiteralType:
                    return visitNode(cbNode, node.as<LiteralTypeNode>().literal);
                case SyntaxKind::NamedTupleMember:
                    return visitNode(cbNode, node.as<NamedTupleMember>().dotDotDotToken) ||
                        visitNode(cbNode, node.as<NamedTupleMember>().name) ||
                        visitNode(cbNode, node.as<NamedTupleMember>().questionToken) ||
                        visitNode(cbNode, node.as<NamedTupleMember>().type);
                case SyntaxKind::ObjectBindingPattern:
                case SyntaxKind::ArrayBindingPattern:
                    return visitNodes(cbNode, cbNodes, node.as<BindingPattern>().elements);
                case SyntaxKind::ArrayLiteralExpression:
                    return visitNodes(cbNode, cbNodes, node.as<ArrayLiteralExpression>().elements);
                case SyntaxKind::ObjectLiteralExpression:
                    return visitNodes(cbNode, cbNodes, node.as<ObjectLiteralExpression>().properties);
                case SyntaxKind::PropertyAccessExpression:
                    return visitNode(cbNode, node.as<PropertyAccessExpression>().expression) ||
                        visitNode(cbNode, node.as<PropertyAccessExpression>().questionDotToken) ||
                        visitNode(cbNode, node.as<PropertyAccessExpression>().name);
                case SyntaxKind::ElementAccessExpression:
                    return visitNode(cbNode, node.as<ElementAccessExpression>().expression) ||
                        visitNode(cbNode, node.as<ElementAccessExpression>().questionDotToken) ||
                        visitNode(cbNode, node.as<ElementAccessExpression>().argumentExpression);
                case SyntaxKind::CallExpression:
                case SyntaxKind::NewExpression:
                    return visitNode(cbNode, node.as<CallExpression>().expression) ||
                        visitNode(cbNode, node.as<CallExpression>().questionDotToken) ||
                        visitNodes(cbNode, cbNodes, node.as<CallExpression>().typeArguments) ||
                        visitNodes(cbNode, cbNodes, node.as<CallExpression>().arguments);
                case SyntaxKind::TaggedTemplateExpression:
                    return visitNode(cbNode, node.as<TaggedTemplateExpression>().tag) ||
                        visitNode(cbNode, node.as<TaggedTemplateExpression>().questionDotToken) ||
                        visitNodes(cbNode, cbNodes, node.as<TaggedTemplateExpression>().typeArguments) ||
                        visitNode(cbNode, node.as<TaggedTemplateExpression>()._template);
                case SyntaxKind::TypeAssertionExpression:
                    return visitNode(cbNode, node.as<TypeAssertion>().type) ||
                        visitNode(cbNode, node.as<TypeAssertion>().expression);
                case SyntaxKind::ParenthesizedExpression:
                    return visitNode(cbNode, node.as<ParenthesizedExpression>().expression);
                case SyntaxKind::DeleteExpression:
                    return visitNode(cbNode, node.as<DeleteExpression>().expression);
                case SyntaxKind::TypeOfExpression:
                    return visitNode(cbNode, node.as<TypeOfExpression>().expression);
                case SyntaxKind::VoidExpression:
                    return visitNode(cbNode, node.as<VoidExpression>().expression);
                case SyntaxKind::PrefixUnaryExpression:
                    return visitNode(cbNode, node.as<PrefixUnaryExpression>().operand);
                case SyntaxKind::YieldExpression:
                    return visitNode(cbNode, node.as<YieldExpression>().asteriskToken) ||
                        visitNode(cbNode, node.as<YieldExpression>().expression);
                case SyntaxKind::AwaitExpression:
                    return visitNode(cbNode, node.as<AwaitExpression>().expression);
                case SyntaxKind::PostfixUnaryExpression:
                    return visitNode(cbNode, node.as<PostfixUnaryExpression>().operand);
                case SyntaxKind::BinaryExpression:
                    return visitNode(cbNode, node.as<BinaryExpression>().left) ||
                        visitNode(cbNode, node.as<BinaryExpression>().operatorToken) ||
                        visitNode(cbNode, node.as<BinaryExpression>().right);
                case SyntaxKind::AsExpression:
                    return visitNode(cbNode, node.as<AsExpression>().expression) ||
                        visitNode(cbNode, node.as<AsExpression>().type);
                case SyntaxKind::NonNullExpression:
                    return visitNode(cbNode, node.as<NonNullExpression>().expression);
                case SyntaxKind::MetaProperty:
                    return visitNode(cbNode, node.as<MetaProperty>().name);
                case SyntaxKind::ConditionalExpression:
                    return visitNode(cbNode, node.as<ConditionalExpression>().condition) ||
                        visitNode(cbNode, node.as<ConditionalExpression>().questionToken) ||
                        visitNode(cbNode, node.as<ConditionalExpression>().whenTrue) ||
                        visitNode(cbNode, node.as<ConditionalExpression>().colonToken) ||
                        visitNode(cbNode, node.as<ConditionalExpression>().whenFalse);
                case SyntaxKind::SpreadElement:
                    return visitNode(cbNode, node.as<SpreadElement>().expression);
                case SyntaxKind::Block:
                case SyntaxKind::ModuleBlock:
                    return visitNodes(cbNode, cbNodes, node.as<Block>().statements);
                case SyntaxKind::SourceFile:
                    return visitNodes(cbNode, cbNodes, node.as<SourceFile>().statements) ||
                        visitNode(cbNode, node.as<SourceFile>().endOfFileToken);
                case SyntaxKind::VariableStatement:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<VariableStatement>().declarationList);
                case SyntaxKind::VariableDeclarationList:
                    return visitNodes(cbNode, cbNodes, node.as<VariableDeclarationList>().declarations);
                case SyntaxKind::ExpressionStatement:
                    return visitNode(cbNode, node.as<ExpressionStatement>().expression);
                case SyntaxKind::IfStatement:
                    return visitNode(cbNode, node.as<IfStatement>().expression) ||
                        visitNode(cbNode, node.as<IfStatement>().thenStatement) ||
                        visitNode(cbNode, node.as<IfStatement>().elseStatement);
                case SyntaxKind::DoStatement:
                    return visitNode(cbNode, node.as<DoStatement>().statement) ||
                        visitNode(cbNode, node.as<DoStatement>().expression);
                case SyntaxKind::WhileStatement:
                    return visitNode(cbNode, node.as<WhileStatement>().expression) ||
                        visitNode(cbNode, node.as<WhileStatement>().statement);
                case SyntaxKind::ForStatement:
                    return visitNode(cbNode, node.as<ForStatement>().initializer) ||
                        visitNode(cbNode, node.as<ForStatement>().condition) ||
                        visitNode(cbNode, node.as<ForStatement>().incrementor) ||
                        visitNode(cbNode, node.as<ForStatement>().statement);
                case SyntaxKind::ForInStatement:
                    return visitNode(cbNode, node.as<ForInStatement>().initializer) ||
                        visitNode(cbNode, node.as<ForInStatement>().expression) ||
                        visitNode(cbNode, node.as<ForInStatement>().statement);
                case SyntaxKind::ForOfStatement:
                    return visitNode(cbNode, node.as<ForOfStatement>().awaitModifier) ||
                        visitNode(cbNode, node.as<ForOfStatement>().initializer) ||
                        visitNode(cbNode, node.as<ForOfStatement>().expression) ||
                        visitNode(cbNode, node.as<ForOfStatement>().statement);
                case SyntaxKind::ContinueStatement:
                case SyntaxKind::BreakStatement:
                    return visitNode(cbNode, node.as<BreakOrContinueStatement>().label);
                case SyntaxKind::ReturnStatement:
                    return visitNode(cbNode, node.as<ReturnStatement>().expression);
                case SyntaxKind::WithStatement:
                    return visitNode(cbNode, node.as<WithStatement>().expression) ||
                        visitNode(cbNode, node.as<WithStatement>().statement);
                case SyntaxKind::SwitchStatement:
                    return visitNode(cbNode, node.as<SwitchStatement>().expression) ||
                        visitNode(cbNode, node.as<SwitchStatement>().caseBlock);
                case SyntaxKind::CaseBlock:
                    return visitNodes(cbNode, cbNodes, node.as<CaseBlock>().clauses);
                case SyntaxKind::CaseClause:
                    return visitNode(cbNode, node.as<CaseClause>().expression) ||
                        visitNodes(cbNode, cbNodes, node.as<CaseClause>().statements);
                case SyntaxKind::DefaultClause:
                    return visitNodes(cbNode, cbNodes, node.as<DefaultClause>().statements);
                case SyntaxKind::LabeledStatement:
                    return visitNode(cbNode, node.as<LabeledStatement>().label) ||
                        visitNode(cbNode, node.as<LabeledStatement>().statement);
                case SyntaxKind::ThrowStatement:
                    return visitNode(cbNode, node.as<ThrowStatement>().expression);
                case SyntaxKind::TryStatement:
                    return visitNode(cbNode, node.as<TryStatement>().tryBlock) ||
                        visitNode(cbNode, node.as<TryStatement>().catchClause) ||
                        visitNode(cbNode, node.as<TryStatement>().finallyBlock);
                case SyntaxKind::CatchClause:
                    return visitNode(cbNode, node.as<CatchClause>().variableDeclaration) ||
                        visitNode(cbNode, node.as<CatchClause>().block);
                case SyntaxKind::Decorator:
                    return visitNode(cbNode, node.as<Decorator>().expression);
                case SyntaxKind::ClassDeclaration:
                case SyntaxKind::ClassExpression:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<ClassLikeDeclaration>().name) ||
                        visitNodes(cbNode, cbNodes, node.as<ClassLikeDeclaration>().typeParameters) ||
                        visitNodes(cbNode, cbNodes, node.as<ClassLikeDeclaration>().heritageClauses) ||
                        visitNodes(cbNode, cbNodes, node.as<ClassLikeDeclaration>().members);
                case SyntaxKind::InterfaceDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<InterfaceDeclaration>().name) ||
                        visitNodes(cbNode, cbNodes, node.as<InterfaceDeclaration>().typeParameters) ||
                        visitNodes(cbNode, cbNodes, node.as<ClassDeclaration>().heritageClauses) ||
                        visitNodes(cbNode, cbNodes, node.as<InterfaceDeclaration>().members);
                case SyntaxKind::TypeAliasDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<TypeAliasDeclaration>().name) ||
                        visitNodes(cbNode, cbNodes, node.as<TypeAliasDeclaration>().typeParameters) ||
                        visitNode(cbNode, node.as<TypeAliasDeclaration>().type);
                case SyntaxKind::EnumDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<EnumDeclaration>().name) ||
                        visitNodes(cbNode, cbNodes, node.as<EnumDeclaration>().members);
                case SyntaxKind::EnumMember:
                    return visitNode(cbNode, node.as<EnumMember>().name) ||
                        visitNode(cbNode, node.as<EnumMember>().initializer);
                case SyntaxKind::ModuleDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<ModuleDeclaration>().name) ||
                        visitNode(cbNode, node.as<ModuleDeclaration>().body);
                case SyntaxKind::ImportEqualsDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<ImportEqualsDeclaration>().name) ||
                        visitNode(cbNode, node.as<ImportEqualsDeclaration>().moduleReference);
                case SyntaxKind::ImportDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<ImportDeclaration>().importClause) ||
                        visitNode(cbNode, node.as<ImportDeclaration>().moduleSpecifier);
                case SyntaxKind::ImportClause:
                    return visitNode(cbNode, node.as<ImportClause>().name) ||
                        visitNode(cbNode, node.as<ImportClause>().namedBindings);
                case SyntaxKind::NamespaceExportDeclaration:
                    return visitNode(cbNode, node.as<NamespaceExportDeclaration>().name);

                case SyntaxKind::NamespaceImport:
                    return visitNode(cbNode, node.as<NamespaceImport>().name);
                case SyntaxKind::NamespaceExport:
                    return visitNode(cbNode, node.as<NamespaceExport>().name);
                case SyntaxKind::NamedImports:
                case SyntaxKind::NamedExports:
                    return visitNodes(cbNode, cbNodes, node.as<NamedImportsOrExports>().elements);
                case SyntaxKind::ExportDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<ExportDeclaration>().exportClause) ||
                        visitNode(cbNode, node.as<ExportDeclaration>().moduleSpecifier);
                case SyntaxKind::ImportSpecifier:
                case SyntaxKind::ExportSpecifier:
                    return visitNode(cbNode, node.as<ImportOrExportSpecifier>().propertyName) ||
                        visitNode(cbNode, node.as<ImportOrExportSpecifier>().name);
                case SyntaxKind::ExportAssignment:
                    return visitNodes(cbNode, cbNodes, node.decorators) ||
                        visitNodes(cbNode, cbNodes, node.modifiers) ||
                        visitNode(cbNode, node.as<ExportAssignment>().expression);
                case SyntaxKind::TemplateExpression:
                    return visitNode(cbNode, node.as<TemplateExpression>().head) || visitNodes(cbNode, cbNodes, node.as<TemplateExpression>().templateSpans);
                case SyntaxKind::TemplateSpan:
                    return visitNode(cbNode, node.as<TemplateSpan>().expression) || visitNode(cbNode, node.as<TemplateSpan>().literal);
                case SyntaxKind::TemplateLiteralType:
                    return visitNode(cbNode, node.as<TemplateLiteralTypeNode>().head) || visitNodes(cbNode, cbNodes, node.as<TemplateLiteralTypeNode>().templateSpans);
                case SyntaxKind::TemplateLiteralTypeSpan:
                    return visitNode(cbNode, node.as<TemplateLiteralTypeSpan>().type) || visitNode(cbNode, node.as<TemplateLiteralTypeSpan>().literal);
                case SyntaxKind::ComputedPropertyName:
                    return visitNode(cbNode, node.as<ComputedPropertyName>().expression);
                case SyntaxKind::HeritageClause:
                    return visitNodes(cbNode, cbNodes, node.as<HeritageClause>().types);
                case SyntaxKind::ExpressionWithTypeArguments:
                    return visitNode(cbNode, node.as<ExpressionWithTypeArguments>().expression) ||
                        visitNodes(cbNode, cbNodes, node.as<ExpressionWithTypeArguments>().typeArguments);
                case SyntaxKind::ExternalModuleReference:
                    return visitNode(cbNode, node.as<ExternalModuleReference>().expression);
                case SyntaxKind::MissingDeclaration:
                    return visitNodes(cbNode, cbNodes, node.decorators);
                case SyntaxKind::CommaListExpression:
                    return visitNodes(cbNode, cbNodes, node.as<CommaListExpression>().elements);

                case SyntaxKind::JsxElement:
                    return visitNode(cbNode, node.as<JsxElement>().openingElement) ||
                        visitNodes(cbNode, cbNodes, node.as<JsxElement>().children) ||
                        visitNode(cbNode, node.as<JsxElement>().closingElement);
                case SyntaxKind::JsxFragment:
                    return visitNode(cbNode, node.as<JsxFragment>().openingFragment) ||
                        visitNodes(cbNode, cbNodes, node.as<JsxFragment>().children) ||
                        visitNode(cbNode, node.as<JsxFragment>().closingFragment);
                case SyntaxKind::JsxSelfClosingElement:
                case SyntaxKind::JsxOpeningElement:
                    return visitNode(cbNode, node.as<JsxOpeningLikeElement>().tagName) ||
                        visitNodes(cbNode, cbNodes, node.as<JsxOpeningLikeElement>().typeArguments) ||
                        visitNode(cbNode, node.as<JsxOpeningLikeElement>().attributes);
                case SyntaxKind::JsxAttributes:
                    return visitNodes(cbNode, cbNodes, node.as<JsxAttributes>().properties);
                case SyntaxKind::JsxAttribute:
                    return visitNode(cbNode, node.as<JsxAttribute>().name) ||
                        visitNode(cbNode, node.as<JsxAttribute>().initializer);
                case SyntaxKind::JsxSpreadAttribute:
                    return visitNode(cbNode, node.as<JsxSpreadAttribute>().expression);
                case SyntaxKind::JsxExpression:
                    return visitNode(cbNode, node.as<JsxExpression>().dotDotDotToken) ||
                        visitNode(cbNode, node.as<JsxExpression>().expression);
                case SyntaxKind::JsxClosingElement:
                    return visitNode(cbNode, node.as<JsxClosingElement>().tagName);

                case SyntaxKind::OptionalType:
                    return visitNode(cbNode, node.as<OptionalTypeNode>().type);
                case SyntaxKind::RestType:
                    return visitNode(cbNode, node.as<RestTypeNode>().type);
                case SyntaxKind::JSDocTypeExpression:
                    return visitNode(cbNode, node.as<JSDocTypeExpression>().type);
                case SyntaxKind::JSDocNonNullableType:
                    return visitNode(cbNode, node.as<JSDocNonNullableTypeNode>().type);
                case SyntaxKind::JSDocNullableType:
                    return visitNode(cbNode, node.as<JSDocNullableTypeNode>().type);
                case SyntaxKind::JSDocOptionalType:
                    return visitNode(cbNode, node.as<JSDocOptionalTypeNode>().type);
                case SyntaxKind::JSDocVariadicType:
                    return visitNode(cbNode, node.as<JSDocVariadicTypeNode>().type);
                case SyntaxKind::JSDocFunctionType:
                    return visitNodes(cbNode, cbNodes, node.as<JSDocFunctionType>().parameters) ||
                        visitNode(cbNode, node.as<JSDocFunctionType>().type);
                case SyntaxKind::JSDocComment:
                    return visitNodes(cbNode, cbNodes, node.as<JSDoc>().tags);
                case SyntaxKind::JSDocSeeTag:
                    return visitNode(cbNode, node.as<JSDocSeeTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocSeeTag>().name);
                case SyntaxKind::JSDocNameReference:
                    return visitNode(cbNode, node.as<JSDocNameReference>().name);
                case SyntaxKind::JSDocParameterTag:
                case SyntaxKind::JSDocPropertyTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        (node.as<JSDocPropertyLikeTag>().isNameFirst
                            ? visitNode(cbNode, node.as<JSDocPropertyLikeTag>().name) ||
                                visitNode(cbNode, node.as<JSDocPropertyLikeTag>().typeExpression)
                            : visitNode(cbNode, node.as<JSDocPropertyLikeTag>().typeExpression) ||
                                visitNode(cbNode, node.as<JSDocPropertyLikeTag>().name));
                case SyntaxKind::JSDocAuthorTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName);
                case SyntaxKind::JSDocImplementsTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocImplementsTag>()._class);
                case SyntaxKind::JSDocAugmentsTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocAugmentsTag>()._class);
                case SyntaxKind::JSDocTemplateTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocTemplateTag>().constraint) ||
                        visitNodes(cbNode, cbNodes, node.as<JSDocTemplateTag>().typeParameters);
                case SyntaxKind::JSDocTypedefTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        (node.as<JSDocTypedefTag>().typeExpression &&
                            node.as<JSDocTypedefTag>().typeExpression.kind == SyntaxKind::JSDocTypeExpression
                            ? visitNode(cbNode, node.as<JSDocTypedefTag>().typeExpression) ||
                                visitNode(cbNode, node.as<JSDocTypedefTag>().fullName)
                            : visitNode(cbNode, node.as<JSDocTypedefTag>().fullName) ||
                                visitNode(cbNode, node.as<JSDocTypedefTag>().typeExpression));
                case SyntaxKind::JSDocCallbackTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocCallbackTag>().fullName) ||
                        visitNode(cbNode, node.as<JSDocCallbackTag>().typeExpression);
                case SyntaxKind::JSDocReturnTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocReturnTag>().typeExpression);
                case SyntaxKind::JSDocTypeTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocTypeTag>().typeExpression);
                case SyntaxKind::JSDocThisTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocThisTag>().typeExpression);
                case SyntaxKind::JSDocEnumTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName) ||
                        visitNode(cbNode, node.as<JSDocEnumTag>().typeExpression);
                case SyntaxKind::JSDocSignature:
                    return forEach(node.as<JSDocSignature>().typeParameters, cbNode) ||
                        forEach(node.as<JSDocSignature>().parameters, cbNode) ||
                        visitNode(cbNode, node.as<JSDocSignature>().type);
                case SyntaxKind::JSDocTypeLiteral:
                    return forEach(node.as<JSDocTypeLiteral>().jsDocPropertyTags, cbNode);
                case SyntaxKind::JSDocTag:
                case SyntaxKind::JSDocClassTag:
                case SyntaxKind::JSDocPublicTag:
                case SyntaxKind::JSDocPrivateTag:
                case SyntaxKind::JSDocProtectedTag:
                case SyntaxKind::JSDocReadonlyTag:
                    return visitNode(cbNode, node.as<JSDocTag>().tagName);
                case SyntaxKind::PartiallyEmittedExpression:
                    return visitNode(cbNode, node.as<PartiallyEmittedExpression>().expression);
            }
        }

        /** @internal */
        /**
         * Invokes a callback for each child of the given node. The 'cbNode' callback is invoked for all child nodes
         * stored in properties. If a 'cbNodes' callback is specified, it is invoked for embedded arrays; additionally,
         * unlike `forEachChild`, embedded arrays are flattened and the 'cbNode' callback is invoked for each element.
         *  If a callback returns a truthy value, iteration stops and that value is returned. Otherwise, undefined is returned.
         *
         * @param node a given node to visit its children
         * @param cbNode a callback to be invoked for all child nodes
         * @param cbNodes a callback to be invoked for embedded array
         *
         * @remarks Unlike `forEachChild`, `forEachChildRecursively` handles recursively invoking the traversal on each child node found,
         * and while doing so, handles traversing the structure without relying on the callstack to encode the tree structure.
         */
        template <typename T>
        auto forEachChildRecursively(Node rootNode, NodeWithParentFuncT<T> cbNode, NodeWithParentArrayFuncT<T> cbNodes = nullptr) -> T {
            auto queue = gatherPossibleChildren(rootNode);
            NodeArray parents; // tracks parent references for elements in queue
            while (parents.size() < queue.size()) {
                parents.push_back(rootNode);
            }
            while (queue.size() != 0) {
                auto current = queue.pop()!;
                auto parent = parents.pop()!;
                if (isArray(current)) {
                    if (cbNodes) {
                        auto res = cbNodes(current, parent);
                        if (res) {
                            if (res == "skip") continue;
                            return res;
                        }
                    }
                    for (int i = current.size() - 1; i >= 0; --i) {
                        queue.push_back(current[i]);
                        parents.push_back(parent);
                    }
                }
                else {
                    auto res = cbNode(current, parent);
                    if (res) {
                        if (res == "skip") continue;
                        return res;
                    }
                    if (current.kind >= SyntaxKind::FirstNode) {
                        // add children in reverse order to the queue, so popping gives the first child
                        for (auto child : gatherPossibleChildren(current)) {
                            queue.push_back(child);
                            parents.push_back(current);
                        }
                    }
                }
            }
        }

        auto gatherPossibleChildren(Node node) -> NodeArray<Node> {
            NodeArray<Node> children;

            auto addWorkItem = [&](auto n) -> Node {
                children.emplace(children.begin(), n);
                return Node();
            };

            forEachChild<Node>(node, addWorkItem, addWorkItem); // By using a stack above and `unshift` here, we emulate a depth-first preorder traversal
            return children;
        }

        /** @internal */
        auto isDeclarationFileName(string fileName) -> boolean {
            return fileExtensionIs(fileName, Extension::Dts);
        }

        // Implement the parser.as<a>() singleton module.  We do this for perf reasons because creating
        // parser instances can actually be expensive enough to impact us on projects with many source
        // files.
        class Parser {
            // Share a single scanner across all calls to parse a source file.  This helps speed things
            // up by avoiding the cost of creating/compiling scanners over and over again.
            Scanner scanner;

            NodeFlags disallowInAndDecoratorContext = NodeFlags::DisallowInContext | NodeFlags::DecoratorContext;

            NodeCreateFunc NodeConstructor;
            NodeCreateFunc TokenConstructor;
            NodeCreateFunc IdentifierConstructor;
            NodeCreateFunc PrivateIdentifierConstructor;
            NodeCreateFunc SourceFileConstructor;

            string fileName;
            NodeFlags sourceFlags;
            string sourceText;
            ScriptTarget languageVersion;
            ScriptKind scriptKind;
            LanguageVariant languageVariant;
            std::vector<DiagnosticWithDetachedLocation> parseDiagnostics;
            std::vector<DiagnosticWithDetachedLocation> jsDocDiagnostics;
            Undefined<IncrementalParser::SyntaxCursor> syntaxCursor;

            SyntaxKind currentToken;
            number nodeCount;
            std::map<string, string> identifiers;
            std::map<string, string> privateIdentifiers;
            number identifierCount;

            ParsingContext parsingContext;

            std::vector<number> notParenthesizedArrow;

            // Flags that dictate what parsing context we're in.  For example:
            // Whether or not we are in strict parsing mode.  All that changes in strict parsing mode is
            // that some tokens that would be considered identifiers may be considered keywords.
            //
            // When adding more parser context flags, consider which is the more common case that the
            // flag will be in.  This should be the 'false' state for that flag.  The reason for this is
            // that we don't store data in our nodes unless the value is in the *non-default* state.  So,
            // for example, more often than code 'allows-in' (or doesn't 'disallow-in').  We opt for
            // 'disallow-in' set to 'false'.  Otherwise, if we had 'allowsIn' set to 'true', then almost
            // all nodes would need extra state on them to store this info.
            //
            // Note: 'allowIn' and 'allowYield' track 1:1 with the [in] and [yield] concepts in the ES6
            // grammar specification.
            //
            // An important thing about these context concepts.  By default they are effectively inherited
            // while parsing through every grammar production.  i.e. if you don't change them, then when
            // you parse a sub-production, it will have the same context values.as<the>() parent production.
            // This is great most of the time.  After all, consider all the 'expression' grammar productions
            // and how nearly all of them pass along the 'in' and 'yield' context values:
            //
            // EqualityExpression[In, Yield] :
            //      RelationalExpression[?In, ?Yield]
            //      EqualityExpression[?In, ?Yield] == RelationalExpression[?In, ?Yield]
            //      EqualityExpression[?In, ?Yield] != RelationalExpression[?In, ?Yield]
            //      EqualityExpression[?In, ?Yield] == RelationalExpression[?In, ?Yield]
            //      EqualityExpression[?In, ?Yield] != RelationalExpression[?In, ?Yield]
            //
            // Where you have to be careful is then understanding what the points are in the grammar
            // where the values are *not* passed along.  For example:
            //
            // SingleNameBinding[Yield,GeneratorParameter]
            //      [+GeneratorParameter]BindingIdentifier[Yield] Initializer[In]opt
            //      [~GeneratorParameter]BindingIdentifier[?Yield]Initializer[In, ?Yield]opt
            //
            // Here this is saying that if the GeneratorParameter context flag is set, that we should
            // explicitly set the 'yield' context flag to false before calling into the BindingIdentifier
            // and we should explicitly unset the 'yield' context flag before calling into the Initializer.
            // production.  Conversely, if the GeneratorParameter context flag is not set, then we
            // should leave the 'yield' context flag alone.
            //
            // Getting this all correct is tricky and requires careful reading of the grammar to
            // understand when these values should be changed versus when they should be inherited.
            //
            // it Note should not be necessary to save/restore these flags during speculative/lookahead
            // parsing.  These context flags are naturally stored and restored through normal recursive
            // descent parsing and unwinding.
            NodeFlags contextFlags;

            // Indicates whether we are currently parsing top-level statements.
            boolean topLevel = true;

            // Whether or not we've had a parse error since creating the last AST node.  If we have
            // encountered an error, it will be stored on the next AST node we create.  Parse errors
            // can be broken down into three categories:
            //
            // 1) An error that occurred during scanning.  For example, an unterminated literal, or a
            //    character that was completely not understood.
            //
            // 2) A token was expected, but was not present.  This type of error is commonly produced
            //    by the 'parseExpected' function.
            //
            // 3) A token was present that no parsing auto was able to consume.  This type of error
            //    only occurs in the 'abortParsingListOrMoveToNextToken' auto when the parser
            //    decides to skip the token.
            //
            // In all of these cases, we want to mark the next node.as<having>() had an error before it.
            // With this mark, we can know in incremental settings if this node can be reused, or if
            // we have to reparse it.  If we don't keep this information around, we may just reuse the
            // node.  in that event we would then not produce the same errors.as<we>() did before, causing
            // significant confusion problems.
            //
            // it Note is necessary that this value be saved/restored during speculative/lookahead
            // parsing.  During lookahead parsing, we will often create a node.  That node will have
            // this value attached, and then this value will be set back to 'false'.  If we decide to
            // rewind, we must get back to the same value we had prior to the lookahead.
            //
            // any Note errors at the end of the file that do not precede a regular node, should get
            // attached to the EOF token.
            boolean parseErrorBeforeNextFinishedNode = false;

            // Rather than using `createBaseNodeFactory` here, we establish a `BaseNodeFactory` that closes over the
            // constructors above, which are reset each time `initializeState` is called.
            BaseNodeFactory baseNodeFactory;
            NodeFactory factory;

            // Share a single scanner across all calls to parse a source file.  This helps speed things
            // up by avoiding the cost of creating/compiling scanners over and over again.
            Parser() : 
                scanner(ScriptTarget::Latest, /*skipTrivia*/ true), 
                baseNodeFactory{ 
                    [&](SyntaxKind kind) { return countNode(SourceFileConstructor(kind, /*pos*/ 0, /*end*/ 0)); },
                    [&](SyntaxKind kind) { return countNode(IdentifierConstructor(kind, /*pos*/ 0, /*end*/ 0)); },
                    [&](SyntaxKind kind) { return countNode(PrivateIdentifierConstructor(kind, /*pos*/ 0, /*end*/ 0)); },
                    [&](SyntaxKind kind) { return countNode(TokenConstructor(kind, /*pos*/ 0, /*end*/ 0)); },
                    [&](SyntaxKind kind) { return countNode(NodeConstructor(kind, /*pos*/ 0, /*end*/ 0)); }
                },
                factory(NodeFactoryFlags::NoParenthesizerRules | NodeFactoryFlags::NoNodeConverters | NodeFactoryFlags::NoOriginalNode, baseNodeFactory)
            {
            }

            auto countNode(Node node) -> Node {
                nodeCount++;
                return node;
            }

            auto parseSourceFile(string fileName, string sourceText, ScriptTarget languageVersion, Undefined<IncrementalParser::SyntaxCursor> syntaxCursor, boolean setParentNodes = false, ScriptKind scriptKind = ScriptKind::Unknown) -> SourceFile {
                scriptKind = ensureScriptKind(fileName, scriptKind);
                if (scriptKind == ScriptKind::JSON) {
                    auto result = parseJsonText(fileName, sourceText, languageVersion, syntaxCursor, setParentNodes);
                    // TODO: review if we need it
                    //convertToObjectWorker(result, result.statements[0].expression, result.parseDiagnostics, /*returnValue*/ false, /*knownRootOptions*/ undefined, /*jsonConversionNotifier*/ undefined);
                    result.referencedFiles.clear();
                    result.typeReferenceDirectives.clear();
                    result.libReferenceDirectives.clear();
                    result.amdDependencies.clear();
                    result.hasNoDefaultLib = false;
                    result.pragmas.clear();
                    return result;
                }

                initializeState(fileName, sourceText, languageVersion, syntaxCursor, scriptKind);

                auto result = parseSourceFileWorker(languageVersion, setParentNodes, scriptKind);

                clearState();

                return result;
            }

            auto parseIsolatedEntityName(string content, ScriptTarget languageVersion) -> EntityName {
                // Choice of `isDeclarationFile` should be arbitrary
                initializeState(string(), content, languageVersion, undefined, ScriptKind::JS);
                // Prime the scanner.
                nextToken();
                auto entityName = parseEntityName(/*allowReservedWords*/ true);
                auto isInvalid = token() == SyntaxKind::EndOfFileToken && !parseDiagnostics.size();
                clearState();
                return isInvalid ? entityName : Node();
            }

            auto parseJsonText(string fileName, string sourceText, ScriptTarget languageVersion = ScriptTarget::ES2015, Undefined<IncrementalParser::SyntaxCursor> syntaxCursor = undefined, boolean setParentNodes = false) -> JsonSourceFile {
                initializeState(fileName, sourceText, languageVersion, syntaxCursor, ScriptKind::JSON);
                sourceFlags = contextFlags;

                // Prime the scanner.
                nextToken();
                auto pos = getNodePos();
                Node statements, endOfFileToken;
                if (token() == SyntaxKind::EndOfFileToken) {
                    statements = createNodeArray(Node(), pos, pos);
                    endOfFileToken = parseTokenNode<EndOfFileToken>();
                }
                else {
                    // Loop and synthesize an ArrayLiteralExpression if there are more than
                    // one top-level expressions to ensure all input text is consumed.
                    Node expressions;
                    while (token() != SyntaxKind::EndOfFileToken) {
                        Node expression;
                        switch (token()) {
                            case SyntaxKind::OpenBracketToken:
                                expression = parseArrayLiteralExpression();
                                break;
                            case SyntaxKind::TrueKeyword:
                            case SyntaxKind::FalseKeyword:
                            case SyntaxKind::NullKeyword:
                                expression = parseTokenNode<BooleanLiteral, NullLiteral>();
                                break;
                            case SyntaxKind::MinusToken:
                                if (lookAhead<boolean>([&]() { return nextToken() == SyntaxKind::NumericLiteral && nextToken() != SyntaxKind::ColonToken; })) {
                                    expression = parsePrefixUnaryExpression();
                                }
                                else {
                                    expression = parseObjectLiteralExpression();
                                }
                                break;
                            case SyntaxKind::NumericLiteral:
                            case SyntaxKind::StringLiteral:
                                if (lookAhead<boolean>([&]() { return nextToken() != SyntaxKind::ColonToken; })) {
                                    expression = parseLiteralNode();
                                    break;
                                }
                                // falls through
                            default:
                                expression = parseObjectLiteralExpression();
                                break;
                        }

                        // Error collect recovery multiple top-level expressions
                        if (expressions) {
                            expressions.push_back(expression);
                        }
                        else {
                            expressions = expression;
                            if (token() != SyntaxKind::EndOfFileToken) {
                                parseErrorAtCurrentToken(Diagnostics::Unexpected_token);
                            }
                        }
                    }

                    auto expression = isArray(expressions) ? finishNode(factory.createArrayLiteralExpression(expressions), pos) : Debug::checkDefined(expressions);
                    auto statement = factory.createExpressionStatement(expression);
                    finishNode(statement, pos);
                    statements = createNodeArray(statement, pos);
                    endOfFileToken = parseExpectedToken(SyntaxKind::EndOfFileToken, Diagnostics::Unexpected_token);
                }

                // Set source file so that errors will be reported with this file name
                auto sourceFile = createSourceFile(fileName, ScriptTarget::ES2015, ScriptKind::JSON, /*isDeclaration*/ false, statements, endOfFileToken, sourceFlags);

                if (setParentNodes) {
                    fixupParentReferences(sourceFile);
                }

                sourceFile.nodeCount = nodeCount;
                sourceFile.identifierCount = identifierCount;
                sourceFile.identifiers = identifiers;
                //sourceFile.parseDiagnostics = attachFileToDiagnostics(parseDiagnostics, sourceFile);
                sourceFile.parseDiagnostics = parseDiagnostics;
                if (!jsDocDiagnostics.empty()) {
                    //sourceFile.jsDocDiagnostics = attachFileToDiagnostics(jsDocDiagnostics, sourceFile);
                    sourceFile.jsDocDiagnostics = jsDocDiagnostics;
                }

                auto result = JsonSourceFile(sourceFile);
                clearState();
                return result;
            }

            auto initializeState(string _fileName, string _sourceText, ScriptTarget _languageVersion, Undefined<IncrementalParser::SyntaxCursor> _syntaxCursor, ScriptKind _scriptKind) -> void {
                NodeConstructor = [] (SyntaxKind kind, number start, number end) {
                    return Node(kind, start, end);
                };
                TokenConstructor = [] (SyntaxKind kind, number start, number end) {
                    return Node(kind, start, end);
                };
                IdentifierConstructor = [] (SyntaxKind kind, number start, number end) {
                    return Node(kind, start, end);
                };
                PrivateIdentifierConstructor = [] (SyntaxKind kind, number start, number end) {
                    return Node(kind, start, end);
                };
                SourceFileConstructor = [] (SyntaxKind kind, number start, number end) {
                    return Node(kind, start, end);
                };

                fileName = normalizePath(_fileName);
                sourceText = _sourceText;
                languageVersion = _languageVersion;
                syntaxCursor = _syntaxCursor;
                scriptKind = _scriptKind;
                languageVariant = getLanguageVariant(_scriptKind);

                parseDiagnostics.clear();
                parsingContext = ParsingContext::Unknown;
                identifiers.clear();
                privateIdentifiers.clear();
                identifierCount = 0;
                nodeCount = 0;
                sourceFlags = NodeFlags::None;
                topLevel = true;

                switch (scriptKind) {
                    case ScriptKind::JS:
                    case ScriptKind::JSX:
                        contextFlags = NodeFlags::JavaScriptFile;
                        break;
                    case ScriptKind::JSON:
                        contextFlags = NodeFlags::JavaScriptFile | NodeFlags::JsonFile;
                        break;
                    default:
                        contextFlags = NodeFlags::None;
                        break;
                }
                parseErrorBeforeNextFinishedNode = false;

                // Initialize and prime the scanner before parsing the source elements.
                scanner.setText(sourceText);
                scanner.setOnError(std::bind(&Parser::scanError, this, std::placeholders::_1, std::placeholders::_2));
                scanner.setScriptTarget(languageVersion);
                scanner.setLanguageVariant(languageVariant);
            }

            auto clearState() -> void {
                // Clear out the text the scanner is pointing at, so it doesn't keep anything alive unnecessarily.
                scanner.clearCommentDirectives();
                scanner.setText(string());
                scanner.setOnError(nullptr);

                // Clear any data.  We don't want to accidentally hold onto it for too long.
                sourceText = string();
                languageVersion = ScriptTarget::ES3;
                syntaxCursor = undefined;
                scriptKind = ScriptKind::Unknown;
                languageVariant = LanguageVariant::Standard;
                sourceFlags = NodeFlags::None;
                parseDiagnostics.clear();
                jsDocDiagnostics.clear();
                parsingContext = ParsingContext::Unknown;
                identifiers.clear();
                notParenthesizedArrow.clear();
                topLevel = true;
            }

            auto parseSourceFileWorker(ScriptTarget languageVersion, boolean setParentNodes, ScriptKind scriptKind) -> SourceFile {
                auto isDeclarationFile = isDeclarationFileName(fileName);
                if (isDeclarationFile) {
                    contextFlags |= NodeFlags::Ambient;
                }

                sourceFlags = contextFlags;

                // Prime the scanner.
                nextToken();

                auto statements = parseList<Statement>(ParsingContext::SourceElements, std::bind(&Parser::parseStatement, this));
                Debug::_assert(token() == SyntaxKind::EndOfFileToken);
                auto endOfFileToken = addJSDocComment(parseTokenNode<EndOfFileToken>());

                auto sourceFile = createSourceFile(fileName, languageVersion, scriptKind, isDeclarationFile, statements, endOfFileToken, sourceFlags);

                // A member of ReadonlyArray<T> isn't assignable to a member of T[] (and prevents a direct cast) - but this is where we set up those members so they can be in the future
                processCommentPragmas(sourceFile, sourceText);

                auto reportPragmaDiagnostic = [&](number pos, number end, DiagnosticMessage diagnostic) -> void {
                    parseDiagnostics.push_back(createDetachedDiagnostic(fileName, pos, end, diagnostic));
                };
                processPragmasIntoFields(sourceFile, reportPragmaDiagnostic);

                sourceFile.commentDirectives = scanner.getCommentDirectives();
                sourceFile.nodeCount = nodeCount;
                sourceFile.identifierCount = identifierCount;
                sourceFile.identifiers = identifiers;
                //sourceFile.parseDiagnostics = attachFileToDiagnostics(parseDiagnostics, sourceFile);
                sourceFile.parseDiagnostics = parseDiagnostics;
                if (!jsDocDiagnostics.empty()) {
                    //sourceFile.jsDocDiagnostics = attachFileToDiagnostics(jsDocDiagnostics, sourceFile);
                    sourceFile.jsDocDiagnostics = jsDocDiagnostics;
                }

                if (setParentNodes) {
                    fixupParentReferences(sourceFile);
                }

                return sourceFile;
            }

            template <typename T>
            auto withJSDoc(T node, boolean hasJSDoc) -> T {
                return hasJSDoc ? addJSDocComment(node) : node;
            }

            boolean hasDeprecatedTag = false;
            template <typename T>
            auto addJSDocComment(T node) -> T {
                Debug::_assert(!node.jsDoc); // Should only be called once per node
                auto jsDoc = mapDefined(getJSDocCommentRanges(node, sourceText), comment => JSDocParser::parseJSDocComment(node, comment.pos, comment.end - comment.pos));
                if (jsDoc.size()) node.jsDoc = jsDoc;
                if (hasDeprecatedTag) {
                    hasDeprecatedTag = false;
                    node->flags |= NodeFlags::Deprecated;
                }
                return node;
            }

            auto reparseTopLevelAwait(SourceFile sourceFile) -> SourceFile {
                auto savedSyntaxCursor = syntaxCursor;
                auto baseSyntaxCursor = IncrementalParser::createSyntaxCursor(sourceFile);

                auto containsPossibleTopLevelAwait = [](Node node) {
                    return !(node->flags & NodeFlags::AwaitContext)
                        && !!(node->transformFlags & TransformFlags::ContainsPossibleTopLevelAwait);
                };

                auto findNextStatementWithAwait = [&](Node statements, number start) {
                    for (auto i = start; i < statements.size(); i++) {
                        if (containsPossibleTopLevelAwait(statements[i])) {
                            return i;
                        }
                    }
                    return -1;
                };

                auto findNextStatementWithoutAwait = [&](Node statements, number start) {
                    for (auto i = start; i < statements.size(); i++) {
                        if (!containsPossibleTopLevelAwait(statements[i])) {
                            return i;
                        }
                    }
                    return -1;
                };

                auto currentNode = [&](number position) {
                    auto node = baseSyntaxCursor.currentNode(position);
                    if (topLevel && node && containsPossibleTopLevelAwait(node)) {
                        node.intersectsChange = true;
                    }
                    return node;
                };

                syntaxCursor = IncrementalParser::SyntaxCursor{ currentNode };

                Node statements;
                auto savedParseDiagnostics = parseDiagnostics;

                parseDiagnostics.clear();

                auto pos = 0;
                auto start = findNextStatementWithAwait(sourceFile.statements, 0);
                while (start != -1) {
                    // append all statements between pos and start
                    auto prevStatement = sourceFile.statements[pos];
                    auto nextStatement = sourceFile.statements[start];
                    addRange(statements, sourceFile.statements, pos, start);
                    pos = findNextStatementWithoutAwait(sourceFile.statements, start);

                    // append all diagnostics associated with the copied range
                    auto diagnosticStart = findIndex(savedParseDiagnostics, [&](auto diagnostic, number index) { return diagnostic.start >= prevStatement->pos; });
                    auto diagnosticEnd = diagnosticStart >= 0 ? findIndex(savedParseDiagnostics, [&](auto diagnostic, number index) { return diagnostic.start >= nextStatement->pos, diagnosticStart; }) : -1;
                    if (diagnosticStart >= 0) {
                        addRange(parseDiagnostics, savedParseDiagnostics, diagnosticStart, diagnosticEnd >= 0 ? diagnosticEnd : -1);
                    }

                    // reparse all statements between start and pos. We skip existing diagnostics for the same range and allow the parser to generate new ones.
                    speculationHelper<void>([&] () {
                        auto savedContextFlags = contextFlags;
                        contextFlags |= NodeFlags::AwaitContext;
                        scanner.setTextPos(nextStatement->pos);
                        nextToken();

                        while (token() != SyntaxKind::EndOfFileToken) {
                            auto startPos = scanner.getStartPos();
                            auto statement = parseListElement<Statement>(ParsingContext::SourceElements, std::bind(&Parser::parseStatement, this));
                            statements.push_back(statement);
                            if (startPos == scanner.getStartPos()) {
                                nextToken();
                            }

                            if (pos >= 0) {
                                auto nonAwaitStatement = sourceFile.statements[pos];
                                if (statement->end == nonAwaitStatement->pos) {
                                    // done reparsing this section
                                    break;
                                }
                                if (statement->end > nonAwaitStatement->pos) {
                                    // we ate into the next statement, so we must reparse it.
                                    pos = findNextStatementWithoutAwait(sourceFile.statements, pos + 1);
                                }
                            }
                        }

                        contextFlags = savedContextFlags;
                    }, SpeculationKind::Reparse);

                    // find the next statement containing an `await`
                    start = pos >= 0 ? findNextStatementWithAwait(sourceFile.statements, pos) : -1;
                }

                // append all statements between pos and the end of the list
                if (pos >= 0) {
                    auto prevStatement = sourceFile.statements[pos];
                    addRange(statements, sourceFile.statements, pos);

                    // append all diagnostics associated with the copied range
                    auto diagnosticStart = findIndex(savedParseDiagnostics, [&](auto diagnostic, number index) { return diagnostic.start >= prevStatement->pos; });
                    if (diagnosticStart >= 0) {
                        addRange(parseDiagnostics, savedParseDiagnostics, diagnosticStart);
                    }
                }

                syntaxCursor = savedSyntaxCursor;
                return factory.updateSourceFile(sourceFile, setTextRange(factory.createNodeArray(statements), sourceFile.statements));
            }

            auto fixupParentReferences(Node rootNode) -> void {
                // normally parent references are set during binding. However, for clients that only need
                // a syntax tree, and no semantic features, then the binding process is an unnecessary
                // overhead.  This functions allows us to set all the parents, without all the expense of
                // binding.
                setParentRecursive(rootNode, /*incremental*/ true);
            }

            auto createSourceFile(string fileName, ScriptTarget languageVersion, ScriptKind scriptKind, boolean isDeclarationFile, Node statements, Node endOfFileToken, NodeFlags flags) -> SourceFile {
                // code from createNode is inlined here so createNode won't have to deal with special case of creating source files
                // this is quite rare comparing to other nodes and createNode should be.as<fast>().as<possible>()
                auto sourceFile = factory.createSourceFile(statements, endOfFileToken, flags);
                setTextRangePosWidth(sourceFile, 0, sourceText.size());
                setExternalModuleIndicator(sourceFile);

                // If we parsed this.as<an>() external module, it may contain top-level await
                if (!isDeclarationFile && isExternalModule(sourceFile) && !!(sourceFile->transformFlags & TransformFlags::ContainsPossibleTopLevelAwait)) {
                    sourceFile = reparseTopLevelAwait(sourceFile);
                }

                sourceFile.text = sourceText;
                sourceFile.bindDiagnostics.clear();
                sourceFile.bindSuggestionDiagnostics.clear();
                sourceFile.languageVersion = languageVersion;
                sourceFile.fileName = fileName;
                sourceFile.languageVariant = getLanguageVariant(scriptKind);
                sourceFile.isDeclarationFile = isDeclarationFile;
                sourceFile.scriptKind = scriptKind;

                return sourceFile;
            }

            auto setContextFlag(boolean val, NodeFlags flag) {
                if (val) {
                    contextFlags |= flag;
                }
                else {
                    contextFlags &= ~flag;
                }
            }

            auto setDisallowInContext(boolean val) {
                setContextFlag(val, NodeFlags::DisallowInContext);
            }

            auto setYieldContext(boolean val) {
                setContextFlag(val, NodeFlags::YieldContext);
            }

            auto setDecoratorContext(boolean val) {
                setContextFlag(val, NodeFlags::DecoratorContext);
            }

            auto setAwaitContext(boolean val) {
                setContextFlag(val, NodeFlags::AwaitContext);
            }

            template<typename T>
            auto doOutsideOfContext(NodeFlags context, std::function<T()> func) -> T {
                // contextFlagsToClear will contain only the context flags that are
                // currently set that we need to temporarily clear
                // We don't just blindly reset to the previous flags to ensure
                // that we do not mutate cached flags for the incremental
                // parser (ThisNodeHasError, ThisNodeOrAnySubNodesHasError, and
                // HasAggregatedChildData).
                auto contextFlagsToClear = context & contextFlags;
                if (contextFlagsToClear) {
                    // clear the requested context flags
                    setContextFlag(/*val*/ false, contextFlagsToClear);
                    auto result = func();
                    // restore the context flags we just cleared
                    setContextFlag(/*val*/ true, contextFlagsToClear);
                    return result;
                }

                // no need to do anything special.as<we>() are not in any of the requested contexts
                return func();
            }

            template<typename T>
            auto doInsideOfContext(NodeFlags context, std::function<T()> func) -> T {
                // contextFlagsToSet will contain only the context flags that
                // are not currently set that we need to temporarily enable.
                // We don't just blindly reset to the previous flags to ensure
                // that we do not mutate cached flags for the incremental
                // parser (ThisNodeHasError, ThisNodeOrAnySubNodesHasError, and
                // HasAggregatedChildData).
                auto contextFlagsToSet = context & ~contextFlags;
                if (contextFlagsToSet) {
                    // set the requested context flags
                    setContextFlag(/*val*/ true, contextFlagsToSet);
                    auto result = func();
                    // reset the context flags we just set
                    setContextFlag(/*val*/ false, contextFlagsToSet);
                    return result;
                }

                // no need to do anything special.as<we>() are already in all of the requested contexts
                return func();
            }

            template<typename T>
            auto allowInAnd(std::function<T()> func) -> T {
                return doOutsideOfContext(NodeFlags::DisallowInContext, func);
            }

            template<typename T>
            auto disallowInAnd(std::function<T()> func) -> T {
                return doInsideOfContext(NodeFlags::DisallowInContext, func);
            }

            template<typename T>
            auto doInYieldContext(std::function<T()> func) -> T {
                return doInsideOfContext(NodeFlags::YieldContext, func);
            }

            template<typename T>
            auto doInDecoratorContext(std::function<T()> func) -> T {
                return doInsideOfContext(NodeFlags::DecoratorContext, func);
            }

            template<typename T>
            auto doInAwaitContext(std::function<T()> func) -> T {
                return doInsideOfContext(NodeFlags::AwaitContext, func);
            }

            template<typename T>
            auto doOutsideOfAwaitContext(std::function<T()> func) -> T {
                return doOutsideOfContext(NodeFlags::AwaitContext, func);
            }

            template<typename T>
            auto doInYieldAndAwaitContext(std::function<T()> func) -> T {
                return doInsideOfContext(NodeFlags::YieldContext | NodeFlags::AwaitContext, func);
            }
            
            template<typename T>
            auto doOutsideOfYieldAndAwaitContext(std::function<T()> func) -> T {
                return doOutsideOfContext(NodeFlags::YieldContext | NodeFlags::AwaitContext, func);
            }

            auto inContext(NodeFlags flags) {
                return (contextFlags & flags) != NodeFlags::None;
            }

            auto inYieldContext() {
                return inContext(NodeFlags::YieldContext);
            }

            auto inDisallowInContext() {
                return inContext(NodeFlags::DisallowInContext);
            }

            auto inDecoratorContext() {
                return inContext(NodeFlags::DecoratorContext);
            }

            auto inAwaitContext() {
                return inContext(NodeFlags::AwaitContext);
            }

            auto parseErrorAtCurrentToken(DiagnosticMessage message, string arg0 = string()) -> void {
                parseErrorAt(scanner.getTokenPos(), scanner.getTextPos(), message, arg0);
            }

            auto parseErrorAtPosition(number start, number length, DiagnosticMessage message, string arg0 = string()) -> void {
                // Don't report another error if it would just be at the same position.as<the>() last error.
                auto lastError = lastOrUndefined(parseDiagnostics);
                if (!lastError || start != lastError.start) {
                    parseDiagnostics.push_back(createDetachedDiagnostic(fileName, start, length, message, arg0));
                }

                // Mark that we've encountered an error.  We'll set an appropriate bit on the next
                // node we finish so that it can't be reused incrementally.
                parseErrorBeforeNextFinishedNode = true;
            }

            auto parseErrorAt(number start, number end, DiagnosticMessage message) -> void {
                parseErrorAtPosition(start, end - start, message);
            }

            template<typename T>
            auto parseErrorAt(number start, number end, DiagnosticMessage message, T arg0) -> void {
                parseErrorAtPosition(start, end - start, message, arg0);
            }

            template<typename T>
            auto parseErrorAtRange(TextRange range, DiagnosticMessage message, T arg0) -> void {
                parseErrorAt(range.pos, range.end, message, arg0);
            }

            auto scanError(DiagnosticMessage message, number length) -> void {
                parseErrorAtPosition(scanner.getTextPos(), length, message);
            }

            auto getNodePos() -> number {
                return scanner.getStartPos();
            }

            auto hasPrecedingJSDocComment() {
                return scanner.hasPrecedingJSDocComment();
            }

            // Use this auto to access the current token instead of reading the currentToken
            // variable. Since auto results aren't narrowed in control flow analysis, this ensures
            // that the type checker doesn't make wrong assumptions about the type of the current
            // token (e.g. a call to nextToken() changes the current token but the checker doesn't
            // reason about this side effect).  Mainstream VMs inline simple functions like this, so
            // there is no performance penalty.
            auto token() -> SyntaxKind {
                return currentToken;
            }

            auto nextTokenWithoutCheck() {
                return currentToken = scanner.scan();
            }

            template<typename T>
            auto nextTokenAnd(std::function<T()> func) -> T {
                nextToken();
                return func();
            }

            auto nextToken() -> SyntaxKind {
                // if the keyword had an escape
                if (isKeyword(currentToken) && (scanner.hasUnicodeEscape() || scanner.hasExtendedUnicodeEscape())) {
                    // issue a parse error for the escape
                    parseErrorAt(scanner.getTokenPos(), scanner.getTextPos(), Diagnostics::Keywords_cannot_contain_escape_characters);
                }
                return nextTokenWithoutCheck();
            }

            auto nextTokenJSDoc() -> SyntaxKind {
                return currentToken = scanner.scanJsDocToken();
            }

            auto reScanGreaterToken() -> SyntaxKind {
                return currentToken = scanner.reScanGreaterToken();
            }

            auto reScanSlashToken() -> SyntaxKind {
                return currentToken = scanner.reScanSlashToken();
            }

            auto reScanTemplateToken(boolean isTaggedTemplate) -> SyntaxKind {
                return currentToken = scanner.reScanTemplateToken(isTaggedTemplate);
            }

            auto reScanTemplateHeadOrNoSubstitutionTemplate() -> SyntaxKind {
                return currentToken = scanner.reScanTemplateHeadOrNoSubstitutionTemplate();
            }

            auto reScanLessThanToken() -> SyntaxKind {
                return currentToken = scanner.reScanLessThanToken();
            }

            auto scanJsxIdentifier() -> SyntaxKind {
                return currentToken = scanner.scanJsxIdentifier();
            }

            auto scanJsxText() -> SyntaxKind {
                return currentToken = scanner.scanJsxToken();
            }

            auto scanJsxAttributeValue() -> SyntaxKind {
                return currentToken = scanner.scanJsxAttributeValue();
            }

            template<typename T>
            auto speculationHelper(std::function<T()> callback, SpeculationKind speculationKind) -> T {
                // Keep track of the state we'll need to rollback to if lookahead fails (or if the
                // caller asked us to always reset our state).
                auto saveToken = currentToken;
                auto saveParseDiagnosticsLength = parseDiagnostics::size();
                auto saveParseErrorBeforeNextFinishedNode = parseErrorBeforeNextFinishedNode;

                // it Note is not actually necessary to save/restore the context flags here.  That's
                // because the saving/restoring of these flags happens naturally through the recursive
                // descent nature of our Parser::  However, we still store this here just so we can
                // assert that invariant holds.
                auto saveContextFlags = contextFlags;

                // If we're only looking ahead, then tell the scanner to only lookahead.as<well>().
                // Otherwise, if we're actually speculatively parsing, then tell the scanner to do the
                // same.
                auto result = speculationKind != SpeculationKind::TryParse
                    ? scanner.lookAhead(callback)
                    : scanner.tryScan(callback);

                Debug::_assert(saveContextFlags == contextFlags);

                // If our callback returned something 'falsy' or we're just looking ahead,
                // then unconditionally restore us to where we were.
                if (!result || speculationKind != SpeculationKind::TryParse) {
                    currentToken = saveToken;
                    if (speculationKind != SpeculationKind::Reparse) {
                        parseDiagnostics::size() = saveParseDiagnosticsLength;
                    }
                    parseErrorBeforeNextFinishedNode = saveParseErrorBeforeNextFinishedNode;
                }

                return result;
            }

            /** Invokes the provided callback then unconditionally restores the parser to the state it
             * was in immediately prior to invoking the callback.  The result of invoking the callback
             * is returned from this function.
             */
            template <typename T> 
            auto lookAhead(std::function<T()> callback) -> T {
                return speculationHelper<T>(callback, SpeculationKind::Lookahead);
            }

            /** Invokes the provided callback.  If the callback returns something falsy, then it restores
             * the parser to the state it was in immediately prior to invoking the callback.  If the
             * callback returns something truthy, then the parser state is not rolled back.  The result
             * of invoking the callback is returned from this function.
             */
            template <typename T> 
            auto tryParse(std::function<T()> callback) -> T {
                return speculationHelper<T>(callback, SpeculationKind::TryParse);
            }

            auto isBindingIdentifier() -> boolean {
                if (token() == SyntaxKind::Identifier) {
                    return true;
                }
                return token() > SyntaxKind::LastReservedWord;
            }

            // Ignore strict mode flag because we will report an error in type checker instead.
            auto isIdentifier() -> boolean {
                if (token() == SyntaxKind::Identifier) {
                    return true;
                }

                // If we have a 'yield' keyword, and we're in the [yield] context, then 'yield' is
                // considered a keyword and is not an identifier.
                if (token() == SyntaxKind::YieldKeyword && inYieldContext()) {
                    return false;
                }

                // If we have a 'await' keyword, and we're in the [Await] context, then 'await' is
                // considered a keyword and is not an identifier.
                if (token() == SyntaxKind::AwaitKeyword && inAwaitContext()) {
                    return false;
                }

                return token() > SyntaxKind::LastReservedWord;
            }

            auto parseExpected(SyntaxKind kind, DiagnosticMessage diagnosticMessage = DiagnosticMessage(), boolean shouldAdvance = true) -> boolean {
                if (token() == kind) {
                    if (shouldAdvance) {
                        nextToken();
                    }
                    return true;
                }

                // Report specific message if provided with one.  Otherwise, report generic fallback message.
                if (!!diagnosticMessage) {
                    parseErrorAtCurrentToken(diagnosticMessage);
                }
                else {
                    parseErrorAtCurrentToken(Diagnostics::_0_expected, scanner.tokenToString(kind));
                }
                return false;
            }

            auto parseExpectedJSDoc(SyntaxKind kind) {
                if (token() == kind) {
                    nextTokenJSDoc();
                    return true;
                }
                parseErrorAtCurrentToken(Diagnostics::_0_expected, scanner.tokenToString(kind));
                return false;
            }

            auto parseOptional(SyntaxKind t) -> boolean {
                if (token() == t) {
                    nextToken();
                    return true;
                }
                return false;
            }

            auto parseOptionalToken(SyntaxKind t) -> Node {
                if (token() == t) {
                    return parseTokenNode();
                }
                return Node();
            }

            auto parseOptionalTokenJSDoc(SyntaxKind t) -> Node {
                if (token() == t) {
                    return parseTokenNodeJSDoc();
                }
                return Node();
            }

            auto parseExpectedToken(SyntaxKind t, DiagnosticMessage diagnosticMessage) -> Node {
                return parseOptionalToken(t) ||
                    createMissingNode(t, /*reportAtCurrentPosition*/ false, diagnosticMessage);
            }            

            auto parseExpectedTokenJSDoc(SyntaxKind t) -> Node {
                return parseOptionalTokenJSDoc(t) ||
                    createMissingNode(t, /*reportAtCurrentPosition*/ false, Diagnostics::_0_expected, scanner.tokenToString(t));
            }

            template <typename ... T>
            auto parseTokenNode() -> Node {
                auto pos = getNodePos();
                auto kind = token();
                nextToken();
                return finishNode(factory.createToken(kind), pos);
            }

            auto parseTokenNodeJSDoc() -> Node {
                auto pos = getNodePos();
                auto kind = token();
                nextTokenJSDoc();
                return finishNode(factory.createToken(kind), pos);
            }

            auto canParseSemicolon() {
                // If there's a real semicolon, then we can always parse it out.
                if (token() == SyntaxKind::SemicolonToken) {
                    return true;
                }

                // We can parse out an optional semicolon in ASI cases in the following cases.
                return token() == SyntaxKind::CloseBraceToken || token() == SyntaxKind::EndOfFileToken || scanner.hasPrecedingLineBreak();
            }

            auto parseSemicolon() -> boolean {
                if (canParseSemicolon()) {
                    if (token() == SyntaxKind::SemicolonToken) {
                        // consume the semicolon if it was explicitly provided.
                        nextToken();
                    }

                    return true;
                }
                else {
                    return parseExpected(SyntaxKind::SemicolonToken);
                }
            }

            auto createNodeArray(Node elements, number pos, number end = -1, boolean hasTrailingComma = false) -> Node {
                auto array = factory.createNodeArray(elements, hasTrailingComma);
                setTextRangePosEnd(array, pos, end != -1 ? end : scanner.getStartPos());
                return array;
            }

            auto finishNode(Node node, number pos, number end = -1) -> Node {
                setTextRangePosEnd(node, pos, end != -1 ? end : scanner.getStartPos());
                if (!!contextFlags) {
                    node->flags |= contextFlags;
                }

                // Keep track on the node if we encountered an error while parsing it.  If we did, then
                // we cannot reuse the node incrementally.  Once we've marked this node, clear out the
                // flag so that we don't mark any subsequent nodes.
                if (parseErrorBeforeNextFinishedNode) {
                    parseErrorBeforeNextFinishedNode = false;
                    node->flags |= NodeFlags::ThisNodeHasError;
                }

                return node;
            }

            template <typename T = Node>
            auto createMissingNode(SyntaxKind kind, boolean reportAtCurrentPosition, DiagnosticMessage diagnosticMessage, string arg0 = string()) -> Node {
                if (reportAtCurrentPosition) {
                    parseErrorAtPosition(scanner.getStartPos(), 0, diagnosticMessage, arg0);
                }
                else if (!!diagnosticMessage) {
                    parseErrorAtCurrentToken(diagnosticMessage, arg0);
                }

                auto pos = getNodePos();
                auto result =
                    kind == SyntaxKind::Identifier ? factory.createIdentifier(string()) :
                    isTemplateLiteralKind(kind) ? factory.createTemplateLiteralLikeNode(kind, string(), string(), /*templateFlags*/ TokenFlags::None) :
                    kind == SyntaxKind::NumericLiteral ? factory.createNumericLiteral(string(), /*numericLiteralFlags*/ TokenFlags::None) :
                    kind == SyntaxKind::StringLiteral ? factory.createStringLiteral(string(), /*isSingleQuote*/ false) :
                    kind == SyntaxKind::MissingDeclaration ? factory.createMissingDeclaration() :
                    factory.createToken(kind);
                return finishNode(result, pos);
            }

            auto internIdentifier(string text) -> string {
                auto identifier = identifiers.at(text);
                if (!identifier.empty()) {
                    identifiers[text] = (identifier = text);
                }
                return identifier;
            }

            // An identifier that starts with two underscores has an extra underscore character prepended to it to avoid issues
            // with magic property names like '__proto__'. The 'identifiers' object is used to share a single string instance for
            // each identifier in order to reduce memory consumption.
            auto createIdentifier(boolean isIdentifier, DiagnosticMessage diagnosticMessage = DiagnosticMessage(), DiagnosticMessage privateIdentifierDiagnosticMessage = DiagnosticMessage()) -> Identifier {
                if (isIdentifier) {
                    identifierCount++;
                    auto pos = getNodePos();
                    // Store original token kind if it is not just an Identifier so we can report appropriate error later in type checker
                    auto originalKeywordKind = token();
                    auto text = internIdentifier(scanner.getTokenValue());
                    nextTokenWithoutCheck();
                    return finishNode(factory.createIdentifier(text, /*typeArguments*/ undefined, originalKeywordKind), pos);
                }

                if (token() == SyntaxKind::PrivateIdentifier) {
                    parseErrorAtCurrentToken(!!privateIdentifierDiagnosticMessage ? privateIdentifierDiagnosticMessage : Diagnostics::Private_identifiers_are_not_allowed_outside_class_bodies);
                    return createIdentifier(/*isIdentifier*/ true);
                }

                if (token() == SyntaxKind::Unknown && scanner.tryScan<boolean>([&]() { return scanner.reScanInvalidIdentifier() == SyntaxKind::Identifier; })) {
                    // Scanner has already recorded an 'Invalid character' error, so no need to add another from the Parser::
                    return createIdentifier(/*isIdentifier*/ true);
                }

                identifierCount++;
                // Only for end of file because the error gets reported incorrectly on embedded script tags.
                auto reportAtCurrentPosition = token() == SyntaxKind::EndOfFileToken;

                auto isReservedWord = scanner.isReservedWord();
                auto msgArg = scanner.getTokenText();

                auto defaultMessage = isReservedWord ?
                    Diagnostics::Identifier_expected_0_is_a_reserved_word_that_cannot_be_used_here :
                    Diagnostics::Identifier_expected;

                return createMissingNode<Identifier>(SyntaxKind::Identifier, reportAtCurrentPosition, !!diagnosticMessage ? diagnosticMessage : defaultMessage, msgArg);
            }

            auto parseBindingIdentifier(DiagnosticMessage privateIdentifierDiagnosticMessage) {
                return createIdentifier(isBindingIdentifier(), /*diagnosticMessage*/ DiagnosticMessage(), privateIdentifierDiagnosticMessage);
            }

            auto parseIdentifier(DiagnosticMessage diagnosticMessage = DiagnosticMessage(), DiagnosticMessage privateIdentifierDiagnosticMessage = DiagnosticMessage()) -> Identifier {
                return createIdentifier(isIdentifier(), diagnosticMessage, privateIdentifierDiagnosticMessage);
            }

            auto parseIdentifierName(DiagnosticMessage diagnosticMessage = DiagnosticMessage()) -> Identifier {
                return createIdentifier(scanner.tokenIsIdentifierOrKeyword(token()), diagnosticMessage);
            }

            auto isLiteralPropertyName() -> boolean {
                return scanner.tokenIsIdentifierOrKeyword(token()) ||
                    token() == SyntaxKind::StringLiteral ||
                    token() == SyntaxKind::NumericLiteral;
            }

            auto parsePropertyNameWorker(boolean allowComputedPropertyNames) -> Node {
                if (token() == SyntaxKind::StringLiteral || token() == SyntaxKind::NumericLiteral) {
                    auto node = parseLiteralNode();
                    node->text = internIdentifier(node->text);
                    return node;
                }
                if (allowComputedPropertyNames && token() == SyntaxKind::OpenBracketToken) {
                    return parseComputedPropertyName();
                }
                if (token() == SyntaxKind::PrivateIdentifier) {
                    return parsePrivateIdentifier();
                }
                return parseIdentifierName();
            }

            auto parsePropertyName() -> Node {
                return parsePropertyNameWorker(/*allowComputedPropertyNames*/ true);
            }

            auto parseComputedPropertyName() -> Node {
                // PropertyName [Yield]:
                //      LiteralPropertyName
                //      ComputedPropertyName[?Yield]
                auto pos = getNodePos();
                parseExpected(SyntaxKind::OpenBracketToken);
                // We parse any expression (including a comma expression). But the grammar
                // says that only an assignment expression is allowed, so the grammar checker
                // will error if it sees a comma expression.
                auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
                parseExpected(SyntaxKind::CloseBracketToken);
                return finishNode(factory.createComputedPropertyName(expression), pos);
            }

            auto internPrivateIdentifier(string text) -> string {
                auto privateIdentifier = privateIdentifiers.at(text);
                if (!privateIdentifier.empty()) {
                    privateIdentifiers[text] = (privateIdentifier = text);
                }
                return privateIdentifier;
            }

            auto parsePrivateIdentifier() -> Node {
                auto pos = getNodePos();
                auto node = factory.createPrivateIdentifier(internPrivateIdentifier(scanner.getTokenText()));
                nextToken();
                return finishNode(node, pos);
            }

            auto parseContextualModifier(SyntaxKind t) -> boolean {
                return token() == t && tryParse<boolean>(std::bind(&Parser::nextTokenCanFollowModifier, this));
            }

            auto nextTokenIsOnSameLineAndCanFollowModifier() {
                nextToken();
                if (scanner.hasPrecedingLineBreak()) {
                    return false;
                }
                return canFollowModifier();
            }

            auto nextTokenCanFollowModifier() -> boolean {
                switch (token()) {
                    case SyntaxKind::ConstKeyword:
                        // 'const' is only a modifier if followed by 'enum'.
                        return nextToken() == SyntaxKind::EnumKeyword;
                    case SyntaxKind::ExportKeyword:
                        nextToken();
                        if (token() == SyntaxKind::DefaultKeyword) {
                            return lookAhead<boolean>(std::bind(&Parser::nextTokenCanFollowDefaultKeyword, this));
                        }
                        if (token() == SyntaxKind::TypeKeyword) {
                            return lookAhead<boolean>(std::bind(&Parser::nextTokenCanFollowExportModifier, this));
                        }
                        return canFollowExportModifier();
                    case SyntaxKind::DefaultKeyword:
                        return nextTokenCanFollowDefaultKeyword();
                    case SyntaxKind::StaticKeyword:
                        return nextTokenIsOnSameLineAndCanFollowModifier();
                    case SyntaxKind::GetKeyword:
                    case SyntaxKind::SetKeyword:
                        nextToken();
                        return canFollowModifier();
                    default:
                        return nextTokenIsOnSameLineAndCanFollowModifier();
                }
            }

            auto canFollowExportModifier() -> boolean {
                return token() != SyntaxKind::AsteriskToken
                    && token() != SyntaxKind::AsKeyword
                    && token() != SyntaxKind::OpenBraceToken
                    && canFollowModifier();
            }

            auto nextTokenCanFollowExportModifier() -> boolean {
                nextToken();
                return canFollowExportModifier();
            }

            auto parseAnyContextualModifier() -> boolean {
                return isModifierKind(token()) && tryParse<boolean>(std::bind(&Parser::nextTokenCanFollowModifier, this));
            }

            auto canFollowModifier() -> boolean {
                return token() == SyntaxKind::OpenBracketToken
                    || token() == SyntaxKind::OpenBraceToken
                    || token() == SyntaxKind::AsteriskToken
                    || token() == SyntaxKind::DotDotDotToken
                    || isLiteralPropertyName();
            }

            auto nextTokenCanFollowDefaultKeyword() -> boolean {
                nextToken();
                return token() == SyntaxKind::ClassKeyword || token() == SyntaxKind::FunctionKeyword ||
                    token() == SyntaxKind::InterfaceKeyword ||
                    (token() == SyntaxKind::AbstractKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsClassKeywordOnSameLine, this))) ||
                    (token() == SyntaxKind::AsyncKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsFunctionKeywordOnSameLine, this)));
            }

            // True if positioned at the start of a list element
            auto isListElement(ParsingContext parsingContext, boolean inErrorRecovery) -> boolean {
                auto node = currentNode(parsingContext);
                if (node) {
                    return true;
                }

                switch (parsingContext) {
                    case ParsingContext::SourceElements:
                    case ParsingContext::BlockStatements:
                    case ParsingContext::SwitchClauseStatements:
                        // If we're in error recovery, then we don't want to treat ';'.as<an>() empty statement.
                        // The problem is that ';' can show up in far too many contexts, and if we see one
                        // and assume it's a statement, then we may bail out inappropriately from whatever
                        // we're parsing.  For example, if we have a semicolon in the middle of a class, then
                        // we really don't want to assume the class is over and we're on a statement in the
                        // outer module.  We just want to consume and move on.
                        return !(token() == SyntaxKind::SemicolonToken && inErrorRecovery) && isStartOfStatement();
                    case ParsingContext::SwitchClauses:
                        return token() == SyntaxKind::CaseKeyword || token() == SyntaxKind::DefaultKeyword;
                    case ParsingContext::TypeMembers:
                        return lookAhead<boolean>(std::bind(&Parser::isTypeMemberStart, this));
                    case ParsingContext::ClassMembers:
                        // We allow semicolons.as<class>() elements (as specified by ES6).as<long>().as<we>()'re
                        // not in error recovery.  If we're in error recovery, we don't want an errant
                        // semicolon to be treated.as<a>() class member (since they're almost always used
                        // for statements.
                        return lookAhead<boolean>(std::bind(&Parser::isClassMemberStart, this)) || (token() == SyntaxKind::SemicolonToken && !inErrorRecovery);
                    case ParsingContext::EnumMembers:
                        // Include open bracket computed properties. This technically also lets in indexers,
                        // which would be a candidate for improved error reporting.
                        return token() == SyntaxKind::OpenBracketToken || isLiteralPropertyName();
                    case ParsingContext::ObjectLiteralMembers:
                        switch (token()) {
                            case SyntaxKind::OpenBracketToken:
                            case SyntaxKind::AsteriskToken:
                            case SyntaxKind::DotDotDotToken:
                            case SyntaxKind::DotToken: // Not an object literal member, but don't want to close the object (see `tests/cases/fourslash/completionsDotInObjectLiteral.ts`)
                                return true;
                            default:
                                return isLiteralPropertyName();
                        }
                    case ParsingContext::RestProperties:
                        return isLiteralPropertyName();
                    case ParsingContext::ObjectBindingElements:
                        return token() == SyntaxKind::OpenBracketToken || token() == SyntaxKind::DotDotDotToken || isLiteralPropertyName();
                    case ParsingContext::HeritageClauseElement:
                        // If we see `{ ... }` then only consume it.as<an>() expression if it is followed by `,` or `{`
                        // That way we won't consume the body of a class in its heritage clause.
                        if (token() == SyntaxKind::OpenBraceToken) {
                            return lookAhead<boolean>(std::bind(&Parser::isValidHeritageClauseObjectLiteral, this));
                        }

                        if (!inErrorRecovery) {
                            return isStartOfLeftHandSideExpression() && !isHeritageClauseExtendsOrImplementsKeyword();
                        }
                        else {
                            // If we're in error recovery we tighten up what we're willing to match.
                            // That way we don't treat something like "this".as<a>() valid heritage clause
                            // element during recovery.
                            return isIdentifier() && !isHeritageClauseExtendsOrImplementsKeyword();
                        }
                    case ParsingContext::VariableDeclarations:
                        return isBindingIdentifierOrPrivateIdentifierOrPattern();
                    case ParsingContext::ArrayBindingElements:
                        return token() == SyntaxKind::CommaToken || token() == SyntaxKind::DotDotDotToken || isBindingIdentifierOrPrivateIdentifierOrPattern();
                    case ParsingContext::TypeParameters:
                        return isIdentifier();
                    case ParsingContext::ArrayLiteralMembers:
                        switch (token()) {
                            case SyntaxKind::CommaToken:
                            case SyntaxKind::DotToken: // Not an array literal member, but don't want to close the array (see `tests/cases/fourslash/completionsDotInArrayLiteralInObjectLiteral.ts`)
                                return true;
                        }
                        // falls through
                    case ParsingContext::ArgumentExpressions:
                        return token() == SyntaxKind::DotDotDotToken || isStartOfExpression();
                    case ParsingContext::Parameters:
                        return isStartOfParameter(/*isJSDocParameter*/ false);
                    case ParsingContext::JSDocParameters:
                        return isStartOfParameter(/*isJSDocParameter*/ true);
                    case ParsingContext::TypeArguments:
                    case ParsingContext::TupleElementTypes:
                        return token() == SyntaxKind::CommaToken || isStartOfType();
                    case ParsingContext::HeritageClauses:
                        return isHeritageClause();
                    case ParsingContext::ImportOrExportSpecifiers:
                        return scanner.tokenIsIdentifierOrKeyword(token());
                    case ParsingContext::JsxAttributes:
                        return scanner.tokenIsIdentifierOrKeyword(token()) || token() == SyntaxKind::OpenBraceToken;
                    case ParsingContext::JsxChildren:
                        return true;
                }

                return Debug::fail(S("Non-exhaustive case in 'isListElement'.")), false;
            }

            auto isValidHeritageClauseObjectLiteral() -> boolean {
                Debug::_assert(token() == SyntaxKind::OpenBraceToken);
                if (nextToken() == SyntaxKind::CloseBraceToken) {
                    // if we see "extends {}" then only treat the {}.as<what>() we're extending (and not
                    // the class body) if we have:
                    //
                    //      extends {} {
                    //      extends {},
                    //      extends {} extends
                    //      extends {} implements

                    auto next = nextToken();
                    return next == SyntaxKind::CommaToken || next == SyntaxKind::OpenBraceToken || next == SyntaxKind::ExtendsKeyword || next == SyntaxKind::ImplementsKeyword;
                }

                return true;
            }

            auto nextTokenIsIdentifier() -> boolean {
                nextToken();
                return isIdentifier();
            }

            auto nextTokenIsIdentifierOrKeyword() -> boolean {
                nextToken();
                return scanner.tokenIsIdentifierOrKeyword(token());
            }

            auto nextTokenIsIdentifierOrKeywordOrGreaterThan() -> boolean {
                nextToken();
                return scanner.tokenIsIdentifierOrKeywordOrGreaterThan(token());
            }

            auto isHeritageClauseExtendsOrImplementsKeyword() -> boolean {
                if (token() == SyntaxKind::ImplementsKeyword ||
                    token() == SyntaxKind::ExtendsKeyword) {

                    return lookAhead<boolean>(std::bind(&Parser::nextTokenIsStartOfExpression, this));
                }

                return false;
            }

            auto nextTokenIsStartOfExpression() -> boolean {
                nextToken();
                return isStartOfExpression();
            }

            auto nextTokenIsStartOfType() -> boolean {
                nextToken();
                return isStartOfType();
            }

            // True if positioned at a list terminator
            auto isListTerminator(ParsingContext kind) -> boolean {
                if (token() == SyntaxKind::EndOfFileToken) {
                    // Being at the end of the file ends all lists.
                    return true;
                }

                switch (kind) {
                    case ParsingContext::BlockStatements:
                    case ParsingContext::SwitchClauses:
                    case ParsingContext::TypeMembers:
                    case ParsingContext::ClassMembers:
                    case ParsingContext::EnumMembers:
                    case ParsingContext::ObjectLiteralMembers:
                    case ParsingContext::ObjectBindingElements:
                    case ParsingContext::ImportOrExportSpecifiers:
                        return token() == SyntaxKind::CloseBraceToken;
                    case ParsingContext::SwitchClauseStatements:
                        return token() == SyntaxKind::CloseBraceToken || token() == SyntaxKind::CaseKeyword || token() == SyntaxKind::DefaultKeyword;
                    case ParsingContext::HeritageClauseElement:
                        return token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::ExtendsKeyword || token() == SyntaxKind::ImplementsKeyword;
                    case ParsingContext::VariableDeclarations:
                        return isVariableDeclaratorListTerminator();
                    case ParsingContext::TypeParameters:
                        // Tokens other than '>' are here for better error recovery
                        return token() == SyntaxKind::GreaterThanToken || token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::ExtendsKeyword || token() == SyntaxKind::ImplementsKeyword;
                    case ParsingContext::ArgumentExpressions:
                        // Tokens other than ')' are here for better error recovery
                        return token() == SyntaxKind::CloseParenToken || token() == SyntaxKind::SemicolonToken;
                    case ParsingContext::ArrayLiteralMembers:
                    case ParsingContext::TupleElementTypes:
                    case ParsingContext::ArrayBindingElements:
                        return token() == SyntaxKind::CloseBracketToken;
                    case ParsingContext::JSDocParameters:
                    case ParsingContext::Parameters:
                    case ParsingContext::RestProperties:
                        // Tokens other than ')' and ']' (the latter for index signatures) are here for better error recovery
                        return token() == SyntaxKind::CloseParenToken || token() == SyntaxKind::CloseBracketToken /*|| token == SyntaxKind::OpenBraceToken*/;
                    case ParsingContext::TypeArguments:
                        // All other tokens should cause the type-argument to terminate except comma token
                        return token() != SyntaxKind::CommaToken;
                    case ParsingContext::HeritageClauses:
                        return token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::CloseBraceToken;
                    case ParsingContext::JsxAttributes:
                        return token() == SyntaxKind::GreaterThanToken || token() == SyntaxKind::SlashToken;
                    case ParsingContext::JsxChildren:
                        return token() == SyntaxKind::LessThanToken && lookAhead<boolean>(std::bind(&Parser::nextTokenIsSlash, this));
                    default:
                        return false;
                }
            }

            auto isVariableDeclaratorListTerminator() -> boolean {
                // If we can consume a semicolon (either explicitly, or with ASI), then consider us done
                // with parsing the list of variable declarators.
                if (canParseSemicolon()) {
                    return true;
                }

                // in the case where we're parsing the variable declarator of a 'for-in' statement, we
                // are done if we see an 'in' keyword in front of us. Same with for-of
                if (isInOrOfKeyword(token())) {
                    return true;
                }

                // ERROR RECOVERY TWEAK:
                // For better error recovery, if we see an '=>' then we just stop immediately.  We've got an
                // arrow auto here and it's going to be very unlikely that we'll resynchronize and get
                // another variable declaration.
                if (token() == SyntaxKind::EqualsGreaterThanToken) {
                    return true;
                }

                // Keep trying to parse out variable declarators.
                return false;
            }

            // True if positioned at element or terminator of the current list or any enclosing list
            auto isInSomeParsingContext() -> boolean {
                for (auto kind = (number)ParsingContext::Unknown; kind < (number)ParsingContext::Count; kind++) {
                    if (!!(parsingContext & (ParsingContext)(1 << kind))) {
                        if (isListElement((ParsingContext)kind, /*inErrorRecovery*/ true) || isListTerminator((ParsingContext)kind)) {
                            return true;
                        }
                    }
                }

                return false;
            }

            // Parses a list of elements
            template <typename T>
            auto parseList(ParsingContext kind, std::function<T()> parseElement) -> Node {
                auto saveParsingContext = parsingContext;
                parsingContext |= 1 << kind;
                Node list;
                auto listPos = getNodePos();

                while (!isListTerminator(kind)) {
                    if (isListElement(kind, /*inErrorRecovery*/ false)) {
                        auto element = parseListElement(kind, parseElement);
                        list.push_back(element);

                        continue;
                    }

                    if (abortParsingListOrMoveToNextToken(kind)) {
                        break;
                    }
                }

                parsingContext = saveParsingContext;
                return createNodeArray(list, listPos);
            }

            template <typename T> 
            auto parseListElement(ParsingContext parsingContext, std::function <T()> parseElement) -> T {
                auto node = currentNode(parsingContext);
                if (node) {
                    return consumeNode(node).as<T>().as<T>();
                }

                return parseElement();
            }

            auto currentNode(ParsingContext parsingContext) -> Node {
                // If we don't have a cursor or the parsing context isn't reusable, there's nothing to reuse.
                //
                // If there is an outstanding parse error that we've encountered, but not attached to
                // some node, then we cannot get a node from the old source tree.  This is because we
                // want to mark the next node we encounter.as<being>() unusable.
                //
                // This Note may be too conservative.  Perhaps we could reuse the node and set the bit
                // on it (or its leftmost child).as<having>() the error.  For now though, being conservative
                // is nice and likely won't ever affect perf.
                if (!syntaxCursor || !isReusableParsingContext(parsingContext) || parseErrorBeforeNextFinishedNode) {
                    return undefined;
                }

                auto node = ((IncrementalParser::SyntaxCursor &)syntaxCursor).currentNode(scanner.getStartPos());

                // Can't reuse a missing node.
                // Can't reuse a node that intersected the change range.
                // Can't reuse a node that contains a parse error.  This is necessary so that we
                // produce the same set of errors again.
                if (nodeIsMissing(node) || node.intersectsChange || containsParseError(node)) {
                    return undefined;
                }

                // We can only reuse a node if it was parsed under the same strict mode that we're
                // currently in.  i.e. if we originally parsed a node in non-strict mode, but then
                // the user added 'using strict' at the top of the file, then we can't use that node
                // again.as<the>() presence of strict mode may cause us to parse the tokens in the file
                // differently.
                //
                // we Note *can* reuse tokens when the strict mode changes.  That's because tokens
                // are unaffected by strict mode.  It's just the parser will decide what to do with it
                // differently depending on what mode it is in.
                //
                // This also applies to all our other context flags.as<well>().
                auto nodeContextFlags = node->flags & NodeFlags::ContextFlags;
                if (nodeContextFlags != contextFlags) {
                    return undefined;
                }

                // Ok, we have a node that looks like it could be reused.  Now verify that it is valid
                // in the current list parsing context that we're currently at.
                if (!canReuseNode(node, parsingContext)) {
                    return undefined;
                }

                if (node.as<JSDocContainer>().jsDocCache) {
                    // jsDocCache may include tags from parent nodes, which might have been modified.
                    node.as<JSDocContainer>().jsDocCache = undefined;
                }

                return node;
            }

            auto consumeNode(Node node) {
                // Move the scanner so it is after the node we just consumed.
                scanner.setTextPos(node->end);
                nextToken();
                return node;
            }

            auto isReusableParsingContext(ParsingContext parsingContext) -> boolean {
                switch (parsingContext) {
                    case ParsingContext::ClassMembers:
                    case ParsingContext::SwitchClauses:
                    case ParsingContext::SourceElements:
                    case ParsingContext::BlockStatements:
                    case ParsingContext::SwitchClauseStatements:
                    case ParsingContext::EnumMembers:
                    case ParsingContext::TypeMembers:
                    case ParsingContext::VariableDeclarations:
                    case ParsingContext::JSDocParameters:
                    case ParsingContext::Parameters:
                        return true;
                }
                return false;
            }

            auto canReuseNode(Node node, ParsingContext parsingContext) -> boolean {
                switch (parsingContext) {
                    case ParsingContext::ClassMembers:
                        return isReusableClassMember(node);

                    case ParsingContext::SwitchClauses:
                        return isReusableSwitchClause(node);

                    case ParsingContext::SourceElements:
                    case ParsingContext::BlockStatements:
                    case ParsingContext::SwitchClauseStatements:
                        return isReusableStatement(node);

                    case ParsingContext::EnumMembers:
                        return isReusableEnumMember(node);

                    case ParsingContext::TypeMembers:
                        return isReusableTypeMember(node);

                    case ParsingContext::VariableDeclarations:
                        return isReusableVariableDeclaration(node);

                    case ParsingContext::JSDocParameters:
                    case ParsingContext::Parameters:
                        return isReusableParameter(node);

                    // Any other lists we do not care about reusing nodes in.  But feel free to add if
                    // you can do so safely.  Danger areas involve nodes that may involve speculative
                    // parsing.  If speculative parsing is involved with the node, then the range the
                    // parser reached while looking ahead might be in the edited range (see the example
                    // in canReuseVariableDeclaratorNode for a good case of this).

                    // case ParsingContext::HeritageClauses:
                    // This would probably be safe to reuse.  There is no speculative parsing with
                    // heritage clauses.

                    // case ParsingContext::TypeParameters:
                    // This would probably be safe to reuse.  There is no speculative parsing with
                    // type parameters.  Note that that's because type *parameters* only occur in
                    // unambiguous *type* contexts.  While type *arguments* occur in very ambiguous
                    // *expression* contexts.

                    // case ParsingContext::TupleElementTypes:
                    // This would probably be safe to reuse.  There is no speculative parsing with
                    // tuple types.

                    // Technically, type argument list types are probably safe to reuse.  While
                    // speculative parsing is involved with them (since type argument lists are only
                    // produced from speculative parsing a <.as<a>() type argument list), we only have
                    // the types because speculative parsing succeeded.  Thus, the lookahead never
                    // went past the end of the list and rewound.
                    // case ParsingContext::TypeArguments:

                    // these Note are almost certainly not safe to ever reuse.  Expressions commonly
                    // need a large amount of lookahead, and we should not reuse them.as<they>() may
                    // have actually intersected the edit.
                    // case ParsingContext::ArgumentExpressions:

                    // This is not safe to reuse for the same reason.as<the>() 'AssignmentExpression'
                    // cases.  i.e. a property assignment may end with an expression, and thus might
                    // have lookahead far beyond it's old node.
                    // case ParsingContext::ObjectLiteralMembers:

                    // This is probably not safe to reuse.  There can be speculative parsing with
                    // type names in a heritage clause.  There can be generic names in the type
                    // name list, and there can be left hand side expressions (which can have type
                    // arguments.)
                    // case ParsingContext::HeritageClauseElement:

                    // Perhaps safe to reuse, but it's unlikely we'd see more than a dozen attributes
                    // on any given element. Same for children.
                    // case ParsingContext::JsxAttributes:
                    // case ParsingContext::JsxChildren:

                }

                return false;
            }

            auto isReusableClassMember(Node node) -> boolean {
                if (node) {
                    switch (node.kind) {
                        case SyntaxKind::Constructor:
                        case SyntaxKind::IndexSignature:
                        case SyntaxKind::GetAccessor:
                        case SyntaxKind::SetAccessor:
                        case SyntaxKind::PropertyDeclaration:
                        case SyntaxKind::SemicolonClassElement:
                            return true;
                        case SyntaxKind::MethodDeclaration:
                            // Method declarations are not necessarily reusable.  An object-literal
                            // may have a method calls "constructor(...)" and we must reparse that
                            // into an actual .ConstructorDeclaration.
                            auto methodDeclaration = node.as<MethodDeclaration>();
                            auto nameIsConstructor = methodDeclaration.name.kind == SyntaxKind::Identifier &&
                                methodDeclaration.name->originalKeywordKind == SyntaxKind::ConstructorKeyword;

                            return !nameIsConstructor;
                    }
                }

                return false;
            }

            auto isReusableSwitchClause(Node node) -> boolean  {
                if (node) {
                    switch (node.kind) {
                        case SyntaxKind::CaseClause:
                        case SyntaxKind::DefaultClause:
                            return true;
                    }
                }

                return false;
            }

            auto isReusableStatement(Node node) -> boolean  {
                if (node) {
                    switch (node.kind) {
                        case SyntaxKind::FunctionDeclaration:
                        case SyntaxKind::VariableStatement:
                        case SyntaxKind::Block:
                        case SyntaxKind::IfStatement:
                        case SyntaxKind::ExpressionStatement:
                        case SyntaxKind::ThrowStatement:
                        case SyntaxKind::ReturnStatement:
                        case SyntaxKind::SwitchStatement:
                        case SyntaxKind::BreakStatement:
                        case SyntaxKind::ContinueStatement:
                        case SyntaxKind::ForInStatement:
                        case SyntaxKind::ForOfStatement:
                        case SyntaxKind::ForStatement:
                        case SyntaxKind::WhileStatement:
                        case SyntaxKind::WithStatement:
                        case SyntaxKind::EmptyStatement:
                        case SyntaxKind::TryStatement:
                        case SyntaxKind::LabeledStatement:
                        case SyntaxKind::DoStatement:
                        case SyntaxKind::DebuggerStatement:
                        case SyntaxKind::ImportDeclaration:
                        case SyntaxKind::ImportEqualsDeclaration:
                        case SyntaxKind::ExportDeclaration:
                        case SyntaxKind::ExportAssignment:
                        case SyntaxKind::ModuleDeclaration:
                        case SyntaxKind::ClassDeclaration:
                        case SyntaxKind::InterfaceDeclaration:
                        case SyntaxKind::EnumDeclaration:
                        case SyntaxKind::TypeAliasDeclaration:
                            return true;
                    }
                }

                return false;
            }

            auto isReusableEnumMember(Node node) -> boolean  {
                return node.kind == SyntaxKind::EnumMember;
            }

            auto isReusableTypeMember(Node node) -> boolean  {
                if (node) {
                    switch (node.kind) {
                        case SyntaxKind::ConstructSignature:
                        case SyntaxKind::MethodSignature:
                        case SyntaxKind::IndexSignature:
                        case SyntaxKind::PropertySignature:
                        case SyntaxKind::CallSignature:
                            return true;
                    }
                }

                return false;
            }

            auto isReusableVariableDeclaration(Node node) -> boolean  {
                if (node.kind != SyntaxKind::VariableDeclaration) {
                    return false;
                }

                // Very subtle incremental parsing bug.  Consider the following code:
                //
                //      auto v = new List < A, B
                //
                // This is actually legal code.  It's a list of variable declarators "v = new List<A"
                // on one side and "B" on the other. If you then change that to:
                //
                //      auto v = new List < A, B >()
                //
                // then we have a problem.  "v = new List<A" doesn't intersect the change range, so we
                // start reparsing at "B" and we completely fail to handle this properly.
                //
                // In order to prevent this, we do not allow a variable declarator to be reused if it
                // has an initializer.
                auto variableDeclarator = node.as<VariableDeclaration>();
                return variableDeclarator.initializer == undefined;
            }

            auto isReusableParameter(Node node) -> boolean  {
                if (node.kind != SyntaxKind::Parameter) {
                    return false;
                }

                // See the comment in isReusableVariableDeclaration for why we do this.
                auto parameter = node.as<ParameterDeclaration>();
                return parameter.initializer == undefined;
            }

            // Returns true if we should abort parsing.
            auto abortParsingListOrMoveToNextToken(ParsingContext kind) -> boolean  {
                parsingContextErrors(kind);
                if (isInSomeParsingContext()) {
                    return true;
                }

                nextToken();
                return false;
            }

            auto parsingContextErrors(ParsingContext context) -> void {
                switch (context) {
                    case ParsingContext::SourceElements: return parseErrorAtCurrentToken(Diagnostics::Declaration_or_statement_expected);
                    case ParsingContext::BlockStatements: return parseErrorAtCurrentToken(Diagnostics::Declaration_or_statement_expected);
                    case ParsingContext::SwitchClauses: return parseErrorAtCurrentToken(Diagnostics::case_or_default_expected);
                    case ParsingContext::SwitchClauseStatements: return parseErrorAtCurrentToken(Diagnostics::Statement_expected);
                    case ParsingContext::RestProperties: // fallthrough
                    case ParsingContext::TypeMembers: return parseErrorAtCurrentToken(Diagnostics::Property_or_signature_expected);
                    case ParsingContext::ClassMembers: return parseErrorAtCurrentToken(Diagnostics::Unexpected_token_A_constructor_method_accessor_or_property_was_expected);
                    case ParsingContext::EnumMembers: return parseErrorAtCurrentToken(Diagnostics::Enum_member_expected);
                    case ParsingContext::HeritageClauseElement: return parseErrorAtCurrentToken(Diagnostics::Expression_expected);
                    case ParsingContext::VariableDeclarations:
                        return isKeyword(token())
                            ? parseErrorAtCurrentToken(Diagnostics::_0_is_not_allowed_as_a_variable_declaration_name, scanner.tokenToString(token()))
                            : parseErrorAtCurrentToken(Diagnostics::Variable_declaration_expected);
                    case ParsingContext::ObjectBindingElements: return parseErrorAtCurrentToken(Diagnostics::Property_destructuring_pattern_expected);
                    case ParsingContext::ArrayBindingElements: return parseErrorAtCurrentToken(Diagnostics::Array_element_destructuring_pattern_expected);
                    case ParsingContext::ArgumentExpressions: return parseErrorAtCurrentToken(Diagnostics::Argument_expression_expected);
                    case ParsingContext::ObjectLiteralMembers: return parseErrorAtCurrentToken(Diagnostics::Property_assignment_expected);
                    case ParsingContext::ArrayLiteralMembers: return parseErrorAtCurrentToken(Diagnostics::Expression_or_comma_expected);
                    case ParsingContext::JSDocParameters: return parseErrorAtCurrentToken(Diagnostics::Parameter_declaration_expected);
                    case ParsingContext::Parameters: return parseErrorAtCurrentToken(Diagnostics::Parameter_declaration_expected);
                    case ParsingContext::TypeParameters: return parseErrorAtCurrentToken(Diagnostics::Type_parameter_declaration_expected);
                    case ParsingContext::TypeArguments: return parseErrorAtCurrentToken(Diagnostics::Type_argument_expected);
                    case ParsingContext::TupleElementTypes: return parseErrorAtCurrentToken(Diagnostics::Type_expected);
                    case ParsingContext::HeritageClauses: return parseErrorAtCurrentToken(Diagnostics::Unexpected_token_expected);
                    case ParsingContext::ImportOrExportSpecifiers: return parseErrorAtCurrentToken(Diagnostics::Identifier_expected);
                    case ParsingContext::JsxAttributes: return parseErrorAtCurrentToken(Diagnostics::Identifier_expected);
                    case ParsingContext::JsxChildren: return parseErrorAtCurrentToken(Diagnostics::Identifier_expected);
                    return; // GH TODO#18217 `Debug::_assertNever default(context);`
                }
            }

            // Parses a comma-delimited list of elements
            template <typename T> 
            auto parseDelimitedList(ParsingContext kind, std::function <T()> parseElement, boolean considerSemicolonAsDelimiter) -> NodeArray<T> {
                auto saveParsingContext = parsingContext;
                parsingContext |= 1 << kind;
                auto list = [];
                auto listPos = getNodePos();

                auto commaStart = -1; // Meaning the previous token was not a comma
                while (true) {
                    if (isListElement(kind, /*inErrorRecovery*/ false)) {
                        auto startPos = scanner.getStartPos();
                        list.push_back(parseListElement(kind, parseElement));
                        commaStart = scanner.getTokenPos();

                        if (parseOptional(SyntaxKind::CommaToken)) {
                            // No need to check for a zero length node since we know we parsed a comma
                            continue;
                        }

                        commaStart = -1; // Back to the state where the last token was not a comma
                        if (isListTerminator(kind)) {
                            break;
                        }

                        // We didn't get a comma, and the list wasn't terminated, explicitly parse
                        // out a comma so we give a good error message.
                        parseExpected(SyntaxKind::CommaToken, getExpectedCommaDiagnostic(kind));

                        // If the token was a semicolon, and the caller allows that, then skip it and
                        // continue.  This ensures we get back on track and don't result in tons of
                        // parse errors.  For example, this can happen when people do things like use
                        // a semicolon to delimit object literal members.   we Note'll have already
                        // reported an error when we called parseExpected above.
                        if (considerSemicolonAsDelimiter && token() == SyntaxKind::SemicolonToken && !scanner.hasPrecedingLineBreak()) {
                            nextToken();
                        }
                        if (startPos == scanner.getStartPos()) {
                            // What we're parsing isn't actually remotely recognizable.as<a>() element and we've consumed no tokens whatsoever
                            // Consume a token to advance the parser in some way and avoid an infinite loop
                            // This can happen when we're speculatively parsing parenthesized expressions which we think may be arrow functions,
                            // or when a modifier keyword which is disallowed.as<a>() parameter name (ie, `static` in strict mode) is supplied
                            nextToken();
                        }
                        continue;
                    }

                    if (isListTerminator(kind)) {
                        break;
                    }

                    if (abortParsingListOrMoveToNextToken(kind)) {
                        break;
                    }
                }

                parsingContext = saveParsingContext;
                // Recording the trailing comma is deliberately done after the previous
                // loop, and not just if we see a list terminator. This is because the list
                // may have ended incorrectly, but it is still important to know if there
                // was a trailing comma.
                // Check if the last token was a comma.
                // Always preserve a trailing comma by marking it on the NodeArray
                return createNodeArray(list, listPos, /*end*/ undefined, commaStart >= 0);
            }

            auto getExpectedCommaDiagnostic(ParsingContext kind) -> DiagnosticMessage {
                return kind == ParsingContext::EnumMembers ? Diagnostics::An_enum_member_name_must_be_followed_by_a_or : undefined;
            }

            template <typename T> 
            using MissingList = NodeArray<T>;

            template <typename T> 
            auto createMissingList() -> MissingList<T> {
                auto list = createNodeArray<T>([], getNodePos()).as<MissingList>()<T>;
                list.isMissingList = true;
                return list;
            }

            auto isMissingList(NodeArray<Node> arr) -> boolean {
                return ((MissingList<Node>)arr).isMissingList;
            }

            template <typename T>
            auto parseBracketedList(ParsingContext kind, std::function <T()> parseElement, SyntaxKind open, SyntaxKind close) -> NodeArray<T> {
                if (parseExpected(open)) {
                    auto result = parseDelimitedList(kind, parseElement);
                    parseExpected(close);
                    return result;
                }

                return createMissingList<T>();
            }

            auto parseEntityName(boolean allowReservedWords, Undefined<DiagnosticMessage> diagnosticMessage = undefined) -> Node {
                auto pos = getNodePos();
                auto entity = allowReservedWords ? parseIdentifierName(diagnosticMessage) : parseIdentifier(diagnosticMessage);
                auto dotPos = getNodePos();
                while (parseOptional(SyntaxKind::DotToken)) {
                    if (token() == SyntaxKind::LessThanToken) {
                        // the entity is part of a JSDoc-style generic, so record the trailing dot for later error reporting
                        entity->jsdocDotPos = dotPos;
                        break;
                    }
                    dotPos = getNodePos();
                    entity = finishNode(
                        factory.createQualifiedName(
                            entity,
                            parseRightSideOfDot(allowReservedWords, /* allowPrivateIdentifiers */ false).as<Identifier>()
                        ),
                        pos
                    );
                }
                return entity;
            }

            auto createQualifiedName(EntityName entity, Identifier name) -> QualifiedName {
                return finishNode(factory.createQualifiedName(entity, name), entity->pos).as<QualifiedName>();
            }

            auto parseRightSideOfDot(boolean allowIdentifierNames, boolean allowPrivateIdentifiers) -> Node {
                // Technically a keyword is valid here.as<all>() identifiers and keywords are identifier names.
                // However, often we'll encounter this in error situations when the identifier or keyword
                // is actually starting another valid construct.
                //
                // So, we check for the following specific case:
                //
                //      name.
                //      identifierOrKeyword identifierNameOrKeyword
                //
                // the Note newlines are important here.  For example, if that above code
                // were rewritten into:
                //
                //      name.identifierOrKeyword
                //      identifierNameOrKeyword
                //
                // Then we would consider it valid.  That's because ASI would take effect and
                // the code would be implicitly: "name.identifierOrKeyword; identifierNameOrKeyword".
                // In the first case though, ASI will not take effect because there is not a
                // line terminator after the identifier or keyword.
                if (scanner.hasPrecedingLineBreak() && scanner.tokenIsIdentifierOrKeyword(token())) {
                    auto matchesPattern = lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOnSameLine, this));

                    if (matchesPattern) {
                        // Report that we need an identifier.  However, report it right after the dot,
                        // and not on the next token.  This is because the next token might actually
                        // be an identifier and the error would be quite confusing.
                        return createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ true, Diagnostics::Identifier_expected);
                    }
                }

                if (token() == SyntaxKind::PrivateIdentifier) {
                    auto node = parsePrivateIdentifier();
                    return allowPrivateIdentifiers ? node : createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ true, Diagnostics::Identifier_expected);
                }

                return allowIdentifierNames ? parseIdentifierName() : parseIdentifier();
            }

            auto parseTemplateSpans(boolean isTaggedTemplate) -> Node {
                auto pos = getNodePos();
                Node list;
                TemplateSpan node;
                do {
                    node = parseTemplateSpan(isTaggedTemplate);
                    list.push_back(node);
                }
                while (node.literal.kind == SyntaxKind::TemplateMiddle);
                return createNodeArray(list, pos);
            }

            auto parseTemplateExpression(boolean isTaggedTemplate) -> TemplateExpression {
                auto pos = getNodePos();
                return finishNode(
                    factory.createTemplateExpression(
                        parseTemplateHead(isTaggedTemplate),
                        parseTemplateSpans(isTaggedTemplate)
                    ),
                    pos
                );
            }

            auto parseTemplateType() -> TemplateLiteralTypeNode {
                auto pos = getNodePos();
                return finishNode(
                    factory.createTemplateLiteralType(
                        parseTemplateHead(/*isTaggedTemplate*/ false),
                        parseTemplateTypeSpans()
                    ),
                    pos
                );
            }

            auto parseTemplateTypeSpans() -> Node {
                auto pos = getNodePos();
                Node list;
                TemplateLiteralTypeSpan node;
                do {
                    node = parseTemplateTypeSpan();
                    list.push_back(node);
                }
                while (node.literal.kind == SyntaxKind::TemplateMiddle);
                return createNodeArray(list, pos);
            }

            auto parseTemplateTypeSpan() -> TemplateLiteralTypeSpan {
                auto pos = getNodePos();
                return finishNode(
                    factory.createTemplateLiteralTypeSpan(
                        parseType(),
                        parseLiteralOfTemplateSpan(/*isTaggedTemplate*/ false)
                    ),
                    pos
                );
            }

            auto parseLiteralOfTemplateSpan(boolean isTaggedTemplate) {
                if (token() == SyntaxKind::CloseBraceToken) {
                    reScanTemplateToken(isTaggedTemplate);
                    return parseTemplateMiddleOrTemplateTail();
                }
                else {
                    // TODO(rbuckton) -> Do we need to call `parseExpectedToken` or can we just call `createMissingNode` directly?
                    return <TemplateTail>parseExpectedToken(SyntaxKind::TemplateTail, Diagnostics::_0_expected, scanner.tokenToString(SyntaxKind::CloseBraceToken));
                }
            }

            auto parseTemplateSpan(boolean isTaggedTemplate) -> TemplateSpan {
                auto pos = getNodePos();
                return finishNode(
                    factory.createTemplateSpan(
                        allowInAnd(parseExpression),
                        parseLiteralOfTemplateSpan(isTaggedTemplate)
                    ),
                    pos
                );
            }

            auto parseLiteralNode() -> LiteralExpression {
                return parseLiteralLikeNode(token()).as<LiteralExpression>();
            }

            auto parseTemplateHead(boolean isTaggedTemplate) -> TemplateHead {
                if (isTaggedTemplate) {
                    reScanTemplateHeadOrNoSubstitutionTemplate();
                }
                auto fragment = parseLiteralLikeNode(token());
                Debug::_assert(fragment.kind == SyntaxKind::TemplateHead, "Template head has wrong token kind");
                return fragment.as<TemplateHead>();
            }

            auto parseTemplateMiddleOrTemplateTail() -> Node {
                auto fragment = parseLiteralLikeNode(token());
                Debug::_assert(fragment.kind == SyntaxKind::TemplateMiddle || fragment.kind == SyntaxKind::TemplateTail, "Template fragment has wrong token kind");
                return fragment;
            }

            auto getTemplateLiteralRawText(SyntaxKind kind) -> string {
                auto isLast = kind == SyntaxKind::NoSubstitutionTemplateLiteral || kind == SyntaxKind::TemplateTail;
                auto tokenText = scanner.getTokenText();
                return tokenText.substring(1, tokenText.size() - (scanner.isUnterminated() ? 0 : isLast ? 1 : 2));
            }

            auto parseLiteralLikeNode(SyntaxKind kind) -> LiteralLikeNode {
                auto pos = getNodePos();
                auto node =
                    isTemplateLiteralKind(kind) ? factory.createTemplateLiteralLikeNode(kind, scanner.getTokenValue(), getTemplateLiteralRawText(kind), scanner.getTokenFlags() & TokenFlags.TemplateLiteralLikeFlags) :
                    // Octal literals are not allowed in strict mode or ES5
                    // Note that theoretically the following condition would hold true literals like 009,
                    // which is not octal. But because of how the scanner separates the tokens, we would
                    // never get a token like this. Instead, we would get 00 and 9.as<two>() separate tokens.
                    // We also do not need to check for negatives because any prefix operator would be part of a
                    // parent unary expression.
                    kind == SyntaxKind::NumericLiteral ? factory.createNumericLiteral(scanner.getTokenValue(), scanner.getNumericLiteralFlags()) :
                    kind == SyntaxKind::StringLiteral ? factory.createStringLiteral(scanner.getTokenValue(), /*isSingleQuote*/ undefined, scanner.hasExtendedUnicodeEscape()) :
                    isLiteralKind(kind) ? factory.createLiteralLikeNode(kind, scanner.getTokenValue()) :
                    Debug::fail();

                if (scanner.hasExtendedUnicodeEscape()) {
                    node.hasExtendedUnicodeEscape = true;
                }

                if (scanner.isUnterminated()) {
                    node.isUnterminated = true;
                }

                nextToken();
                return finishNode(node, pos);
            }

            // TYPES

            auto parseEntityNameOfTypeReference() {
                return parseEntityName(/*allowReservedWords*/ true, Diagnostics::Type_expected);
            }

            auto parseTypeArgumentsOfTypeReference() {
                if (!scanner.hasPrecedingLineBreak() && reScanLessThanToken() == SyntaxKind::LessThanToken) {
                    return parseBracketedList(ParsingContext::TypeArguments, parseType, SyntaxKind::LessThanToken, SyntaxKind::GreaterThanToken);
                }
            }

            auto parseTypeReference() -> TypeReferenceNode {
                auto pos = getNodePos();
                return finishNode(
                    factory.createTypeReferenceNode(
                        parseEntityNameOfTypeReference(),
                        parseTypeArgumentsOfTypeReference()
                    ),
                    pos
                );
            }

            // If true, we should abort parsing an error function.
            auto typeHasArrowFunctionBlockingParseError(TypeNode node) -> boolean {
                switch (node.kind) {
                    case SyntaxKind::TypeReference:
                        return nodeIsMissing(node.as<TypeReferenceNode>().typeName);
                    case SyntaxKind::FunctionType:
                    case SyntaxKind::ConstructorType: {
                        auto { parameters, type } = node.as<FunctionOrConstructorTypeNode>();
                        return isMissingList(parameters) || typeHasArrowFunctionBlockingParseError(type);
                    }
                    case SyntaxKind::ParenthesizedType:
                        return typeHasArrowFunctionBlockingParseError(node.as<ParenthesizedTypeNode>().type);
                    default:
                        return false;
                }
            }

            auto parseThisTypePredicate(ThisTypeNode lhs) -> TypePredicateNode {
                nextToken();
                return finishNode(factory.createTypePredicateNode(/*assertsModifier*/ undefined, lhs, parseType()), lhs.pos);
            }

            auto parseThisTypeNode() -> ThisTypeNode {
                auto pos = getNodePos();
                nextToken();
                return finishNode(factory.createThisTypeNode(), pos);
            }

            auto parseJSDocAllType() -> Node {
                auto pos = getNodePos();
                nextToken();
                return finishNode(factory.createJSDocAllType(), pos);
            }

            auto parseJSDocNonNullableType() -> TypeNode {
                auto pos = getNodePos();
                nextToken();
                return finishNode(factory.createJSDocNonNullableType(parseNonArrayType()), pos);
            }

            auto parseJSDocUnknownOrNullableType() -> Node {
                auto pos = getNodePos();
                // skip the ?
                nextToken();

                // Need to lookahead to decide if this is a nullable or unknown type.

                // Here are cases where we'll pick the unknown type:
                //
                //      Foo(?,
                //      { a: ? }
                //      Foo(?)
                //      Foo<?>
                //      Foo(?=
                //      (?|
                if (token() == SyntaxKind::CommaToken ||
                    token() == SyntaxKind::CloseBraceToken ||
                    token() == SyntaxKind::CloseParenToken ||
                    token() == SyntaxKind::GreaterThanToken ||
                    token() == SyntaxKind::EqualsToken ||
                    token() == SyntaxKind::BarToken) {
                    return finishNode(factory.createJSDocUnknownType(), pos);
                }
                else {
                    return finishNode(factory.createJSDocNullableType(parseType()), pos);
                }
            }

            auto parseJSDocFunctionType() -> Node {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                if (lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParen, this))) {
                    nextToken();
                    auto parameters = parseParameters(SignatureFlags.Type | SignatureFlags.JSDoc);
                    auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
                    return withJSDoc(finishNode(factory.createJSDocFunctionType(parameters, type), pos), hasJSDoc);
                }
                return finishNode(factory.createTypeReferenceNode(parseIdentifierName(), /*typeArguments*/ undefined), pos);
            }

            auto parseJSDocParameter() -> ParameterDeclaration {
                auto pos = getNodePos();
                auto Identifier name;
                if (token() == SyntaxKind::ThisKeyword || token() == SyntaxKind::NewKeyword) {
                    name = parseIdentifierName();
                    parseExpected(SyntaxKind::ColonToken);
                }
                return finishNode(
                    factory.createParameterDeclaration(
                        /*decorators*/ undefined,
                        /*modifiers*/ undefined,
                        /*dotDotDotToken*/ undefined,
                        // TODO(rbuckton) -> JSDoc parameters don't have names (except `this`/`new`), should we manufacture an empty identifier?
                        name!,
                        /*questionToken*/ undefined,
                        parseJSDocType(),
                        /*initializer*/ undefined
                    ),
                    pos
                );
            }

            auto parseJSDocType() -> TypeNode {
                scanner.setInJSDocType(true);
                auto pos = getNodePos();
                if (parseOptional(SyntaxKind::ModuleKeyword)) {
                    // TODO(rbuckton) -> We never set the type for a JSDocNamepathType. What should we put here?
                    auto moduleTag = factory.createJSDocNamepathType(/*type*/ undefined!);
                    while terminate (true) {
                        switch (token()) {
                            case SyntaxKind::CloseBraceToken:
                            case SyntaxKind::EndOfFileToken:
                            case SyntaxKind::CommaToken:
                            case SyntaxKind::WhitespaceTrivia:
                                break terminate;
                            default:
                                nextTokenJSDoc();
                        }
                    }

                    scanner.setInJSDocType(false);
                    return finishNode(moduleTag, pos);
                }

                auto hasDotDotDot = parseOptional(SyntaxKind::DotDotDotToken);
                auto type = parseTypeOrTypePredicate();
                scanner.setInJSDocType(false);
                if (hasDotDotDot) {
                    type = finishNode(factory.createJSDocVariadicType(type), pos);
                }
                if (token() == SyntaxKind::EqualsToken) {
                    nextToken();
                    return finishNode(factory.createJSDocOptionalType(type), pos);
                }
                return type;
            }

            auto parseTypeQuery() -> TypeQueryNode {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::TypeOfKeyword);
                return finishNode(factory.createTypeQueryNode(parseEntityName(/*allowReservedWords*/ true)), pos);
            }

            auto parseTypeParameter() -> TypeParameterDeclaration {
                auto pos = getNodePos();
                auto name = parseIdentifier();
                auto TypeNode constraint;
                auto Expression expression;
                if (parseOptional(SyntaxKind::ExtendsKeyword)) {
                    // It's not uncommon for people to write improper constraints to a generic.  If the
                    // user writes a constraint that is an expression and not an actual type, then parse
                    // it out.as<an>() expression (so we can recover well), but report that a type is needed
                    // instead.
                    if (isStartOfType() || !isStartOfExpression()) {
                        constraint = parseType();
                    }
                    else {
                        // It was not a type, and it looked like an expression.  Parse out an expression
                        // here so we recover well.  it Note is important that we call parseUnaryExpression
                        // and not parseExpression here.  If the user has:
                        //
                        //      <T extends string()>
                        //
                        // We do *not* want to consume the `>`.as<we>()'re consuming the expression for string().
                        expression = parseUnaryExpressionOrHigher();
                    }
                }

                auto defaultType = parseOptional(SyntaxKind::EqualsToken) ? parseType() : undefined;
                auto node = factory.createTypeParameterDeclaration(name, constraint, defaultType);
                node.expression = expression;
                return finishNode(node, pos);
            }

            auto parseTypeParameters() -> NodeArray<TypeParameterDeclaration> {
                if (token() == SyntaxKind::LessThanToken) {
                    return parseBracketedList(ParsingContext::TypeParameters, parseTypeParameter, SyntaxKind::LessThanToken, SyntaxKind::GreaterThanToken);
                }
            }

            auto isStartOfParameter(boolean isJSDocParameter) -> boolean {
                return token() == SyntaxKind::DotDotDotToken ||
                    isBindingIdentifierOrPrivateIdentifierOrPattern() ||
                    isModifierKind(token()) ||
                    token() == SyntaxKind::AtToken ||
                    isStartOfType(/*inStartOfParameter*/ !isJSDocParameter);
            }

            auto parseNameOfParameter(ModifiersArray modifiers) {
                // FormalParameter [Yield,Await]:
                //      BindingElement[?Yield,?Await]
                auto name = parseIdentifierOrPattern(Diagnostics::Private_identifiers_cannot_be_used_as_parameters);
                if (getFullWidth(name) == 0 && !some(modifiers) && isModifierKind(token())) {
                    // in cases like
                    // 'use strict'
                    // auto foo(static)
                    // isParameter('static') == true, because of isModifier('static')
                    // however 'static' is not a legal identifier in a strict mode.
                    // so result of this auto will be ParameterDeclaration (flags = 0, name = missing, type = undefined, initializer = undefined)
                    // and current token will not change => parsing of the enclosing parameter list will last till the end of time (or OOM)
                    // to avoid this we'll advance cursor to the next token.
                    nextToken();
                }
                return name;
            }

            auto parseParameterInOuterAwaitContext() {
                return parseParameterWorker(/*inOuterAwaitContext*/ true);
            }

            auto parseParameter() -> ParameterDeclaration {
                return parseParameterWorker(/*inOuterAwaitContext*/ false);
            }

            auto parseParameterWorker(boolean inOuterAwaitContext) -> ParameterDeclaration {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                if (token() == SyntaxKind::ThisKeyword) {
                    auto node = factory.createParameterDeclaration(
                        /*decorators*/ undefined,
                        /*modifiers*/ undefined,
                        /*dotDotDotToken*/ undefined,
                        createIdentifier(/*isIdentifier*/ true),
                        /*questionToken*/ undefined,
                        parseTypeAnnotation(),
                        /*initializer*/ undefined
                    );
                    return withJSDoc(finishNode(node, pos), hasJSDoc);
                }

                // FormalParameter [Yield,Await]:
                //      BindingElement[?Yield,?Await]

                // Decorators are parsed in the outer [Await] context, the rest of the parameter is parsed in the function's [Await] context.
                auto decorators = inOuterAwaitContext ? doInAwaitContext(parseDecorators) : parseDecorators();
                auto savedTopLevel = topLevel;
                topLevel = false;
                auto modifiers = parseModifiers();
                auto node = withJSDoc(
                    finishNode(
                        factory.createParameterDeclaration(
                            decorators,
                            modifiers,
                            parseOptionalToken(SyntaxKind::DotDotDotToken),
                            parseNameOfParameter(modifiers),
                            parseOptionalToken(SyntaxKind::QuestionToken),
                            parseTypeAnnotation(),
                            parseInitializer()
                        ),
                        pos
                    ),
                    hasJSDoc
                );
                topLevel = savedTopLevel;
                return node;
            }

            auto parseReturnType(SyntaxKind returnToken, boolean isType) -> TypeNode {
                if (shouldParseReturnType(returnToken, isType)) {
                    return parseTypeOrTypePredicate();
                }
            }

            auto shouldParseReturnType(SyntaxKind returnToken, boolean isType) -> boolean {
                if (returnToken == SyntaxKind::EqualsGreaterThanToken) {
                    parseExpected(returnToken);
                    return true;
                }
                else if (parseOptional(SyntaxKind::ColonToken)) {
                    return true;
                }
                else if (isType && token() == SyntaxKind::EqualsGreaterThanToken) {
                    // This is easy to get backward, especially in type contexts, so parse the type anyway
                    parseErrorAtCurrentToken(Diagnostics::_0_expected, scanner.tokenToString(SyntaxKind::ColonToken));
                    nextToken();
                    return true;
                }
                return false;
            }

            auto parseParametersWorker(SignatureFlags flags) {
                // FormalParameters [Yield,Await]: (modified)
                //      [empty]
                //      FormalParameterList[?Yield,Await]
                //
                // FormalParameter[Yield,Await]: (modified)
                //      BindingElement[?Yield,Await]
                //
                // BindingElement [Yield,Await]: (modified)
                //      SingleNameBinding[?Yield,?Await]
                //      BindingPattern[?Yield,?Await]Initializer [In, ?Yield,?Await] opt
                //
                // SingleNameBinding [Yield,Await]:
                //      BindingIdentifier[?Yield,?Await]Initializer [In, ?Yield,?Await] opt
                auto savedYieldContext = inYieldContext();
                auto savedAwaitContext = inAwaitContext();

                setYieldContext(!!(flags & SignatureFlags.Yield));
                setAwaitContext(!!(flags & SignatureFlags.Await));

                auto parameters = flags & SignatureFlags.JSDoc ?
                    parseDelimitedList(ParsingContext::JSDocParameters, parseJSDocParameter) :
                    parseDelimitedList(ParsingContext::Parameters, savedAwaitContext ? parseParameterInOuterAwaitContext : parseParameter);

                setYieldContext(savedYieldContext);
                setAwaitContext(savedAwaitContext);

                return parameters;
            }

            auto parseParameters(SignatureFlags flags) -> NodeArray<ParameterDeclaration> {
                // FormalParameters [Yield,Await]: (modified)
                //      [empty]
                //      FormalParameterList[?Yield,Await]
                //
                // FormalParameter[Yield,Await]: (modified)
                //      BindingElement[?Yield,Await]
                //
                // BindingElement [Yield,Await]: (modified)
                //      SingleNameBinding[?Yield,?Await]
                //      BindingPattern[?Yield,?Await]Initializer [In, ?Yield,?Await] opt
                //
                // SingleNameBinding [Yield,Await]:
                //      BindingIdentifier[?Yield,?Await]Initializer [In, ?Yield,?Await] opt
                if (!parseExpected(SyntaxKind::OpenParenToken)) {
                    return createMissingList<ParameterDeclaration>();
                }

                auto parameters = parseParametersWorker(flags);
                parseExpected(SyntaxKind::CloseParenToken);
                return parameters;
            }

            auto parseTypeMemberSemicolon() {
                // We allow type members to be separated by commas or (possibly ASI) semicolons.
                // First check if it was a comma.  If so, we're done with the member.
                if (parseOptional(SyntaxKind::CommaToken)) {
                    return;
                }

                // Didn't have a comma.  We must have a (possible ASI) semicolon.
                parseSemicolon();
            }

            auto parseSignatureMember(SyntaxKind kind) -> Node {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                if (kind == SyntaxKind::ConstructSignature) {
                    parseExpected(SyntaxKind::NewKeyword);
                }

                auto typeParameters = parseTypeParameters();
                auto parameters = parseParameters(SignatureFlags.Type);
                auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ true);
                parseTypeMemberSemicolon();
                auto node = kind == SyntaxKind::CallSignature
                    ? factory.createCallSignature(typeParameters, parameters, type)
                    : factory.createConstructSignature(typeParameters, parameters, type);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto isIndexSignature() -> boolean {
                return token() == SyntaxKind::OpenBracketToken && lookAhead<boolean>(std::bind(&Parser::isUnambiguouslyIndexSignature, this));
            }

            auto isUnambiguouslyIndexSignature() {
                // The only allowed sequence is:
                //
                //   [id:
                //
                // However, for error recovery, we also check the following cases:
                //
                //   [...
                //   [id,
                //   [id?,
                //   [id?:
                //   [id?]
                //   [public id
                //   [private id
                //   [protected id
                //   []
                //
                nextToken();
                if (token() == SyntaxKind::DotDotDotToken || token() == SyntaxKind::CloseBracketToken) {
                    return true;
                }

                if (isModifierKind(token())) {
                    nextToken();
                    if (isIdentifier()) {
                        return true;
                    }
                }
                else if (!isIdentifier()) {
                    return false;
                }
                else {
                    // Skip the identifier
                    nextToken();
                }

                // A colon signifies a well formed indexer
                // A comma should be a badly formed indexer because comma expressions are not allowed
                // in computed properties.
                if (token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken) {
                    return true;
                }

                // Question mark could be an indexer with an optional property,
                // or it could be a conditional expression in a computed property.
                if (token() != SyntaxKind::QuestionToken) {
                    return false;
                }

                // If any of the following tokens are after the question mark, it cannot
                // be a conditional expression, so treat it.as<an>() indexer.
                nextToken();
                return token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken || token() == SyntaxKind::CloseBracketToken;
            }

            auto parseIndexSignatureDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> IndexSignatureDeclaration {
                auto parameters = parseBracketedList(ParsingContext::Parameters, parseParameter, SyntaxKind::OpenBracketToken, SyntaxKind::CloseBracketToken);
                auto type = parseTypeAnnotation();
                parseTypeMemberSemicolon();
                auto node = factory.createIndexSignature(decorators, modifiers, parameters, type);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parsePropertyOrMethodSignature(number pos, boolean hasJSDoc, NodeArray<Modifier> modifiers) -> Node {
                auto name = parsePropertyName();
                auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
                auto PropertySignature node | MethodSignature;
                if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken) {
                    // Method signatures don't exist in expression contexts.  So they have neither
                    // [Yield] nor [Await]
                    auto typeParameters = parseTypeParameters();
                    auto parameters = parseParameters(SignatureFlags.Type);
                    auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ true);
                    node = factory.createMethodSignature(modifiers, name, questionToken, typeParameters, parameters, type);
                }
                else {
                    auto type = parseTypeAnnotation();
                    node = factory.createPropertySignature(modifiers, name, questionToken, type);
                    // Although type literal properties cannot not have initializers, we attempt
                    // to parse an initializer so we can report in the checker that an interface
                    // property or type literal property cannot have an initializer.
                    if (token() == SyntaxKind::EqualsToken) node.initializer = parseInitializer();
                }
                parseTypeMemberSemicolon();
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto isTypeMemberStart() -> boolean {
                // Return true if we have the start of a signature member
                if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken) {
                    return true;
                }
                auto idToken = false;
                // Eat up all modifiers, but hold on to the last one in case it is actually an identifier
                while (isModifierKind(token())) {
                    idToken = true;
                    nextToken();
                }
                // Index signatures and computed property names are type members
                if (token() == SyntaxKind::OpenBracketToken) {
                    return true;
                }
                // Try to get the first property-like token following all modifiers
                if (isLiteralPropertyName()) {
                    idToken = true;
                    nextToken();
                }
                // If we were able to get any potential identifier, check that it is
                // the start of a member declaration
                if (idToken) {
                    return token() == SyntaxKind::OpenParenToken ||
                        token() == SyntaxKind::LessThanToken ||
                        token() == SyntaxKind::QuestionToken ||
                        token() == SyntaxKind::ColonToken ||
                        token() == SyntaxKind::CommaToken ||
                        canParseSemicolon();
                }
                return false;
            }

            auto parseTypeMember() -> TypeElement {
                if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken) {
                    return parseSignatureMember(SyntaxKind::CallSignature);
                }
                if (token() == SyntaxKind::NewKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParenOrLessThan, this))) {
                    return parseSignatureMember(SyntaxKind::ConstructSignature);
                }
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                auto modifiers = parseModifiers();
                if (isIndexSignature()) {
                    return parseIndexSignatureDeclaration(pos, hasJSDoc, /*decorators*/ undefined, modifiers);
                }
                return parsePropertyOrMethodSignature(pos, hasJSDoc, modifiers);
            }

            auto nextTokenIsOpenParenOrLessThan() {
                nextToken();
                return token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken;
            }

            auto nextTokenIsDot() {
                return nextToken() == SyntaxKind::DotToken;
            }

            auto nextTokenIsOpenParenOrLessThanOrDot() {
                switch (nextToken()) {
                    case SyntaxKind::OpenParenToken:
                    case SyntaxKind::LessThanToken:
                    case SyntaxKind::DotToken:
                        return true;
                }
                return false;
            }

            auto parseTypeLiteral() -> TypeLiteralNode {
                auto pos = getNodePos();
                return finishNode(factory.createTypeLiteralNode(parseObjectTypeMembers()), pos);
            }

            auto parseObjectTypeMembers() -> NodeArray<TypeElement> {
                auto NodeArray<TypeElement> members;
                if (parseExpected(SyntaxKind::OpenBraceToken)) {
                    members = parseList(ParsingContext::TypeMembers, parseTypeMember);
                    parseExpected(SyntaxKind::CloseBraceToken);
                }
                else {
                    members = createMissingList<TypeElement>();
                }

                return members;
            }

            auto isStartOfMappedType() {
                nextToken();
                if (token() == SyntaxKind::PlusToken || token() == SyntaxKind::MinusToken) {
                    return nextToken() == SyntaxKind::ReadonlyKeyword;
                }
                if (token() == SyntaxKind::ReadonlyKeyword) {
                    nextToken();
                }
                return token() == SyntaxKind::OpenBracketToken && nextTokenIsIdentifier() && nextToken() == SyntaxKind::InKeyword;
            }

            auto parseMappedTypeParameter() {
                auto pos = getNodePos();
                auto name = parseIdentifierName();
                parseExpected(SyntaxKind::InKeyword);
                auto type = parseType();
                return finishNode(factory.createTypeParameterDeclaration(name, type, /*defaultType*/ undefined), pos);
            }

            auto parseMappedType() {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::OpenBraceToken);
                auto ReadonlyKeyword readonlyToken | PlusToken | MinusToken;
                if (token() == SyntaxKind::ReadonlyKeyword || token() == SyntaxKind::PlusToken || token() == SyntaxKind::MinusToken) {
                    readonlyToken = parseTokenNode<ReadonlyKeyword | PlusToken | MinusToken>();
                    if (readonlyToken.kind != SyntaxKind::ReadonlyKeyword) {
                        parseExpected(SyntaxKind::ReadonlyKeyword);
                    }
                }
                parseExpected(SyntaxKind::OpenBracketToken);
                auto typeParameter = parseMappedTypeParameter();
                auto nameType = parseOptional(SyntaxKind::AsKeyword) ? parseType() : undefined;
                parseExpected(SyntaxKind::CloseBracketToken);
                auto QuestionToken questionToken | PlusToken | MinusToken;
                if (token() == SyntaxKind::QuestionToken || token() == SyntaxKind::PlusToken || token() == SyntaxKind::MinusToken) {
                    questionToken = parseTokenNode<QuestionToken | PlusToken | MinusToken>();
                    if (questionToken.kind != SyntaxKind::QuestionToken) {
                        parseExpected(SyntaxKind::QuestionToken);
                    }
                }
                auto type = parseTypeAnnotation();
                parseSemicolon();
                parseExpected(SyntaxKind::CloseBraceToken);
                return finishNode(factory.createMappedTypeNode(readonlyToken, typeParameter, nameType, questionToken, type), pos);
            }

            auto parseTupleElementType() {
                auto pos = getNodePos();
                if (parseOptional(SyntaxKind::DotDotDotToken)) {
                    return finishNode(factory.createRestTypeNode(parseType()), pos);
                }
                auto type = parseType();
                if (isJSDocNullableType(type) && type.pos == type.type.pos) {
                    auto node = factory.createOptionalTypeNode(type.type);
                    setTextRange(node, type);
                    (node.as<Mutable>()<Node>).flags = type.flags;
                    return node;
                }
                return type;
            }

            auto isNextTokenColonOrQuestionColon() {
                return nextToken() == SyntaxKind::ColonToken || (token() == SyntaxKind::QuestionToken && nextToken() == SyntaxKind::ColonToken);
            }

            auto isTupleElementName() {
                if (token() == SyntaxKind::DotDotDotToken) {
                    return scanner.tokenIsIdentifierOrKeyword(nextToken()) && isNextTokenColonOrQuestionColon();
                }
                return scanner.tokenIsIdentifierOrKeyword(token()) && isNextTokenColonOrQuestionColon();
            }

            auto parseTupleElementNameOrTupleElementType() {
                if (lookAhead<boolean>(std::bind(&Parser::isTupleElementName, this))) {
                    auto pos = getNodePos();
                    auto hasJSDoc = hasPrecedingJSDocComment();
                    auto dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
                    auto name = parseIdentifierName();
                    auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
                    parseExpected(SyntaxKind::ColonToken);
                    auto type = parseTupleElementType();
                    auto node = factory.createNamedTupleMember(dotDotDotToken, name, questionToken, type);
                    return withJSDoc(finishNode(node, pos), hasJSDoc);
                }
                return parseTupleElementType();
            }

            auto parseTupleType() -> TupleTypeNode {
                auto pos = getNodePos();
                return finishNode(
                    factory.createTupleTypeNode(
                        parseBracketedList(ParsingContext::TupleElementTypes, parseTupleElementNameOrTupleElementType, SyntaxKind::OpenBracketToken, SyntaxKind::CloseBracketToken)
                    ),
                    pos
                );
            }

            auto parseParenthesizedType() -> TypeNode {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::OpenParenToken);
                auto type = parseType();
                parseExpected(SyntaxKind::CloseParenToken);
                return finishNode(factory.createParenthesizedType(type), pos);
            }

            auto parseModifiersForConstructorType() -> NodeArray<Modifier> {
                auto NodeArray<Modifier> modifiers;
                if (token() == SyntaxKind::AbstractKeyword) {
                    auto pos = getNodePos();
                    nextToken();
                    auto modifier = finishNode(factory.createToken(SyntaxKind::AbstractKeyword), pos);
                    modifiers = createNodeArray<Modifier>([modifier], pos);
                }
                return modifiers;
            }

            auto parseFunctionOrConstructorType() -> TypeNode {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                auto modifiers = parseModifiersForConstructorType();
                auto isConstructorType = parseOptional(SyntaxKind::NewKeyword);
                auto typeParameters = parseTypeParameters();
                auto parameters = parseParameters(SignatureFlags.Type);
                auto type = parseReturnType(SyntaxKind::EqualsGreaterThanToken, /*isType*/ false);
                auto node = isConstructorType
                    ? factory.createConstructorTypeNode(modifiers, typeParameters, parameters, type)
                    : factory.createFunctionTypeNode(typeParameters, parameters, type);
                if (!isConstructorType) (node.as<Mutable>()<Node>).modifiers = modifiers;
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseKeywordAndNoDot() -> TypeNode {
                auto node = parseTokenNode<TypeNode>();
                return token() == SyntaxKind::DotToken ? undefined : node;
            }

            auto parseLiteralTypeNode(boolean negative) -> LiteralTypeNode {
                auto pos = getNodePos();
                if (negative) {
                    nextToken();
                }
                auto BooleanLiteral expression | NullLiteral | LiteralExpression | PrefixUnaryExpression =
                    token() == SyntaxKind::TrueKeyword || token() == SyntaxKind::FalseKeyword || token() == SyntaxKind::NullKeyword ?
                        parseTokenNode<BooleanLiteral | NullLiteral>() :
                        parseLiteralLikeNode(token()).as<LiteralExpression>();
                if (negative) {
                    expression = finishNode(factory.createPrefixUnaryExpression(SyntaxKind::MinusToken, expression), pos);
                }
                return finishNode(factory.createLiteralTypeNode(expression), pos);
            }

            auto isStartOfTypeOfImportType() {
                nextToken();
                return token() == SyntaxKind::ImportKeyword;
            }

            auto parseImportType() -> ImportTypeNode {
                sourceFlags |= NodeFlags::PossiblyContainsDynamicImport;
                auto pos = getNodePos();
                auto isTypeOf = parseOptional(SyntaxKind::TypeOfKeyword);
                parseExpected(SyntaxKind::ImportKeyword);
                parseExpected(SyntaxKind::OpenParenToken);
                auto type = parseType();
                parseExpected(SyntaxKind::CloseParenToken);
                auto qualifier = parseOptional(SyntaxKind::DotToken) ? parseEntityNameOfTypeReference() : undefined;
                auto typeArguments = parseTypeArgumentsOfTypeReference();
                return finishNode(factory.createImportTypeNode(type, qualifier, typeArguments, isTypeOf), pos);
            }

            auto nextTokenIsNumericOrBigIntLiteral() {
                nextToken();
                return token() == SyntaxKind::NumericLiteral || token() == SyntaxKind::BigIntLiteral;
            }

            auto parseNonArrayType() -> TypeNode {
                switch (token()) {
                    case SyntaxKind::AnyKeyword:
                    case SyntaxKind::UnknownKeyword:
                    case SyntaxKind::StringKeyword:
                    case SyntaxKind::NumberKeyword:
                    case SyntaxKind::BigIntKeyword:
                    case SyntaxKind::SymbolKeyword:
                    case SyntaxKind::BooleanKeyword:
                    case SyntaxKind::UndefinedKeyword:
                    case SyntaxKind::NeverKeyword:
                    case SyntaxKind::ObjectKeyword:
                        // If these are followed by a dot, then parse these out.as<a>() dotted type reference instead.
                        return tryParse<boolean>(std::bind(&Parser::parseKeywordAndNoDot, this)) || parseTypeReference();
                    case SyntaxKind::AsteriskEqualsToken:
                        // If there is '*=', treat it as * followed by postfix =
                        scanner.reScanAsteriskEqualsToken();
                        // falls through
                    case SyntaxKind::AsteriskToken:
                        return parseJSDocAllType();
                    case SyntaxKind::QuestionQuestionToken:
                        // If there is '??', treat it.as<prefix>()-'?' in JSDoc type.
                        scanner.reScanQuestionToken();
                        // falls through
                    case SyntaxKind::QuestionToken:
                        return parseJSDocUnknownOrNullableType();
                    case SyntaxKind::FunctionKeyword:
                        return parseJSDocFunctionType();
                    case SyntaxKind::ExclamationToken:
                        return parseJSDocNonNullableType();
                    case SyntaxKind::NoSubstitutionTemplateLiteral:
                    case SyntaxKind::StringLiteral:
                    case SyntaxKind::NumericLiteral:
                    case SyntaxKind::BigIntLiteral:
                    case SyntaxKind::TrueKeyword:
                    case SyntaxKind::FalseKeyword:
                    case SyntaxKind::NullKeyword:
                        return parseLiteralTypeNode();
                    case SyntaxKind::MinusToken:
                        return lookAhead<boolean>(std::bind(&Parser::nextTokenIsNumericOrBigIntLiteral, this)) ? parseLiteralTypeNode(/*negative*/ true) : parseTypeReference();
                    case SyntaxKind::VoidKeyword:
                        return parseTokenNode<TypeNode>();
                    case SyntaxKind::ThisKeyword: {
                        auto thisKeyword = parseThisTypeNode();
                        if (token() == SyntaxKind::IsKeyword && !scanner.hasPrecedingLineBreak()) {
                            return parseThisTypePredicate(thisKeyword);
                        }
                        else {
                            return thisKeyword;
                        }
                    }
                    case SyntaxKind::TypeOfKeyword:
                        return lookAhead<boolean>(std::bind(&Parser::isStartOfTypeOfImportType, this)) ? parseImportType() : parseTypeQuery();
                    case SyntaxKind::OpenBraceToken:
                        return lookAhead<boolean>(std::bind(&Parser::isStartOfMappedType, this)) ? parseMappedType() : parseTypeLiteral();
                    case SyntaxKind::OpenBracketToken:
                        return parseTupleType();
                    case SyntaxKind::OpenParenToken:
                        return parseParenthesizedType();
                    case SyntaxKind::ImportKeyword:
                        return parseImportType();
                    case SyntaxKind::AssertsKeyword:
                        return lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOnSameLine, this)) ? parseAssertsTypePredicate() : parseTypeReference();
                    case SyntaxKind::TemplateHead:
                        return parseTemplateType();
                    default:
                        return parseTypeReference();
                }
            }

            auto isStartOfType(boolean inStartOfParameter = false) -> boolean {
                switch (token()) {
                    case SyntaxKind::AnyKeyword:
                    case SyntaxKind::UnknownKeyword:
                    case SyntaxKind::StringKeyword:
                    case SyntaxKind::NumberKeyword:
                    case SyntaxKind::BigIntKeyword:
                    case SyntaxKind::BooleanKeyword:
                    case SyntaxKind::ReadonlyKeyword:
                    case SyntaxKind::SymbolKeyword:
                    case SyntaxKind::UniqueKeyword:
                    case SyntaxKind::VoidKeyword:
                    case SyntaxKind::UndefinedKeyword:
                    case SyntaxKind::NullKeyword:
                    case SyntaxKind::ThisKeyword:
                    case SyntaxKind::TypeOfKeyword:
                    case SyntaxKind::NeverKeyword:
                    case SyntaxKind::OpenBraceToken:
                    case SyntaxKind::OpenBracketToken:
                    case SyntaxKind::LessThanToken:
                    case SyntaxKind::BarToken:
                    case SyntaxKind::AmpersandToken:
                    case SyntaxKind::NewKeyword:
                    case SyntaxKind::StringLiteral:
                    case SyntaxKind::NumericLiteral:
                    case SyntaxKind::BigIntLiteral:
                    case SyntaxKind::TrueKeyword:
                    case SyntaxKind::FalseKeyword:
                    case SyntaxKind::ObjectKeyword:
                    case SyntaxKind::AsteriskToken:
                    case SyntaxKind::QuestionToken:
                    case SyntaxKind::ExclamationToken:
                    case SyntaxKind::DotDotDotToken:
                    case SyntaxKind::InferKeyword:
                    case SyntaxKind::ImportKeyword:
                    case SyntaxKind::AssertsKeyword:
                    case SyntaxKind::NoSubstitutionTemplateLiteral:
                    case SyntaxKind::TemplateHead:
                        return true;
                    case SyntaxKind::FunctionKeyword:
                        return !inStartOfParameter;
                    case SyntaxKind::MinusToken:
                        return !inStartOfParameter && lookAhead<boolean>(std::bind(&Parser::nextTokenIsNumericOrBigIntLiteral, this));
                    case SyntaxKind::OpenParenToken:
                        // Only consider '(' the start of a type if followed by ')', '...', an identifier, a modifier,
                        // or something that starts a type. We don't want to consider things like '(1)' a type.
                        return !inStartOfParameter && lookAhead<boolean>(std::bind(&Parser::isStartOfParenthesizedOrFunctionType, this));
                    default:
                        return isIdentifier();
                }
            }

            auto isStartOfParenthesizedOrFunctionType() {
                nextToken();
                return token() == SyntaxKind::CloseParenToken || isStartOfParameter(/*isJSDocParameter*/ false) || isStartOfType();
            }

            auto parsePostfixTypeOrHigher() -> TypeNode {
                auto pos = getNodePos();
                auto type = parseNonArrayType();
                while (!scanner.hasPrecedingLineBreak()) {
                    switch (token()) {
                        case SyntaxKind::ExclamationToken:
                            nextToken();
                            type = finishNode(factory.createJSDocNonNullableType(type), pos);
                            break;
                        case SyntaxKind::QuestionToken:
                            // If next token is start of a type we have a conditional type
                            if (lookAhead<boolean>(std::bind(&Parser::nextTokenIsStartOfType, this))) {
                                return type;
                            }
                            nextToken();
                            type = finishNode(factory.createJSDocNullableType(type), pos);
                            break;
                        case SyntaxKind::OpenBracketToken:
                            parseExpected(SyntaxKind::OpenBracketToken);
                            if (isStartOfType()) {
                                auto indexType = parseType();
                                parseExpected(SyntaxKind::CloseBracketToken);
                                type = finishNode(factory.createIndexedAccessTypeNode(type, indexType), pos);
                            }
                            else {
                                parseExpected(SyntaxKind::CloseBracketToken);
                                type = finishNode(factory.createArrayTypeNode(type), pos);
                            }
                            break;
                        default:
                            return type;
                    }
                }
                return type;
            }

            auto parseTypeOperator(SyntaxKind operator_) {
                auto pos = getNodePos();
                parseExpected(operator_);
                return finishNode(factory.createTypeOperatorNode(operator_, parseTypeOperatorOrHigher()), pos);
            }

            auto parseTypeParameterOfInferType() {
                auto pos = getNodePos();
                return finishNode(
                    factory.createTypeParameterDeclaration(
                        parseIdentifier(),
                        /*constraint*/ undefined,
                        /*defaultType*/ undefined
                    ),
                    pos
                );
            }

            auto parseInferType() -> InferTypeNode {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::InferKeyword);
                return finishNode(factory.createInferTypeNode(parseTypeParameterOfInferType()), pos);
            }

            auto parseTypeOperatorOrHigher() -> TypeNode {
                auto operator = token();
                switch (operator) {
                    case SyntaxKind::KeyOfKeyword:
                    case SyntaxKind::UniqueKeyword:
                    case SyntaxKind::ReadonlyKeyword:
                        return parseTypeOperator(operator);
                    case SyntaxKind::InferKeyword:
                        return parseInferType();
                }
                return parsePostfixTypeOrHigher();
            }

            auto parseFunctionOrConstructorTypeToError(
                boolean isInUnionType
            ) -> TypeNode {
                // the auto type and constructor type shorthand notation
                // are not allowed directly in unions and intersections, but we'll
                // try to parse them gracefully and issue a helpful message.
                if (isStartOfFunctionTypeOrConstructorType()) {
                    auto type = parseFunctionOrConstructorType();
                    auto DiagnosticMessage diagnostic;
                    if (isFunctionTypeNode(type)) {
                        diagnostic = isInUnionType
                            ? Diagnostics::Function_type_notation_must_be_parenthesized_when_used_in_a_union_type
                            : Diagnostics::Function_type_notation_must_be_parenthesized_when_used_in_an_intersection_type;
                    }
                    else {
                        diagnostic = isInUnionType
                            ? Diagnostics::Constructor_type_notation_must_be_parenthesized_when_used_in_a_union_type
                            : Diagnostics::Constructor_type_notation_must_be_parenthesized_when_used_in_an_intersection_type;

                    }
                    parseErrorAtRange(type, diagnostic);
                    return type;
                }
                return undefined;
            }

            auto parseUnionOrIntersectionType(
                SyntaxKind operator_,
                std::function<TypeNode()> parseConstituentType,
                std::function<UnionOrIntersectionTypeNode(NodeArray<TypeNode>)> createTypeNode
            ) -> TypeNode {
                auto pos = getNodePos();
                auto isUnionType = operator_ == SyntaxKind::BarToken;
                auto hasLeadingOperator = parseOptional(operator_);
                auto type = hasLeadingOperator && parseFunctionOrConstructorTypeToError(isUnionType)
                    || parseConstituentType();
                if (token() == operator_ || hasLeadingOperator) {
                    auto types = [type];
                    while (parseOptional(operator_)) {
                        types.push_back(parseFunctionOrConstructorTypeToError(isUnionType) || parseConstituentType());
                    }
                    type = finishNode(createTypeNode(createNodeArray(types, pos)), pos);
                }
                return type;
            }

            auto parseIntersectionTypeOrHigher() -> TypeNode {
                return parseUnionOrIntersectionType(SyntaxKind::AmpersandToken, parseTypeOperatorOrHigher, factory.createIntersectionTypeNode);
            }

            auto parseUnionTypeOrHigher() -> TypeNode {
                return parseUnionOrIntersectionType(SyntaxKind::BarToken, parseIntersectionTypeOrHigher, factory.createUnionTypeNode);
            }

            auto nextTokenIsNewKeyword() -> boolean {
                nextToken();
                return token() == SyntaxKind::NewKeyword;
            }

            auto isStartOfFunctionTypeOrConstructorType() -> boolean {
                if (token() == SyntaxKind::LessThanToken) {
                    return true;
                }
                if (token() == SyntaxKind::OpenParenToken && lookAhead<boolean>(std::bind(&Parser::isUnambiguouslyStartOfFunctionType, this))) {
                    return true;
                }
                return token() == SyntaxKind::NewKeyword ||
                    token() == SyntaxKind::AbstractKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsNewKeyword, this));
            }

            auto skipParameterStart() -> boolean {
                if (isModifierKind(token())) {
                    // Skip modifiers
                    parseModifiers();
                }
                if (isIdentifier() || token() == SyntaxKind::ThisKeyword) {
                    nextToken();
                    return true;
                }
                if (token() == SyntaxKind::OpenBracketToken || token() == SyntaxKind::OpenBraceToken) {
                    // Return true if we can parse an array or object binding pattern with no errors
                    auto previousErrorCount = parseDiagnostics::size();
                    parseIdentifierOrPattern();
                    return previousErrorCount == parseDiagnostics::size();
                }
                return false;
            }

            auto isUnambiguouslyStartOfFunctionType() {
                nextToken();
                if (token() == SyntaxKind::CloseParenToken || token() == SyntaxKind::DotDotDotToken) {
                    // ( )
                    // ( ...
                    return true;
                }
                if (skipParameterStart()) {
                    // We successfully skipped modifiers (if any) and an identifier or binding pattern,
                    // now see if we have something that indicates a parameter declaration
                    if (token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken ||
                        token() == SyntaxKind::QuestionToken || token() == SyntaxKind::EqualsToken) {
                        // ( xxx :
                        // ( xxx ,
                        // ( xxx ?
                        // ( xxx =
                        return true;
                    }
                    if (token() == SyntaxKind::CloseParenToken) {
                        nextToken();
                        if (token() == SyntaxKind::EqualsGreaterThanToken) {
                            // ( xxx ) =>
                            return true;
                        }
                    }
                }
                return false;
            }

            auto parseTypeOrTypePredicate() -> TypeNode {
                auto pos = getNodePos();
                auto typePredicateVariable = isIdentifier() && tryParse<boolean>(std::bind(&Parser::parseTypePredicatePrefix, this));
                auto type = parseType();
                if (typePredicateVariable) {
                    return finishNode(factory.createTypePredicateNode(/*assertsModifier*/ undefined, typePredicateVariable, type), pos);
                }
                else {
                    return type;
                }
            }

            auto parseTypePredicatePrefix() {
                auto id = parseIdentifier();
                if (token() == SyntaxKind::IsKeyword && !scanner.hasPrecedingLineBreak()) {
                    nextToken();
                    return id;
                }
            }

            auto parseAssertsTypePredicate() -> TypeNode {
                auto pos = getNodePos();
                auto assertsModifier = parseExpectedToken(SyntaxKind::AssertsKeyword);
                auto parameterName = token() == SyntaxKind::ThisKeyword ? parseThisTypeNode() : parseIdentifier();
                auto type = parseOptional(SyntaxKind::IsKeyword) ? parseType() : undefined;
                return finishNode(factory.createTypePredicateNode(assertsModifier, parameterName, type), pos);
            }

            auto parseType() -> TypeNode {
                // The rules about 'yield' only apply to actual code/expression contexts.  They don't
                // apply to 'type' contexts.  So we disable these parameters here before moving on.
                return doOutsideOfContext(NodeFlags::TypeExcludesFlags, parseTypeWorker);
            }

            auto parseTypeWorker(boolean noConditionalTypes) -> TypeNode {
                if (isStartOfFunctionTypeOrConstructorType()) {
                    return parseFunctionOrConstructorType();
                }
                auto pos = getNodePos();
                auto type = parseUnionTypeOrHigher();
                if (!noConditionalTypes && !scanner.hasPrecedingLineBreak() && parseOptional(SyntaxKind::ExtendsKeyword)) {
                    // The type following 'extends' is not permitted to be another conditional type
                    auto extendsType = parseTypeWorker(/*noConditionalTypes*/ true);
                    parseExpected(SyntaxKind::QuestionToken);
                    auto trueType = parseTypeWorker();
                    parseExpected(SyntaxKind::ColonToken);
                    auto falseType = parseTypeWorker();
                    return finishNode(factory.createConditionalTypeNode(type, extendsType, trueType, falseType), pos);
                }
                return type;
            }

            auto parseTypeAnnotation() -> TypeNode {
                return parseOptional(SyntaxKind::ColonToken) ? parseType() : undefined;
            }

            // EXPRESSIONS
            auto isStartOfLeftHandSideExpression() -> boolean {
                switch (token()) {
                    case SyntaxKind::ThisKeyword:
                    case SyntaxKind::SuperKeyword:
                    case SyntaxKind::NullKeyword:
                    case SyntaxKind::TrueKeyword:
                    case SyntaxKind::FalseKeyword:
                    case SyntaxKind::NumericLiteral:
                    case SyntaxKind::BigIntLiteral:
                    case SyntaxKind::StringLiteral:
                    case SyntaxKind::NoSubstitutionTemplateLiteral:
                    case SyntaxKind::TemplateHead:
                    case SyntaxKind::OpenParenToken:
                    case SyntaxKind::OpenBracketToken:
                    case SyntaxKind::OpenBraceToken:
                    case SyntaxKind::FunctionKeyword:
                    case SyntaxKind::ClassKeyword:
                    case SyntaxKind::NewKeyword:
                    case SyntaxKind::SlashToken:
                    case SyntaxKind::SlashEqualsToken:
                    case SyntaxKind::Identifier:
                        return true;
                    case SyntaxKind::ImportKeyword:
                        return lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParenOrLessThanOrDot, this));
                    default:
                        return isIdentifier();
                }
            }

            auto isStartOfExpression() -> boolean {
                if (isStartOfLeftHandSideExpression()) {
                    return true;
                }

                switch (token()) {
                    case SyntaxKind::PlusToken:
                    case SyntaxKind::MinusToken:
                    case SyntaxKind::TildeToken:
                    case SyntaxKind::ExclamationToken:
                    case SyntaxKind::DeleteKeyword:
                    case SyntaxKind::TypeOfKeyword:
                    case SyntaxKind::VoidKeyword:
                    case SyntaxKind::PlusPlusToken:
                    case SyntaxKind::MinusMinusToken:
                    case SyntaxKind::LessThanToken:
                    case SyntaxKind::AwaitKeyword:
                    case SyntaxKind::YieldKeyword:
                    case SyntaxKind::PrivateIdentifier:
                        // Yield/await always starts an expression.  Either it is an identifier (in which case
                        // it is definitely an expression).  Or it's a keyword (either because we're in
                        // a generator or async function, or in strict mode (or both)) and it started a yield or await expression.
                        return true;
                    default:
                        // Error tolerance.  If we see the start of some binary operator, we consider
                        // that the start of an expression.  That way we'll parse out a missing identifier,
                        // give a good message about an identifier being missing, and then consume the
                        // rest of the binary expression.
                        if (isBinaryOperator()) {
                            return true;
                        }

                        return isIdentifier();
                }
            }

            auto isStartOfExpressionStatement() -> boolean {
                // As per the grammar, none of '{' or 'function' or 'class' can start an expression statement.
                return token() != SyntaxKind::OpenBraceToken &&
                    token() != SyntaxKind::FunctionKeyword &&
                    token() != SyntaxKind::ClassKeyword &&
                    token() != SyntaxKind::AtToken &&
                    isStartOfExpression();
            }

            auto parseExpression() -> Expression {
                // Expression[in]:
                //      AssignmentExpression[in]
                //      Expression[in] , AssignmentExpression[in]

                // clear the decorator context when parsing Expression,.as<it>() should be unambiguous when parsing a decorator
                auto saveDecoratorContext = inDecoratorContext();
                if (saveDecoratorContext) {
                    setDecoratorContext(/*val*/ false);
                }

                auto pos = getNodePos();
                auto expr = parseAssignmentExpressionOrHigher();
                auto BinaryOperatorToken operatorToken;
                while ((operatorToken = parseOptionalToken(SyntaxKind::CommaToken))) {
                    expr = makeBinaryExpression(expr, operatorToken, parseAssignmentExpressionOrHigher(), pos);
                }

                if (saveDecoratorContext) {
                    setDecoratorContext(/*val*/ true);
                }
                return expr;
            }

            auto parseInitializer() -> Expression {
                return parseOptional(SyntaxKind::EqualsToken) ? parseAssignmentExpressionOrHigher() : undefined;
            }

            auto parseAssignmentExpressionOrHigher() -> Expression {
                //  AssignmentExpression[in,yield]:
                //      1) ConditionalExpression[?in,?yield]
                //      2) LeftHandSideExpression = AssignmentExpression[?in,?yield]
                //      3) LeftHandSideExpression AssignmentOperator AssignmentExpression[?in,?yield]
                //      4) ArrowFunctionExpression[?in,?yield]
                //      5) AsyncArrowFunctionExpression[in,yield,await]
                //      6) [+Yield] YieldExpression[?In]
                //
                // for Note ease of implementation we treat productions '2' and '3'.as<the>() same thing.
                // (i.e. they're both BinaryExpressions with an assignment operator in it).

                // First, do the simple check if we have a YieldExpression (production '6').
                if (isYieldExpression()) {
                    return parseYieldExpression();
                }

                // Then, check if we have an arrow auto (production '4' and '5') that starts with a parenthesized
                // parameter list or is an async arrow function.
                // AsyncArrowFunctionExpression:
                //      1) async[no LineTerminator here]AsyncArrowBindingIdentifier[?Yield][no LineTerminator here]=>AsyncConciseBody[?In]
                //      2) CoverCallExpressionAndAsyncArrowHead[?Yield, ?Await][no LineTerminator here]=>AsyncConciseBody[?In]
                // Production (1) of AsyncArrowFunctionExpression is parsed in "tryParseAsyncSimpleArrowFunctionExpression".
                // And production (2) is parsed in "tryParseParenthesizedArrowFunctionExpression".
                //
                // If we do successfully parse arrow-function, we must *not* recurse for productions 1, 2 or 3. An ArrowFunction is
                // not a LeftHandSideExpression, nor does it start a ConditionalExpression.  So we are done
                // with AssignmentExpression if we see one.
                auto arrowExpression = tryParseParenthesizedArrowFunctionExpression() || tryParseAsyncSimpleArrowFunctionExpression();
                if (arrowExpression) {
                    return arrowExpression;
                }

                // Now try to see if we're in production '1', '2' or '3'.  A conditional expression can
                // start with a LogicalOrExpression, while the assignment productions can only start with
                // LeftHandSideExpressions.
                //
                // So, first, we try to just parse out a BinaryExpression.  If we get something that is a
                // LeftHandSide or higher, then we can try to parse out the assignment expression part.
                // Otherwise, we try to parse out the conditional expression bit.  We want to allow any
                // binary expression here, so we pass in the 'lowest' precedence here so that it matches
                // and consumes anything.
                auto pos = getNodePos();
                auto expr = parseBinaryExpressionOrHigher(OperatorPrecedence.Lowest);

                // To avoid a look-ahead, we did not handle the case of an arrow auto with a single un-parenthesized
                // parameter ('x => ...') above. We handle it here by checking if the parsed expression was a single
                // identifier and the current token is an arrow.
                if (expr.kind == SyntaxKind::Identifier && token() == SyntaxKind::EqualsGreaterThanToken) {
                    return parseSimpleArrowFunctionExpression(pos, expr.as<Identifier>(), /*asyncModifier*/ undefined);
                }

                // Now see if we might be in cases '2' or '3'.
                // If the expression was a LHS expression, and we have an assignment operator, then
                // we're in '2' or '3'. Consume the assignment and return.
                //
                // we Note call reScanGreaterToken so that we get an appropriately merged token
                // for cases like `> > =` becoming `>>=`
                if (isLeftHandSideExpression(expr) && isAssignmentOperator(reScanGreaterToken())) {
                    return makeBinaryExpression(expr, parseTokenNode(), parseAssignmentExpressionOrHigher(), pos);
                }

                // It wasn't an assignment or a lambda.  This is a conditional expression:
                return parseConditionalExpressionRest(expr, pos);
            }

            auto isYieldExpression() -> boolean {
                if (token() == SyntaxKind::YieldKeyword) {
                    // If we have a 'yield' keyword, and this is a context where yield expressions are
                    // allowed, then definitely parse out a yield expression.
                    if (inYieldContext()) {
                        return true;
                    }

                    // We're in a context where 'yield expr' is not allowed.  However, if we can
                    // definitely tell that the user was trying to parse a 'yield expr' and not
                    // just a normal expr that start with a 'yield' identifier, then parse out
                    // a 'yield expr'.  We can then report an error later that they are only
                    // allowed in generator expressions.
                    //
                    // for example, if we see 'yield(foo)', then we'll have to treat that.as<an>()
                    // invocation expression of something called 'yield'.  However, if we have
                    // 'yield foo' then that is not legal.as<a>() normal expression, so we can
                    // definitely recognize this.as<a>() yield expression.
                    //
                    // for now we just check if the next token is an identifier.  More heuristics
                    // can be added here later.as<necessary>().  We just need to make sure that we
                    // don't accidentally consume something legal.
                    return lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOrLiteralOnSameLine, this));
                }

                return false;
            }

            auto nextTokenIsIdentifierOnSameLine() {
                nextToken();
                return !scanner.hasPrecedingLineBreak() && isIdentifier();
            }

            auto parseYieldExpression() -> YieldExpression {
                auto pos = getNodePos();

                // YieldExpression[In] :
                //      yield
                //      yield [no LineTerminator here] [Lexical goal InputElementRegExp]AssignmentExpression[?In, Yield]
                //      yield [no LineTerminator here] * [Lexical goal InputElementRegExp]AssignmentExpression[?In, Yield]
                nextToken();

                if (!scanner.hasPrecedingLineBreak() &&
                    (token() == SyntaxKind::AsteriskToken || isStartOfExpression())) {
                    return finishNode(
                        factory.createYieldExpression(
                            parseOptionalToken(SyntaxKind::AsteriskToken),
                            parseAssignmentExpressionOrHigher()
                        ),
                        pos
                    );
                }
                else {
                    // if the next token is not on the same line.as<yield>().  or we don't have an '*' or
                    // the start of an expression, then this is just a simple "yield" expression.
                    return finishNode(factory.createYieldExpression(/*asteriskToken*/ undefined, /*expression*/ undefined), pos);
                }
            }

            auto parseSimpleArrowFunctionExpression(number pos, Identifier identifier, NodeArray<Modifier> asyncModifier) -> ArrowFunction {
                Debug::_assert(token() == SyntaxKind::EqualsGreaterThanToken, "parseSimpleArrowFunctionExpression should only have been called if we had a =>");
                auto parameter = factory.createParameterDeclaration(
                    /*decorators*/ undefined,
                    /*modifiers*/ undefined,
                    /*dotDotDotToken*/ undefined,
                    identifier,
                    /*questionToken*/ undefined,
                    /*type*/ undefined,
                    /*initializer*/ undefined
                );
                finishNode(parameter, identifier.pos);

                auto parameters = createNodeArray<ParameterDeclaration>([parameter], parameter.pos, parameter.end);
                auto equalsGreaterThanToken = parseExpectedToken(SyntaxKind::EqualsGreaterThanToken);
                auto body = parseArrowFunctionExpressionBody(/*isAsync*/ !!asyncModifier);
                auto node = factory.createArrowFunction(asyncModifier, /*typeParameters*/ undefined, parameters, /*type*/ undefined, equalsGreaterThanToken, body);
                return addJSDocComment(finishNode(node, pos));
            }

            auto tryParseParenthesizedArrowFunctionExpression() -> Expression {
                auto triState = isParenthesizedArrowFunctionExpression();
                if (triState == Tristate.False) {
                    // It's definitely not a parenthesized arrow auto expression.
                    return undefined;
                }

                // If we definitely have an arrow function, then we can just parse one, not requiring a
                // following => or { token. Otherwise, we *might* have an arrow function.  Try to parse
                // it out, but don't allow any ambiguity, and return 'undefined' if this could be an
                // expression instead.
                return triState == Tristate.True ?
                    parseParenthesizedArrowFunctionExpression(/*allowAmbiguity*/ true) :
                    tryParse<boolean>(std::bind(&Parser::parsePossibleParenthesizedArrowFunctionExpression, this));
            }

            //  True        -> We definitely expect a parenthesized arrow auto here.
            //  False       -> There *cannot* be a parenthesized arrow auto here.
            //  Unknown     -> There *might* be a parenthesized arrow auto here.
            //                 Speculatively look ahead to be sure, and rollback if not.
            auto isParenthesizedArrowFunctionExpression() -> Tristate {
                if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken || token() == SyntaxKind::AsyncKeyword) {
                    return lookAhead<boolean>(std::bind(&Parser::isParenthesizedArrowFunctionExpressionWorker, this));
                }

                if (token() == SyntaxKind::EqualsGreaterThanToken) {
                    // ERROR RECOVERY TWEAK:
                    // If we see a standalone => try to parse it.as<an>() arrow auto expression.as<that>()'s
                    // likely what the user intended to write.
                    return Tristate.True;
                }
                // Definitely not a parenthesized arrow function.
                return Tristate.False;
            }

            auto isParenthesizedArrowFunctionExpressionWorker() {
                if (token() == SyntaxKind::AsyncKeyword) {
                    nextToken();
                    if (scanner.hasPrecedingLineBreak()) {
                        return Tristate.False;
                    }
                    if (token() != SyntaxKind::OpenParenToken && token() != SyntaxKind::LessThanToken) {
                        return Tristate.False;
                    }
                }

                auto first = token();
                auto second = nextToken();

                if (first == SyntaxKind::OpenParenToken) {
                    if (second == SyntaxKind::CloseParenToken) {
                        // Simple cases: "() =>", "() -> ", and "() {".
                        // This is an arrow auto with no parameters.
                        // The last one is not actually an arrow function,
                        // but this is probably what the user intended.
                        auto third = nextToken();
                        switch (third) {
                            case SyntaxKind::EqualsGreaterThanToken:
                            case SyntaxKind::ColonToken:
                            case SyntaxKind::OpenBraceToken:
                                return Tristate.True;
                            default:
                                return Tristate.False;
                        }
                    }

                    // If encounter "([" or "({", this could be the start of a binding pattern.
                    // Examples:
                    //      ([ x ]) => { }
                    //      ({ x }) => { }
                    //      ([ x ])
                    //      ({ x })
                    if (second == SyntaxKind::OpenBracketToken || second == SyntaxKind::OpenBraceToken) {
                        return Tristate.Unknown;
                    }

                    // Simple case: "(..."
                    // This is an arrow auto with a rest parameter.
                    if (second == SyntaxKind::DotDotDotToken) {
                        return Tristate.True;
                    }

                    // Check for "(xxx yyy", where xxx is a modifier and yyy is an identifier. This
                    // isn't actually allowed, but we want to treat it.as<a>() lambda so we can provide
                    // a good error message.
                    if (isModifierKind(second) && second != SyntaxKind::AsyncKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifier, this))) {
                        return Tristate.True;
                    }

                    // If we had "(" followed by something that's not an identifier,
                    // then this definitely doesn't look like a lambda.  "this" is not
                    // valid, but we want to parse it and then give a semantic error.
                    if (!isIdentifier() && second != SyntaxKind::ThisKeyword) {
                        return Tristate.False;
                    }

                    switch (nextToken()) {
                        case SyntaxKind::ColonToken:
                            // If we have something like "(a:", then we must have a
                            // type-annotated parameter in an arrow auto expression.
                            return Tristate.True;
                        case SyntaxKind::QuestionToken:
                            nextToken();
                            // If we have "(a?:" or "(a?," or "(a?=" or "(a?)" then it is definitely a lambda.
                            if (token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken || token() == SyntaxKind::EqualsToken || token() == SyntaxKind::CloseParenToken) {
                                return Tristate.True;
                            }
                            // Otherwise it is definitely not a lambda.
                            return Tristate.False;
                        case SyntaxKind::CommaToken:
                        case SyntaxKind::EqualsToken:
                        case SyntaxKind::CloseParenToken:
                            // If we have "(a," or "(a=" or "(a)" this *could* be an arrow function
                            return Tristate.Unknown;
                    }
                    // It is definitely not an arrow function
                    return Tristate.False;
                }
                else {
                    Debug::_assert(first == SyntaxKind::LessThanToken);

                    // If we have "<" not followed by an identifier,
                    // then this definitely is not an arrow function.
                    if (!isIdentifier()) {
                        return Tristate.False;
                    }

                    // JSX overrides
                    if (languageVariant == LanguageVariant::JSX) {
                        auto isArrowFunctionInJsx = lookAhead(() => {
                            auto third = nextToken();
                            if (third == SyntaxKind::ExtendsKeyword) {
                                auto fourth = nextToken();
                                switch (fourth) {
                                    case SyntaxKind::EqualsToken:
                                    case SyntaxKind::GreaterThanToken:
                                        return false;
                                    default:
                                        return true;
                                }
                            }
                            else if (third == SyntaxKind::CommaToken) {
                                return true;
                            }
                            return false;
                        });

                        if (isArrowFunctionInJsx) {
                            return Tristate.True;
                        }

                        return Tristate.False;
                    }

                    // This *could* be a parenthesized arrow function.
                    return Tristate.Unknown;
                }
            }

            auto parsePossibleParenthesizedArrowFunctionExpression() -> ArrowFunction {
                auto tokenPos = scanner.getTokenPos();
                if (notParenthesizedArrow?.has(tokenPos)) {
                    return undefined;
                }

                auto result = parseParenthesizedArrowFunctionExpression(/*allowAmbiguity*/ false);
                if (!result) {
                    (notParenthesizedArrow || (notParenthesizedArrow = new Set())).add(tokenPos);
                }

                return result;
            }

            auto tryParseAsyncSimpleArrowFunctionExpression() -> ArrowFunction {
                // We do a check here so that we won't be doing unnecessarily call to "lookAhead"
                if (token() == SyntaxKind::AsyncKeyword) {
                    if (lookAhead<Tristate>(std::bind(&Parser::isUnParenthesizedAsyncArrowFunctionWorker, this)) == Tristate::True) {
                        auto pos = getNodePos();
                        auto asyncModifier = parseModifiersForArrowFunction();
                        auto expr = parseBinaryExpressionOrHigher(OperatorPrecedence.Lowest);
                        return parseSimpleArrowFunctionExpression(pos, expr.as<Identifier>(), asyncModifier);
                    }
                }
                return undefined;
            }

            auto isUnParenthesizedAsyncArrowFunctionWorker() -> Tristate {
                // AsyncArrowFunctionExpression:
                //      1) async[no LineTerminator here]AsyncArrowBindingIdentifier[?Yield][no LineTerminator here]=>AsyncConciseBody[?In]
                //      2) CoverCallExpressionAndAsyncArrowHead[?Yield, ?Await][no LineTerminator here]=>AsyncConciseBody[?In]
                if (token() == SyntaxKind::AsyncKeyword) {
                    nextToken();
                    // If the "async" is followed by "=>" token then it is not a beginning of an async arrow-function
                    // but instead a simple arrow-auto which will be parsed inside "parseAssignmentExpressionOrHigher"
                    if (scanner.hasPrecedingLineBreak() || token() == SyntaxKind::EqualsGreaterThanToken) {
                        return Tristate::False;
                    }
                    // Check for un-parenthesized AsyncArrowFunction
                    auto expr = parseBinaryExpressionOrHigher(OperatorPrecedence.Lowest);
                    if (!scanner.hasPrecedingLineBreak() && expr.kind == SyntaxKind::Identifier && token() == SyntaxKind::EqualsGreaterThanToken) {
                        return Tristate.True;
                    }
                }

                return Tristate.False;
            }

            auto parseParenthesizedArrowFunctionExpression(boolean allowAmbiguity) -> ArrowFunction {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                auto modifiers = parseModifiersForArrowFunction();
                auto isAsync = some(modifiers, isAsyncModifier) ? SignatureFlags.Await : SignatureFlags.None;
                // Arrow functions are never generators.
                //
                // If we're speculatively parsing a signature for a parenthesized arrow function, then
                // we have to have a complete parameter list.  Otherwise we might see something like
                // a => (b => c)
                // And think that "(b =>" was actually a parenthesized arrow auto with a missing
                // close paren.
                auto typeParameters = parseTypeParameters();

                auto NodeArray<ParameterDeclaration> parameters;
                if (!parseExpected(SyntaxKind::OpenParenToken)) {
                    if (!allowAmbiguity) {
                        return undefined;
                    }
                    parameters = createMissingList<ParameterDeclaration>();
                }
                else {
                    parameters = parseParametersWorker(isAsync);
                    if (!parseExpected(SyntaxKind::CloseParenToken) && !allowAmbiguity) {
                        return undefined;
                    }
                }

                auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
                if (type && !allowAmbiguity && typeHasArrowFunctionBlockingParseError(type)) {
                    return undefined;
                }

                // Parsing a signature isn't enough.
                // Parenthesized arrow signatures often look like other valid expressions.
                // For instance:
                //  - "(x = 10)" is an assignment expression parsed.as<a>() signature with a default parameter value.
                //  - "(x,y)" is a comma expression parsed.as<a>() signature with two parameters.
                //  - "a ? (b) -> c" will have "(b) ->" parsed.as<a>() signature with a return type annotation.
                //  - "a ? (b) -> function() {}" will too, since function() is a valid JSDoc auto type.
                //
                // So we need just a bit of lookahead to ensure that it can only be a signature.
                auto hasJSDocFunctionType = type && isJSDocFunctionType(type);
                if (!allowAmbiguity && token() != SyntaxKind::EqualsGreaterThanToken && (hasJSDocFunctionType || token() != SyntaxKind::OpenBraceToken)) {
                    // Returning undefined here will cause our caller to rewind to where we started from.
                        return undefined;
                }

                // If we have an arrow, then try to parse the body. Even if not, try to parse if we
                // have an opening brace, just in case we're in an error state.
                auto lastToken = token();
                auto equalsGreaterThanToken = parseExpectedToken(SyntaxKind::EqualsGreaterThanToken);
                auto body = (lastToken == SyntaxKind::EqualsGreaterThanToken || lastToken == SyntaxKind::OpenBraceToken)
                    ? parseArrowFunctionExpressionBody(some(modifiers, isAsyncModifier))
                    : parseIdentifier();

                auto node = factory.createArrowFunction(modifiers, typeParameters, parameters, type, equalsGreaterThanToken, body);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseArrowFunctionExpressionBody(boolean isAsync) -> Node {
                if (token() == SyntaxKind::OpenBraceToken) {
                    return parseFunctionBlock(isAsync ? SignatureFlags.Await : SignatureFlags.None);
                }

                if (token() != SyntaxKind::SemicolonToken &&
                    token() != SyntaxKind::FunctionKeyword &&
                    token() != SyntaxKind::ClassKeyword &&
                    isStartOfStatement() &&
                    !isStartOfExpressionStatement()) {
                    // Check if we got a plain statement (i.e. no expression-statements, no function/class expressions/declarations)
                    //
                    // Here we try to recover from a potential error situation in the case where the
                    // user meant to supply a block. For example, if the user wrote:
                    //
                    //  a =>
                    //      auto v = 0;
                    //  }
                    //
                    // they may be missing an open brace.  Check to see if that's the case so we can
                    // try to recover better.  If we don't do this, then the next close curly we see may end
                    // up preemptively closing the containing construct.
                    //
                    // even Note when 'IgnoreMissingOpenBrace' is passed, parseBody will still error.
                    return parseFunctionBlock(SignatureFlags.IgnoreMissingOpenBrace | (isAsync ? SignatureFlags.Await : SignatureFlags.None));
                }

                auto savedTopLevel = topLevel;
                topLevel = false;
                auto node = isAsync
                    ? doInAwaitContext(parseAssignmentExpressionOrHigher)
                    : doOutsideOfAwaitContext(parseAssignmentExpressionOrHigher);
                topLevel = savedTopLevel;
                return node;
            }

            auto parseConditionalExpressionRest(Expression leftOperand, number pos) -> Expression {
                // we Note are passed in an expression which was produced from parseBinaryExpressionOrHigher.
                auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
                if (!questionToken) {
                    return leftOperand;
                }

                // we Note explicitly 'allowIn' in the whenTrue part of the condition expression, and
                // we do not that for the 'whenFalse' part.
                auto colonToken;
                return finishNode(
                    factory.createConditionalExpression(
                        leftOperand,
                        questionToken,
                        doOutsideOfContext(disallowInAndDecoratorContext, parseAssignmentExpressionOrHigher),
                        colonToken = parseExpectedToken(SyntaxKind::ColonToken),
                        nodeIsPresent(colonToken)
                            ? parseAssignmentExpressionOrHigher()
                            : createMissingNode(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ false, Diagnostics::_0_expected, scanner.tokenToString(SyntaxKind::ColonToken))
                    ),
                    pos
                );
            }

            auto parseBinaryExpressionOrHigher(OperatorPrecedence precedence) -> Expression {
                auto pos = getNodePos();
                auto leftOperand = parseUnaryExpressionOrHigher();
                return parseBinaryExpressionRest(precedence, leftOperand, pos);
            }

            auto isInOrOfKeyword(SyntaxKind t) -> boolean {
                return t == SyntaxKind::InKeyword || t == SyntaxKind::OfKeyword;
            }

            auto parseBinaryExpressionRest(OperatorPrecedence precedence, Expression leftOperand, number pos) -> Expression {
                while (true) {
                    // We either have a binary operator here, or we're finished.  We call
                    // reScanGreaterToken so that we merge token sequences like > and = into >=

                    reScanGreaterToken();
                    auto newPrecedence = getBinaryOperatorPrecedence(token());

                    // Check the precedence to see if we should "take" this operator
                    // - For left associative operator (all operator but **), consume the operator,
                    //   recursively call the auto below, and parse binaryExpression.as<a>() rightOperand
                    //   of the caller if the new precedence of the operator is greater then or equal to the current precedence.
                    //   For example:
                    //      a - b - c;
                    //            ^token; leftOperand = b. Return b to the caller.as<a>() rightOperand
                    //      a * b - c
                    //            ^token; leftOperand = b. Return b to the caller.as<a>() rightOperand
                    //      a - b * c;
                    //            ^token; leftOperand = b. Return b * c to the caller.as<a>() rightOperand
                    // - For right associative operator (**), consume the operator, recursively call the function
                    //   and parse binaryExpression.as<a>() rightOperand of the caller if the new precedence of
                    //   the operator is strictly grater than the current precedence
                    //   For example:
                    //      a ** b ** c;
                    //             ^^token; leftOperand = b. Return b ** c to the caller.as<a>() rightOperand
                    //      a - b ** c;
                    //            ^^token; leftOperand = b. Return b ** c to the caller.as<a>() rightOperand
                    //      a ** b - c
                    //             ^token; leftOperand = b. Return b to the caller.as<a>() rightOperand
                    auto consumeCurrentOperator = token() == SyntaxKind::AsteriskAsteriskToken ?
                        newPrecedence >= precedence :
                        newPrecedence > precedence;

                    if (!consumeCurrentOperator) {
                        break;
                    }

                    if (token() == SyntaxKind::InKeyword && inDisallowInContext()) {
                        break;
                    }

                    if (token() == SyntaxKind::AsKeyword) {
                        // Make sure we *do* perform ASI for constructs like this:
                        //    var x = foo
                        //    as (Bar)
                        // This should be parsed.as<an>() initialized variable, followed
                        // by a auto call to 'as' with the argument 'Bar'
                        if (scanner.hasPrecedingLineBreak()) {
                            break;
                        }
                        else {
                            nextToken();
                            leftOperand = makeAsExpression(leftOperand, parseType());
                        }
                    }
                    else {
                        leftOperand = makeBinaryExpression(leftOperand, parseTokenNode(), parseBinaryExpressionOrHigher(newPrecedence), pos);
                    }
                }

                return leftOperand;
            }

            auto isBinaryOperator() {
                if (inDisallowInContext() && token() == SyntaxKind::InKeyword) {
                    return false;
                }

                return getBinaryOperatorPrecedence(token()) > 0;
            }

            auto makeBinaryExpression(Expression left, BinaryOperatorToken operatorToken, Expression right, number pos) -> BinaryExpression {
                return finishNode(factory.createBinaryExpression(left, operatorToken, right), pos);
            }

            auto makeAsExpression(Expression left, TypeNode right) -> AsExpression {
                return finishNode(factory.createAsExpression(left, right), left.pos);
            }

            auto parsePrefixUnaryExpression() -> Node {
                auto pos = getNodePos();
                return finishNode(factory.createPrefixUnaryExpression(token().as<PrefixUnaryOperator>(), nextTokenAnd(parseSimpleUnaryExpression)), pos);
            }

            auto parseDeleteExpression() -> Node {
                auto pos = getNodePos();
                return finishNode(factory.createDeleteExpression(nextTokenAnd(parseSimpleUnaryExpression)), pos);
            }

            auto parseTypeOfExpression() -> Node {
                auto pos = getNodePos();
                return finishNode(factory.createTypeOfExpression(nextTokenAnd(parseSimpleUnaryExpression)), pos);
            }

            auto parseVoidExpression() -> Node {
                auto pos = getNodePos();
                return finishNode(factory.createVoidExpression(nextTokenAnd(parseSimpleUnaryExpression)), pos);
            }

            auto isAwaitExpression() -> boolean {
                if (token() == SyntaxKind::AwaitKeyword) {
                    if (inAwaitContext()) {
                        return true;
                    }

                    // here we are using similar heuristics as 'isYieldExpression'
                    return lookAhead(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOrLiteralOnSameLine, this));
                }

                return false;
            }

            auto parseAwaitExpression() -> Node {
                auto pos = getNodePos();
                return finishNode(factory.createAwaitExpression(nextTokenAnd(parseSimpleUnaryExpression)), pos);
            }

            /**
             * Parse ES7 exponential expression and await expression
             *
             * ES7 ExponentiationExpression:
             *      1) UnaryExpression[?Yield]
             *      2) UpdateExpression[?Yield] ** ExponentiationExpression[?Yield]
             *
             */
            auto parseUnaryExpressionOrHigher() -> Node {
                /**
                 * ES7 UpdateExpression:
                 *      1) LeftHandSideExpression[?Yield]
                 *      2) LeftHandSideExpression[?Yield][no LineTerminator here]++
                 *      3) LeftHandSideExpression[?Yield][no LineTerminator here]--
                 *      4) ++UnaryExpression[?Yield]
                 *      5) --UnaryExpression[?Yield]
                 */
                if (isUpdateExpression()) {
                    auto pos = getNodePos();
                    auto updateExpression = parseUpdateExpression();
                    return token() == SyntaxKind::AsteriskAsteriskToken ?
                        <BinaryExpression>parseBinaryExpressionRest(getBinaryOperatorPrecedence(token()), updateExpression, pos) :
                        updateExpression;
                }

                /**
                 * ES7 UnaryExpression:
                 *      1) UpdateExpression[?yield]
                 *      2) delete UpdateExpression[?yield]
                 *      3) void UpdateExpression[?yield]
                 *      4) typeof UpdateExpression[?yield]
                 *      5) + UpdateExpression[?yield]
                 *      6) - UpdateExpression[?yield]
                 *      7) ~ UpdateExpression[?yield]
                 *      8) ! UpdateExpression[?yield]
                 */
                auto unaryOperator = token();
                auto simpleUnaryExpression = parseSimpleUnaryExpression();
                if (token() == SyntaxKind::AsteriskAsteriskToken) {
                    auto pos = skipTrivia(sourceText, simpleUnaryExpression.pos);
                    auto { end } = simpleUnaryExpression;
                    if (simpleUnaryExpression.kind == SyntaxKind::TypeAssertionExpression) {
                        parseErrorAt(pos, end, Diagnostics::A_type_assertion_expression_is_not_allowed_in_the_left_hand_side_of_an_exponentiation_expression_Consider_enclosing_the_expression_in_parentheses);
                    }
                    else {
                        parseErrorAt(pos, end, Diagnostics::An_unary_expression_with_the_0_operator_is_not_allowed_in_the_left_hand_side_of_an_exponentiation_expression_Consider_enclosing_the_expression_in_parentheses, scanner.tokenToString(unaryOperator));
                    }
                }
                return simpleUnaryExpression;
            }

            /**
             * Parse ES7 simple-unary expression or higher:
             *
             * ES7 UnaryExpression:
             *      1) UpdateExpression[?yield]
             *      2) delete UnaryExpression[?yield]
             *      3) void UnaryExpression[?yield]
             *      4) typeof UnaryExpression[?yield]
             *      5) + UnaryExpression[?yield]
             *      6) - UnaryExpression[?yield]
             *      7) ~ UnaryExpression[?yield]
             *      8) ! UnaryExpression[?yield]
             *      9) [+Await] await UnaryExpression[?yield]
             */
            auto parseSimpleUnaryExpression() -> UnaryExpression {
                switch (token()) {
                    case SyntaxKind::PlusToken:
                    case SyntaxKind::MinusToken:
                    case SyntaxKind::TildeToken:
                    case SyntaxKind::ExclamationToken:
                        return parsePrefixUnaryExpression();
                    case SyntaxKind::DeleteKeyword:
                        return parseDeleteExpression();
                    case SyntaxKind::TypeOfKeyword:
                        return parseTypeOfExpression();
                    case SyntaxKind::VoidKeyword:
                        return parseVoidExpression();
                    case SyntaxKind::LessThanToken:
                        // This is modified UnaryExpression grammar in TypeScript
                        //  UnaryExpression (modified) ->
                        //      < type > UnaryExpression
                        return parseTypeAssertion();
                    case SyntaxKind::AwaitKeyword:
                        if (isAwaitExpression()) {
                            return parseAwaitExpression();
                        }
                        // falls through
                    default:
                        return parseUpdateExpression();
                }
            }

            /**
             * Check if the current token can possibly be an ES7 increment expression.
             *
             * ES7 UpdateExpression:
             *      LeftHandSideExpression[?Yield]
             *      LeftHandSideExpression[?Yield][no LineTerminator here]++
             *      LeftHandSideExpression[?Yield][no LineTerminator here]--
             *      ++LeftHandSideExpression[?Yield]
             *      --LeftHandSideExpression[?Yield]
             */
            auto isUpdateExpression() -> boolean {
                // This auto is called inside parseUnaryExpression to decide
                // whether to call parseSimpleUnaryExpression or call parseUpdateExpression directly
                switch (token()) {
                    case SyntaxKind::PlusToken:
                    case SyntaxKind::MinusToken:
                    case SyntaxKind::TildeToken:
                    case SyntaxKind::ExclamationToken:
                    case SyntaxKind::DeleteKeyword:
                    case SyntaxKind::TypeOfKeyword:
                    case SyntaxKind::VoidKeyword:
                    case SyntaxKind::AwaitKeyword:
                        return false;
                    case SyntaxKind::LessThanToken:
                        // If we are not in JSX context, we are parsing TypeAssertion which is an UnaryExpression
                        if (languageVariant != LanguageVariant::JSX) {
                            return false;
                        }
                        // We are in JSX context and the token is part of JSXElement.
                        // falls through
                    default:
                        return true;
                }
            }

            /**
             * Parse ES7 UpdateExpression. UpdateExpression is used instead of ES6's PostFixExpression.
             *
             * ES7 UpdateExpression[yield]:
             *      1) LeftHandSideExpression[?yield]
             *      2) LeftHandSideExpression[?yield] [[no LineTerminator here]]++
             *      3) LeftHandSideExpression[?yield] [[no LineTerminator here]]--
             *      4) ++LeftHandSideExpression[?yield]
             *      5) --LeftHandSideExpression[?yield]
             * In TypeScript (2), (3) are parsed.as<PostfixUnaryExpression>(). (4), (5) are parsed.as<PrefixUnaryExpression>()
             */
            auto parseUpdateExpression() -> UpdateExpression {
                if (token() == SyntaxKind::PlusPlusToken || token() == SyntaxKind::MinusMinusToken) {
                    auto pos = getNodePos();
                    return finishNode(factory.createPrefixUnaryExpression(token().as<PrefixUnaryOperator>(), nextTokenAnd(parseLeftHandSideExpressionOrHigher)), pos);
                }
                else if (languageVariant == LanguageVariant::JSX && token() == SyntaxKind::LessThanToken && lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOrGreaterThan, this))) {
                    // JSXElement is part of primaryExpression
                    return parseJsxElementOrSelfClosingElementOrFragment(/*inExpressionContext*/ true);
                }

                auto expression = parseLeftHandSideExpressionOrHigher();

                Debug::_assert(isLeftHandSideExpression(expression));
                if ((token() == SyntaxKind::PlusPlusToken || token() == SyntaxKind::MinusMinusToken) && !scanner.hasPrecedingLineBreak()) {
                    auto operator = token().as<PostfixUnaryOperator>();
                    nextToken();
                    return finishNode(factory.createPostfixUnaryExpression(expression, operator), expression.pos);
                }

                return expression;
            }

            auto parseLeftHandSideExpressionOrHigher() -> LeftHandSideExpression {
                // Original Ecma:
                // See LeftHandSideExpression 11.2
                //      NewExpression
                //      CallExpression
                //
                // Our simplification:
                //
                // See LeftHandSideExpression 11.2
                //      MemberExpression
                //      CallExpression
                //
                // See comment in parseMemberExpressionOrHigher on how we replaced NewExpression with
                // MemberExpression to make our lives easier.
                //
                // to best understand the below code, it's important to see how CallExpression expands
                // out into its own productions:
                //
                // CallExpression:
                //      MemberExpression Arguments
                //      CallExpression Arguments
                //      CallExpression[Expression]
                //      CallExpression.IdentifierName
                //      import (AssignmentExpression)
                //      super Arguments
                //      super.IdentifierName
                //
                // Because of the recursion in these calls, we need to bottom out first. There are three
                // bottom out states we can run 1 into) We see 'super' which must start either of
                // the last two CallExpression productions. 2) We see 'import' which must start import call.
                // 3)we have a MemberExpression which either completes the LeftHandSideExpression,
                // or starts the beginning of the first four CallExpression productions.
                auto pos = getNodePos();
                auto MemberExpression expression;
                if (token() == SyntaxKind::ImportKeyword) {
                    if (lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParenOrLessThan, this))) {
                        // We don't want to eagerly consume all import keyword.as<import>() call expression so we look ahead to find "("
                        // For example:
                        //      var foo3 = require("subfolder
                        //      import *.as<foo1>() from "module-from-node
                        // We want this import to be a statement rather than import call expression
                        sourceFlags |= NodeFlags::PossiblyContainsDynamicImport;
                        expression = parseTokenNode<PrimaryExpression>();
                    }
                    else if (lookAhead<boolean>(std::bind(&Parser::nextTokenIsDot, this))) {
                        // This is an 'import.*' metaproperty (i.e. 'import.meta')
                        nextToken(); // advance past the 'import'
                        nextToken(); // advance past the dot
                        expression = finishNode(factory.createMetaProperty(SyntaxKind::ImportKeyword, parseIdentifierName()), pos);
                        sourceFlags |= NodeFlags::PossiblyContainsImportMeta;
                    }
                    else {
                        expression = parseMemberExpressionOrHigher();
                    }
                }
                else {
                    expression = token() == SyntaxKind::SuperKeyword ? parseSuperExpression() : parseMemberExpressionOrHigher();
                }

                // Now, we *may* be complete.  However, we might have consumed the start of a
                // CallExpression or OptionalExpression.  As such, we need to consume the rest
                // of it here to be complete.
                return parseCallExpressionRest(pos, expression);
            }

            auto parseMemberExpressionOrHigher() -> MemberExpression {
                // to Note make our lives simpler, we decompose the NewExpression productions and
                // place ObjectCreationExpression and FunctionExpression into PrimaryExpression.
                // like so:
                //
                //   PrimaryExpression : See 11.1
                //      this
                //      Identifier
                //      Literal
                //      ArrayLiteral
                //      ObjectLiteral
                //      (Expression)
                //      FunctionExpression
                //      new MemberExpression Arguments?
                //
                //   MemberExpression : See 11.2
                //      PrimaryExpression
                //      MemberExpression[Expression]
                //      MemberExpression.IdentifierName
                //
                //   CallExpression : See 11.2
                //      MemberExpression
                //      CallExpression Arguments
                //      CallExpression[Expression]
                //      CallExpression.IdentifierName
                //
                // Technically this is ambiguous.  i.e. CallExpression defines:
                //
                //   CallExpression:
                //      CallExpression Arguments
                //
                // If you see: "new Foo()"
                //
                // Then that could be treated.as<a>() single ObjectCreationExpression, or it could be
                // treated.as<the>() invocation of "new Foo".  We disambiguate that in code (to match
                // the original grammar) by making sure that if we see an ObjectCreationExpression
                // we always consume arguments if they are there. So we treat "new Foo()".as<an>()
                // object creation only, and not at all.as<an>() invocation.  Another way to think
                // about this is that for every "new" that we see, we will consume an argument list if
                // it is there.as<part>() of the *associated* object creation node.  Any additional
                // argument lists we see, will become invocation expressions.
                //
                // Because there are no other places in the grammar now that refer to FunctionExpression
                // or ObjectCreationExpression, it is safe to push down into the PrimaryExpression
                // production.
                //
                // Because CallExpression and MemberExpression are left recursive, we need to bottom out
                // of the recursion immediately.  So we parse out a primary expression to start with.
                auto pos = getNodePos();
                auto expression = parsePrimaryExpression();
                return parseMemberExpressionRest(pos, expression, /*allowOptionalChain*/ true);
            }

            auto parseSuperExpression() -> MemberExpression {
                auto pos = getNodePos();
                auto expression = parseTokenNode<PrimaryExpression>();
                if (token() == SyntaxKind::LessThanToken) {
                    auto startPos = getNodePos();
                    auto typeArguments = tryParse<boolean>(std::bind(&Parser::parseTypeArgumentsInExpression, this));
                    if (typeArguments != undefined) {
                        parseErrorAt(startPos, getNodePos(), Diagnostics::super_may_not_use_type_arguments);
                    }
                }

                if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::DotToken || token() == SyntaxKind::OpenBracketToken) {
                    return expression;
                }

                // If we have seen "super" it must be followed by '(' or '.'.
                // If it wasn't then just try to parse out a '.' and report an error.
                parseExpectedToken(SyntaxKind::DotToken, Diagnostics::super_must_be_followed_by_an_argument_list_or_member_access);
                // private names will never work with `super` (`super.#foo`), but that's a semantic error, not syntactic
                return finishNode(factory.createPropertyAccessExpression(expression, parseRightSideOfDot(/*allowIdentifierNames*/ true, /*allowPrivateIdentifiers*/ true)), pos);
            }

            auto parseJsxElementOrSelfClosingElementOrFragment(boolean inExpressionContext, number topInvalidNodePosition) -> Node {
                auto pos = getNodePos();
                auto opening = parseJsxOpeningOrSelfClosingElementOrOpeningFragment(inExpressionContext);
                auto JsxElement result | JsxSelfClosingElement | JsxFragment;
                if (opening.kind == SyntaxKind::JsxOpeningElement) {
                    auto children = parseJsxChildren(opening);
                    auto closingElement = parseJsxClosingElement(inExpressionContext);

                    if (!tagNamesAreEquivalent(opening.tagName, closingElement.tagName)) {
                        parseErrorAtRange(closingElement, Diagnostics::Expected_corresponding_JSX_closing_tag_for_0, getTextOfNodeFromSourceText(sourceText, opening.tagName));
                    }

                    result = finishNode(factory.createJsxElement(opening, children, closingElement), pos);
                }
                else if (opening.kind == SyntaxKind::JsxOpeningFragment) {
                    result = finishNode(factory.createJsxFragment(opening, parseJsxChildren(opening), parseJsxClosingFragment(inExpressionContext)), pos);
                }
                else {
                    Debug::_assert(opening.kind == SyntaxKind::JsxSelfClosingElement);
                    // Nothing else to do for self-closing elements
                    result = opening;
                }

                // If the user writes the invalid code '<div></div><div></div>' in an expression context (i.e. not wrapped in
                // an enclosing tag), we'll naively try to parse   ^ this.as<a>() 'less than' operator and the remainder of the tag
                //.as<garbage>(), which will cause the formatter to badly mangle the JSX. Perform a speculative parse of a JSX
                // element if we see a < token so that we can wrap it in a synthetic binary expression so the formatter
                // does less damage and we can report a better error.
                // Since JSX elements are invalid < operands anyway, this lookahead parse will only occur in error scenarios
                // of one sort or another.
                if (inExpressionContext && token() == SyntaxKind::LessThanToken) {
                    auto topBadPos = typeof topInvalidNodePosition == "undefined" ? result.pos : topInvalidNodePosition;
                    auto invalidElement = tryParse(() => parseJsxElementOrSelfClosingElementOrFragment(/*inExpressionContext*/ true, topBadPos));
                    if (invalidElement) {
                        auto operatorToken = createMissingNode(SyntaxKind::CommaToken, /*reportAtCurrentPosition*/ false);
                        setTextRangePosWidth(operatorToken, invalidElement.pos, 0);
                        parseErrorAt(skipTrivia(sourceText, topBadPos), invalidElement.end, Diagnostics::JSX_expressions_must_have_one_parent_element);
                        return <JsxElement><Node>finishNode(factory.createBinaryExpression(result, operatorToken.as<Token>()<SyntaxKind::CommaToken>, invalidElement), pos);
                    }
                }

                return result;
            }

            auto parseJsxText() -> JsxText {
                auto pos = getNodePos();
                auto node = factory.createJsxText(scanner.getTokenValue(), currentToken == SyntaxKind::JsxTextAllWhiteSpaces);
                currentToken = scanner.scanJsxToken();
                return finishNode(node, pos);
            }

            auto parseJsxChild(SyntaxKind token) -> JsxChild {
                switch (token) {
                    case SyntaxKind::EndOfFileToken:
                        // If we hit EOF, issue the error at the tag that lacks the closing element
                        // rather than at the end of the file (which is useless)
                        if (isJsxOpeningFragment(openingTag)) {
                            parseErrorAtRange(openingTag, Diagnostics::JSX_fragment_has_no_corresponding_closing_tag);
                        }
                        else {
                            // We want the error span to cover only 'Foo.Bar' in < Foo.Bar >
                            // or to cover only 'Foo' in < Foo >
                            auto tag = openingTag.tagName;
                            auto start = skipTrivia(sourceText, tag.pos);
                            parseErrorAt(start, tag.end, Diagnostics::JSX_element_0_has_no_corresponding_closing_tag, getTextOfNodeFromSourceText(sourceText, openingTag.tagName));
                        }
                        return undefined;
                    case SyntaxKind::LessThanSlashToken:
                    case SyntaxKind::ConflictMarkerTrivia:
                        return undefined;
                    case SyntaxKind::JsxText:
                    case SyntaxKind::JsxTextAllWhiteSpaces:
                        return parseJsxText();
                    case SyntaxKind::OpenBraceToken:
                        return parseJsxExpression(/*inExpressionContext*/ false);
                    case SyntaxKind::LessThanToken:
                        return parseJsxElementOrSelfClosingElementOrFragment(/*inExpressionContext*/ false);
                    default:
                        return Debug::_assertNever(token);
                }
            }

            auto parseJsxChildren(Node openingTag) -> NodeArray<JsxChild> {
                auto list = [];
                auto listPos = getNodePos();
                auto saveParsingContext = parsingContext;
                parsingContext |= 1 << ParsingContext::JsxChildren;

                while (true) {
                    auto child = parseJsxChild(openingTag, currentToken = scanner.reScanJsxToken());
                    if (!child) break;
                    list.push_back(child);
                }

                parsingContext = saveParsingContext;
                return createNodeArray(list, listPos);
            }

            auto parseJsxAttributes() -> JsxAttributes {
                auto pos = getNodePos();
                return finishNode(factory.createJsxAttributes(parseList(ParsingContext::JsxAttributes, parseJsxAttribute)), pos);
            }

            auto parseJsxOpeningOrSelfClosingElementOrOpeningFragment(boolean inExpressionContext) -> Node {
                auto pos = getNodePos();

                parseExpected(SyntaxKind::LessThanToken);

                if (token() == SyntaxKind::GreaterThanToken) {
                    // See below for explanation of scanJsxText
                    scanJsxText();
                    return finishNode(factory.createJsxOpeningFragment(), pos);
                }

                auto tagName = parseJsxElementName();
                auto typeArguments = (contextFlags & NodeFlags::JavaScriptFile) == 0 ? tryParseTypeArguments() : undefined;
                auto attributes = parseJsxAttributes();

                auto JsxOpeningLikeElement node;

                if (token() == SyntaxKind::GreaterThanToken) {
                    // Closing tag, so scan the immediately-following text with the JSX scanning instead
                    // of regular scanning to avoid treating illegal characters (e.g. '#').as<immediate>()
                    // scanning errors
                    scanJsxText();
                    node = factory.createJsxOpeningElement(tagName, typeArguments, attributes);
                }
                else {
                    parseExpected(SyntaxKind::SlashToken);
                    if (inExpressionContext) {
                        parseExpected(SyntaxKind::GreaterThanToken);
                    }
                    else {
                        parseExpected(SyntaxKind::GreaterThanToken, /*diagnostic*/ undefined, /*shouldAdvance*/ false);
                        scanJsxText();
                    }
                    node = factory.createJsxSelfClosingElement(tagName, typeArguments, attributes);
                }

                return finishNode(node, pos);
            }

            auto parseJsxElementName() -> JsxTagNameExpression {
                auto pos = getNodePos();
                scanJsxIdentifier();
                // JsxElement can have name in the form of
                //      propertyAccessExpression
                //      primaryExpression in the form of an identifier and "this" keyword
                // We can't just simply use parseLeftHandSideExpressionOrHigher because then we will start consider class,auto etc.as<a>() keyword
                // We only want to consider "this".as<a>() primaryExpression
                auto JsxTagNameExpression expression = token() == SyntaxKind::ThisKeyword ?
                    parseTokenNode<ThisExpression>() : parseIdentifierName();
                while (parseOptional(SyntaxKind::DotToken)) {
                    expression = finishNode(factory.createPropertyAccessExpression(expression, parseRightSideOfDot(/*allowIdentifierNames*/ true, /*allowPrivateIdentifiers*/ false)), pos).as<JsxTagNamePropertyAccess>();
                }
                return expression;
            }

            auto parseJsxExpression(boolean inExpressionContext) -> JsxExpression {
                auto pos = getNodePos();
                if (!parseExpected(SyntaxKind::OpenBraceToken)) {
                    return undefined;
                }

                auto DotDotDotToken dotDotDotToken;
                auto Expression expression;
                if (token() != SyntaxKind::CloseBraceToken) {
                    dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
                    // Only an AssignmentExpression is valid here per the JSX spec,
                    // but we can unambiguously parse a comma sequence and provide
                    // a better error message in grammar checking.
                    expression = parseExpression();
                }
                if (inExpressionContext) {
                    parseExpected(SyntaxKind::CloseBraceToken);
                }
                else {
                    if (parseExpected(SyntaxKind::CloseBraceToken, /*message*/ undefined, /*shouldAdvance*/ false)) {
                        scanJsxText();
                    }
                }

                return finishNode(factory.createJsxExpression(dotDotDotToken, expression), pos);
            }

            auto parseJsxAttribute() -> Node {
                if (token() == SyntaxKind::OpenBraceToken) {
                    return parseJsxSpreadAttribute();
                }

                scanJsxIdentifier();
                auto pos = getNodePos();
                return finishNode(
                    factory.createJsxAttribute(
                        parseIdentifierName(),
                        token() != SyntaxKind::EqualsToken ? undefined :
                        scanJsxAttributeValue() == SyntaxKind::StringLiteral ? parseLiteralNode().as<StringLiteral>() :
                        parseJsxExpression(/*inExpressionContext*/ true)
                    ),
                    pos
                );
            }

            auto parseJsxSpreadAttribute() -> JsxSpreadAttribute {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::OpenBraceToken);
                parseExpected(SyntaxKind::DotDotDotToken);
                auto expression = parseExpression();
                parseExpected(SyntaxKind::CloseBraceToken);
                return finishNode(factory.createJsxSpreadAttribute(expression), pos);
            }

            auto parseJsxClosingElement(boolean inExpressionContext) -> JsxClosingElement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::LessThanSlashToken);
                auto tagName = parseJsxElementName();
                if (inExpressionContext) {
                    parseExpected(SyntaxKind::GreaterThanToken);
                }
                else {
                    parseExpected(SyntaxKind::GreaterThanToken, /*diagnostic*/ undefined, /*shouldAdvance*/ false);
                    scanJsxText();
                }
                return finishNode(factory.createJsxClosingElement(tagName), pos);
            }

            auto parseJsxClosingFragment(boolean inExpressionContext) -> JsxClosingFragment {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::LessThanSlashToken);
                if (scanner.tokenIsIdentifierOrKeyword(token())) {
                    parseErrorAtRange(parseJsxElementName(), Diagnostics::Expected_corresponding_closing_tag_for_JSX_fragment);
                }
                if (inExpressionContext) {
                    parseExpected(SyntaxKind::GreaterThanToken);
                }
                else {
                    parseExpected(SyntaxKind::GreaterThanToken, /*diagnostic*/ undefined, /*shouldAdvance*/ false);
                    scanJsxText();
                }
                return finishNode(factory.createJsxJsxClosingFragment(), pos);
            }

            auto parseTypeAssertion() -> TypeAssertion {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::LessThanToken);
                auto type = parseType();
                parseExpected(SyntaxKind::GreaterThanToken);
                auto expression = parseSimpleUnaryExpression();
                return finishNode(factory.createTypeAssertion(type, expression), pos);
            }

            auto nextTokenIsIdentifierOrKeywordOrOpenBracketOrTemplate() {
                nextToken();
                return scanner.tokenIsIdentifierOrKeyword(token())
                    || token() == SyntaxKind::OpenBracketToken
                    || isTemplateStartOfTaggedTemplate();
            }

            auto isStartOfOptionalPropertyOrElementAccessChain() {
                return token() == SyntaxKind::QuestionDotToken
                    && lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOrOpenBracketOrTemplate, this));
            }

            auto tryReparseOptionalChain(Expression node) {
                if (node->flags & NodeFlags::OptionalChain) {
                    return true;
                }
                // check for an optional chain in a non-null expression
                if (isNonNullExpression(node)) {
                    auto expr = node.expression;
                    while (isNonNullExpression(expr) && !(expr.flags & NodeFlags::OptionalChain)) {
                        expr = expr.expression;
                    }
                    if (expr.flags & NodeFlags::OptionalChain) {
                        // this is part of an optional chain. Walk down from `node` to `expression` and set the flag.
                        while (isNonNullExpression(node)) {
                            (node.as<Mutable>()<NonNullExpression>).flags |= NodeFlags::OptionalChain;
                            node = node.expression;
                        }
                        return true;
                    }
                }
                return false;
            }

            auto parsePropertyAccessExpressionRest(number pos, LeftHandSideExpression expression, QuestionDotToken questionDotToken) {
                auto name = parseRightSideOfDot(/*allowIdentifierNames*/ true, /*allowPrivateIdentifiers*/ true);
                auto isOptionalChain = questionDotToken || tryReparseOptionalChain(expression);
                auto propertyAccess = isOptionalChain ?
                    factory.createPropertyAccessChain(expression, questionDotToken, name) :
                    factory.createPropertyAccessExpression(expression, name);
                if (isOptionalChain && isPrivateIdentifier(propertyAccess.name)) {
                    parseErrorAtRange(propertyAccess.name, Diagnostics::An_optional_chain_cannot_contain_private_identifiers);
                }
                return finishNode(propertyAccess, pos);
            }

            auto parseElementAccessExpressionRest(number pos, LeftHandSideExpression expression, QuestionDotToken questionDotToken) {
                auto Expression argumentExpression;
                if (token() == SyntaxKind::CloseBracketToken) {
                    argumentExpression = createMissingNode(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ true, Diagnostics::An_element_access_expression_should_take_an_argument);
                }
                else {
                    auto argument = allowInAnd(parseExpression);
                    if (isStringOrNumericLiteralLike(argument)) {
                        argument.text = internIdentifier(argument.text);
                    }
                    argumentExpression = argument;
                }

                parseExpected(SyntaxKind::CloseBracketToken);

                auto indexedAccess = questionDotToken || tryReparseOptionalChain(expression) ?
                    factory.createElementAccessChain(expression, questionDotToken, argumentExpression) :
                    factory.createElementAccessExpression(expression, argumentExpression);
                return finishNode(indexedAccess, pos);
            }

            auto parseMemberExpressionRest(number pos, LeftHandSideExpression expression, boolean allowOptionalChain) -> MemberExpression {
                while (true) {
                    auto QuestionDotToken questionDotToken;
                    auto isPropertyAccess = false;
                    if (allowOptionalChain && isStartOfOptionalPropertyOrElementAccessChain()) {
                        questionDotToken = parseExpectedToken(SyntaxKind::QuestionDotToken);
                        isPropertyAccess = scanner.tokenIsIdentifierOrKeyword(token());
                    }
                    else {
                        isPropertyAccess = parseOptional(SyntaxKind::DotToken);
                    }

                    if (isPropertyAccess) {
                        expression = parsePropertyAccessExpressionRest(pos, expression, questionDotToken);
                        continue;
                    }

                    if (!questionDotToken && token() == SyntaxKind::ExclamationToken && !scanner.hasPrecedingLineBreak()) {
                        nextToken();
                        expression = finishNode(factory.createNonNullExpression(expression), pos);
                        continue;
                    }

                    // when in the [Decorator] context, we do not parse ElementAccess.as<it>() could be part of a ComputedPropertyName
                    if ((questionDotToken || !inDecoratorContext()) && parseOptional(SyntaxKind::OpenBracketToken)) {
                        expression = parseElementAccessExpressionRest(pos, expression, questionDotToken);
                        continue;
                    }

                    if (isTemplateStartOfTaggedTemplate()) {
                        expression = parseTaggedTemplateRest(pos, expression, questionDotToken, /*typeArguments*/ undefined);
                        continue;
                    }

                    return expression.as<MemberExpression>();
                }
            }

            auto isTemplateStartOfTaggedTemplate() {
                return token() == SyntaxKind::NoSubstitutionTemplateLiteral || token() == SyntaxKind::TemplateHead;
            }

            auto parseTaggedTemplateRest(number pos, LeftHandSideExpression tag, QuestionDotToken questionDotToken, NodeArray<TypeNode> typeArguments) {
                auto tagExpression = factory.createTaggedTemplateExpression(
                    tag,
                    typeArguments,
                    token() == SyntaxKind::NoSubstitutionTemplateLiteral ?
                        (reScanTemplateHeadOrNoSubstitutionTemplate(), parseLiteralNode().as<NoSubstitutionTemplateLiteral>()) :
                        parseTemplateExpression(/*isTaggedTemplate*/ true)
                );
                if (questionDotToken || tag.flags & NodeFlags::OptionalChain) {
                    (tagExpression.as<Mutable>()<Node>).flags |= NodeFlags::OptionalChain;
                }
                tagExpression.questionDotToken = questionDotToken;
                return finishNode(tagExpression, pos);
            }

            auto parseCallExpressionRest(number pos, LeftHandSideExpression expression) -> LeftHandSideExpression {
                while (true) {
                    expression = parseMemberExpressionRest(pos, expression, /*allowOptionalChain*/ true);
                    auto questionDotToken = parseOptionalToken(SyntaxKind::QuestionDotToken);
                    // handle 'foo<<T>()'
                    // parse template arguments only in TypeScript files (not in JavaScript files).
                    if ((contextFlags & NodeFlags::JavaScriptFile) == 0 && (token() == SyntaxKind::LessThanToken || token() == SyntaxKind::LessThanLessThanToken)) {
                        // See if this is the start of a generic invocation.  If so, consume it and
                        // keep checking for postfix expressions.  Otherwise, it's just a '<' that's
                        // part of an arithmetic expression.  Break out so we consume it higher in the
                        // stack.
                        auto typeArguments = tryParse<boolean>(std::bind(&Parser::parseTypeArgumentsInExpression, this));
                        if (typeArguments) {
                            if (isTemplateStartOfTaggedTemplate()) {
                                expression = parseTaggedTemplateRest(pos, expression, questionDotToken, typeArguments);
                                continue;
                            }

                            auto argumentList = parseArgumentList();
                            auto callExpr = questionDotToken || tryReparseOptionalChain(expression) ?
                                factory.createCallChain(expression, questionDotToken, typeArguments, argumentList) :
                                factory.createCallExpression(expression, typeArguments, argumentList);
                            expression = finishNode(callExpr, pos);
                            continue;
                        }
                    }
                    else if (token() == SyntaxKind::OpenParenToken) {
                        auto argumentList = parseArgumentList();
                        auto callExpr = questionDotToken || tryReparseOptionalChain(expression) ?
                            factory.createCallChain(expression, questionDotToken, /*typeArguments*/ undefined, argumentList) :
                            factory.createCallExpression(expression, /*typeArguments*/ undefined, argumentList);
                        expression = finishNode(callExpr, pos);
                        continue;
                    }
                    if (questionDotToken) {
                        // We failed to parse anything, so report a missing identifier here.
                        auto name = createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ false, Diagnostics::Identifier_expected);
                        expression = finishNode(factory.createPropertyAccessChain(expression, questionDotToken, name), pos);
                    }
                    break;
                }
                return expression;
            }

            auto parseArgumentList() {
                parseExpected(SyntaxKind::OpenParenToken);
                auto result = parseDelimitedList(ParsingContext::ArgumentExpressions, parseArgumentExpression);
                parseExpected(SyntaxKind::CloseParenToken);
                return result;
            }

            auto parseTypeArgumentsInExpression() {
                if ((contextFlags & NodeFlags::JavaScriptFile) != 0) {
                    // TypeArguments must not be parsed in JavaScript files to avoid ambiguity with binary operators.
                    return undefined;
                }

                if (reScanLessThanToken() != SyntaxKind::LessThanToken) {
                    return undefined;
                }
                nextToken();

                auto typeArguments = parseDelimitedList(ParsingContext::TypeArguments, parseType);
                if (!parseExpected(SyntaxKind::GreaterThanToken)) {
                    // If it doesn't have the closing `>` then it's definitely not an type argument list.
                    return undefined;
                }

                // If we have a '<', then only parse this.as<a>() argument list if the type arguments
                // are complete and we have an open paren.  if we don't, rewind and return nothing.
                return typeArguments && canFollowTypeArgumentsInExpression()
                    ? typeArguments
                    : undefined;
            }

            auto canFollowTypeArgumentsInExpression() -> boolean {
                switch (token()) {
                    case SyntaxKind::OpenParenToken:                 // foo<x>(
                    case SyntaxKind::NoSubstitutionTemplateLiteral:  // foo<T> `...`
                    case SyntaxKind::TemplateHead:                   // foo<T> `...${100}...`
                    // these are the only tokens can legally follow a type argument
                    // list. So we definitely want to treat them.as<type>() arg lists.
                    // falls through
                    case SyntaxKind::DotToken:                       // foo<x>.
                    case SyntaxKind::CloseParenToken:                // foo<x>)
                    case SyntaxKind::CloseBracketToken:              // foo<x>]
                    case SyntaxKind::ColonToken:                     // foo<x>:
                    case SyntaxKind::SemicolonToken:                 // foo<x>;
                    case SyntaxKind::QuestionToken:                  // foo<x>?
                    case SyntaxKind::EqualsEqualsToken:              // foo<x> ==
                    case SyntaxKind::EqualsEqualsEqualsToken:        // foo<x> ==
                    case SyntaxKind::ExclamationEqualsToken:         // foo<x> !=
                    case SyntaxKind::ExclamationEqualsEqualsToken:   // foo<x> !=
                    case SyntaxKind::AmpersandAmpersandToken:        // foo<x> &&
                    case SyntaxKind::BarBarToken:                    // foo<x> ||
                    case SyntaxKind::QuestionQuestionToken:          // foo<x> ??
                    case SyntaxKind::CaretToken:                     // foo<x> ^
                    case SyntaxKind::AmpersandToken:                 // foo<x> &
                    case SyntaxKind::BarToken:                       // foo<x> |
                    case SyntaxKind::CloseBraceToken:                // foo<x> }
                    case SyntaxKind::EndOfFileToken:                 // foo<x>
                        // these cases can't legally follow a type arg list.  However, they're not legal
                        // expressions either.  The user is probably in the middle of a generic type. So
                        // treat it.as<such>().
                        return true;

                    case SyntaxKind::CommaToken:                     // foo<x>,
                    case SyntaxKind::OpenBraceToken:                 // foo<x> {
                    // We don't want to treat these.as<type>() arguments.  Otherwise we'll parse this
                    //.as<an>() invocation expression.  Instead, we want to parse out the expression
                    // in isolation from the type arguments.
                    // falls through
                    default:
                        // Anything else treat.as<an>() expression.
                        return false;
                }
            }

            auto parsePrimaryExpression() -> PrimaryExpression {
                switch (token()) {
                    case SyntaxKind::NumericLiteral:
                    case SyntaxKind::BigIntLiteral:
                    case SyntaxKind::StringLiteral:
                    case SyntaxKind::NoSubstitutionTemplateLiteral:
                        return parseLiteralNode();
                    case SyntaxKind::ThisKeyword:
                    case SyntaxKind::SuperKeyword:
                    case SyntaxKind::NullKeyword:
                    case SyntaxKind::TrueKeyword:
                    case SyntaxKind::FalseKeyword:
                        return parseTokenNode<PrimaryExpression>();
                    case SyntaxKind::OpenParenToken:
                        return parseParenthesizedExpression();
                    case SyntaxKind::OpenBracketToken:
                        return parseArrayLiteralExpression();
                    case SyntaxKind::OpenBraceToken:
                        return parseObjectLiteralExpression();
                    case SyntaxKind::AsyncKeyword:
                        // Async arrow functions are parsed earlier in parseAssignmentExpressionOrHigher.
                        // If we encounter `async [no LineTerminator here] function` then this is an async
                        // function; otherwise, its an identifier.
                        if (!lookAhead<boolean>(std::bind(&Parser::nextTokenIsFunctionKeywordOnSameLine, this))) {
                            break;
                        }

                        return parseFunctionExpression();
                    case SyntaxKind::ClassKeyword:
                        return parseClassExpression();
                    case SyntaxKind::FunctionKeyword:
                        return parseFunctionExpression();
                    case SyntaxKind::NewKeyword:
                        return parseNewExpressionOrNewDotTarget();
                    case SyntaxKind::SlashToken:
                    case SyntaxKind::SlashEqualsToken:
                        if (reScanSlashToken() == SyntaxKind::RegularExpressionLiteral) {
                            return parseLiteralNode();
                        }
                        break;
                    case SyntaxKind::TemplateHead:
                        return parseTemplateExpression(/* isTaggedTemplate */ false);
                }

                return parseIdentifier(Diagnostics::Expression_expected);
            }

            auto parseParenthesizedExpression() -> ParenthesizedExpression {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                parseExpected(SyntaxKind::OpenParenToken);
                auto expression = allowInAnd(parseExpression);
                parseExpected(SyntaxKind::CloseParenToken);
                return withJSDoc(finishNode(factory.createParenthesizedExpression(expression), pos), hasJSDoc);
            }

            auto parseSpreadElement() -> Expression {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::DotDotDotToken);
                auto expression = parseAssignmentExpressionOrHigher();
                return finishNode(factory.createSpreadElement(expression), pos);
            }

            auto parseArgumentOrArrayLiteralElement() -> Expression {
                return token() == SyntaxKind::DotDotDotToken ? parseSpreadElement() :
                    token() == SyntaxKind::CommaToken ? finishNode(factory.createOmittedExpression(), getNodePos()) :
                    parseAssignmentExpressionOrHigher();
            }

            auto parseArgumentExpression() -> Expression {
                return doOutsideOfContext(disallowInAndDecoratorContext, parseArgumentOrArrayLiteralElement);
            }

            auto parseArrayLiteralExpression() -> Node {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::OpenBracketToken);
                auto multiLine = scanner.hasPrecedingLineBreak();
                auto elements = parseDelimitedList(ParsingContext::ArrayLiteralMembers, parseArgumentOrArrayLiteralElement);
                parseExpected(SyntaxKind::CloseBracketToken);
                return finishNode(factory.createArrayLiteralExpression(elements, multiLine), pos);
            }

            auto parseObjectLiteralElement() -> ObjectLiteralElementLike {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();

                if (parseOptionalToken(SyntaxKind::DotDotDotToken)) {
                    auto expression = parseAssignmentExpressionOrHigher();
                    return withJSDoc(finishNode(factory.createSpreadAssignment(expression), pos), hasJSDoc);
                }

                auto decorators = parseDecorators();
                auto modifiers = parseModifiers();

                if (parseContextualModifier(SyntaxKind::GetKeyword)) {
                    return parseAccessorDeclaration(pos, hasJSDoc, decorators, modifiers, SyntaxKind::GetAccessor);
                }
                if (parseContextualModifier(SyntaxKind::SetKeyword)) {
                    return parseAccessorDeclaration(pos, hasJSDoc, decorators, modifiers, SyntaxKind::SetAccessor);
                }

                auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
                auto tokenIsIdentifier = isIdentifier();
                auto name = parsePropertyName();

                // Disallowing of optional property assignments and definite assignment assertion happens in the grammar checker.
                auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
                auto exclamationToken = parseOptionalToken(SyntaxKind::ExclamationToken);

                if (asteriskToken || token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken) {
                    return parseMethodDeclaration(pos, hasJSDoc, decorators, modifiers, asteriskToken, name, questionToken, exclamationToken);
                }

                // check if it is short-hand property assignment or normal property assignment
                // if NOTE token is EqualsToken it is interpreted.as<CoverInitializedName>() production
                // CoverInitializedName[Yield] :
                //     IdentifierReference[?Yield] Initializer[In, ?Yield]
                // this is necessary because ObjectLiteral productions are also used to cover grammar for ObjectAssignmentPattern
                auto Mutable node<ShorthandPropertyAssignment | PropertyAssignment>;
                auto isShorthandPropertyAssignment = tokenIsIdentifier && (token() != SyntaxKind::ColonToken);
                if (isShorthandPropertyAssignment) {
                    auto equalsToken = parseOptionalToken(SyntaxKind::EqualsToken);
                    auto objectAssignmentInitializer = equalsToken ? allowInAnd(parseAssignmentExpressionOrHigher) : undefined;
                    node = factory.createShorthandPropertyAssignment(name.as<Identifier>(), objectAssignmentInitializer);
                    // Save equals token for error reporting.
                    // TODO(rbuckton) -> Consider manufacturing this when we need to report an error.as<it>() is otherwise not useful.
                    node.equalsToken = equalsToken;
                }
                else {
                    parseExpected(SyntaxKind::ColonToken);
                    auto initializer = allowInAnd(parseAssignmentExpressionOrHigher);
                    node = factory.createPropertyAssignment(name, initializer);
                }
                // Decorators, Modifiers, questionToken, and exclamationToken are not supported by property assignments and are reported in the grammar checker
                node.decorators = decorators;
                node.modifiers = modifiers;
                node.questionToken = questionToken;
                node.exclamationToken = exclamationToken;
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseObjectLiteralExpression() -> Node {
                auto pos = getNodePos();
                auto openBracePosition = scanner.getTokenPos();
                parseExpected(SyntaxKind::OpenBraceToken);
                auto multiLine = scanner.hasPrecedingLineBreak();
                auto properties = parseDelimitedList(ParsingContext::ObjectLiteralMembers, parseObjectLiteralElement, /*considerSemicolonAsDelimiter*/ true);
                if (!parseExpected(SyntaxKind::CloseBraceToken)) {
                    auto lastError = lastOrUndefined(parseDiagnostics);
                    if (lastError && lastError.code == Diagnostics::_0_expected.code) {
                        addRelatedInfo(
                            lastError,
                            createDetachedDiagnostic(fileName, openBracePosition, 1, Diagnostics::The_parser_expected_to_find_a_to_match_the_token_here)
                        );
                    }
                }
                return finishNode(factory.createObjectLiteralExpression(properties, multiLine), pos);
            }

            auto parseFunctionExpression() -> FunctionExpression {
                // GeneratorExpression:
                //      function* BindingIdentifier [Yield][opt](FormalParameters[Yield]){ GeneratorBody }
                //
                // FunctionExpression:
                //      auto BindingIdentifier[opt](FormalParameters){ FunctionBody }
                auto saveDecoratorContext = inDecoratorContext();
                if (saveDecoratorContext) {
                    setDecoratorContext(/*val*/ false);
                }

                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                auto modifiers = parseModifiers();
                parseExpected(SyntaxKind::FunctionKeyword);
                auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
                auto isGenerator = asteriskToken ? SignatureFlags.Yield : SignatureFlags.None;
                auto isAsync = some(modifiers, isAsyncModifier) ? SignatureFlags.Await : SignatureFlags.None;
                auto name =
                    isGenerator && isAsync ? doInYieldAndAwaitContext(parseOptionalBindingIdentifier) :
                    isGenerator ? doInYieldContext(parseOptionalBindingIdentifier) :
                    isAsync ? doInAwaitContext(parseOptionalBindingIdentifier) :
                    parseOptionalBindingIdentifier();

                auto typeParameters = parseTypeParameters();
                auto parameters = parseParameters(isGenerator | isAsync);
                auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
                auto body = parseFunctionBlock(isGenerator | isAsync);

                if (saveDecoratorContext) {
                    setDecoratorContext(/*val*/ true);
                }

                auto node = factory.createFunctionExpression(modifiers, asteriskToken, name, typeParameters, parameters, type, body);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseOptionalBindingIdentifier() -> Identifier {
                return isBindingIdentifier() ? parseBindingIdentifier() : undefined;
            }

            auto parseNewExpressionOrNewDotTarget() -> Node {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::NewKeyword);
                if (parseOptional(SyntaxKind::DotToken)) {
                    auto name = parseIdentifierName();
                    return finishNode(factory.createMetaProperty(SyntaxKind::NewKeyword, name), pos);
                }

                auto expressionPos = getNodePos();
                auto MemberExpression expression = parsePrimaryExpression();
                auto typeArguments;
                while (true) {
                    expression = parseMemberExpressionRest(expressionPos, expression, /*allowOptionalChain*/ false);
                    typeArguments = tryParse<boolean>(std::bind(&Parser::parseTypeArgumentsInExpression, this));
                    if (isTemplateStartOfTaggedTemplate()) {
                        Debug::_assert(!!typeArguments,
                            "Expected a type argument list; all plain tagged template starts should be consumed in 'parseMemberExpressionRest'");
                        expression = parseTaggedTemplateRest(expressionPos, expression, /*optionalChain*/ undefined, typeArguments);
                        typeArguments = undefined;
                    }
                    break;
                }

                auto NodeArray<Expression> argumentsArray;
                if (token() == SyntaxKind::OpenParenToken) {
                    argumentsArray = parseArgumentList();
                }
                else if (typeArguments) {
                    parseErrorAt(pos, scanner.getStartPos(), Diagnostics::A_new_expression_with_type_arguments_must_always_be_followed_by_a_parenthesized_argument_list);
                }
                return finishNode(factory.createNewExpression(expression, typeArguments, argumentsArray), pos);
            }

            // STATEMENTS
            auto parseBlock(boolean ignoreMissingOpenBrace, DiagnosticMessage diagnosticMessage) -> Block {
                auto pos = getNodePos();
                auto openBracePosition = scanner.getTokenPos();
                if (parseExpected(SyntaxKind::OpenBraceToken, diagnosticMessage) || ignoreMissingOpenBrace) {
                    auto multiLine = scanner.hasPrecedingLineBreak();
                    auto statements = parseList(ParsingContext::BlockStatements, parseStatement);
                    if (!parseExpected(SyntaxKind::CloseBraceToken)) {
                        auto lastError = lastOrUndefined(parseDiagnostics);
                        if (lastError && lastError.code == Diagnostics::_0_expected.code) {
                            addRelatedInfo(
                                lastError,
                                createDetachedDiagnostic(fileName, openBracePosition, 1, Diagnostics::The_parser_expected_to_find_a_to_match_the_token_here)
                            );
                        }
                    }
                    return finishNode(factory.createBlock(statements, multiLine), pos);
                }
                else {
                    auto statements = createMissingList<Statement>();
                    return finishNode(factory.createBlock(statements, /*multiLine*/ undefined), pos);
                }
            }

            auto parseFunctionBlock(SignatureFlags flags, DiagnosticMessage diagnosticMessage) -> Block {
                auto savedYieldContext = inYieldContext();
                setYieldContext(!!(flags & SignatureFlags.Yield));

                auto savedAwaitContext = inAwaitContext();
                setAwaitContext(!!(flags & SignatureFlags.Await));

                auto savedTopLevel = topLevel;
                topLevel = false;

                // We may be in a [Decorator] context when parsing a auto expression or
                // arrow function. The body of the auto is not in [Decorator] context.
                auto saveDecoratorContext = inDecoratorContext();
                if (saveDecoratorContext) {
                    setDecoratorContext(/*val*/ false);
                }

                auto block = parseBlock(!!(flags & SignatureFlags.IgnoreMissingOpenBrace), diagnosticMessage);

                if (saveDecoratorContext) {
                    setDecoratorContext(/*val*/ true);
                }

                topLevel = savedTopLevel;
                setYieldContext(savedYieldContext);
                setAwaitContext(savedAwaitContext);

                return block;
            }

            auto parseEmptyStatement() -> Statement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::SemicolonToken);
                return finishNode(factory.createEmptyStatement(), pos);
            }

            auto parseIfStatement() -> IfStatement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::IfKeyword);
                parseExpected(SyntaxKind::OpenParenToken);
                auto expression = allowInAnd(parseExpression);
                parseExpected(SyntaxKind::CloseParenToken);
                auto thenStatement = parseStatement();
                auto elseStatement = parseOptional(SyntaxKind::ElseKeyword) ? parseStatement() : undefined;
                return finishNode(factory.createIfStatement(expression, thenStatement, elseStatement), pos);
            }

            auto parseDoStatement() -> DoStatement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::DoKeyword);
                auto statement = parseStatement();
                parseExpected(SyntaxKind::WhileKeyword);
                parseExpected(SyntaxKind::OpenParenToken);
                auto expression = allowInAnd(parseExpression);
                parseExpected(SyntaxKind::CloseParenToken);

                // https From://mail.mozilla.org/pipermail/es-discuss/2011-August/016188.html
                // 157 min --- All allen at wirfs-brock.com CONF --- "do{;}while(false)false" prohibited in
                // spec but allowed in consensus reality. Approved -- this is the de-facto standard whereby
                //  do;while(0)x will have a semicolon inserted before x.
                parseOptional(SyntaxKind::SemicolonToken);
                return finishNode(factory.createDoStatement(statement, expression), pos);
            }

            auto parseWhileStatement() -> WhileStatement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::WhileKeyword);
                parseExpected(SyntaxKind::OpenParenToken);
                auto expression = allowInAnd(parseExpression);
                parseExpected(SyntaxKind::CloseParenToken);
                auto statement = parseStatement();
                return finishNode(factory.createWhileStatement(expression, statement), pos);
            }

            auto parseForOrForInOrForOfStatement() -> Statement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::ForKeyword);
                auto awaitToken = parseOptionalToken(SyntaxKind::AwaitKeyword);
                parseExpected(SyntaxKind::OpenParenToken);

                auto initializer!: VariableDeclarationList | Expression;
                if (token() != SyntaxKind::SemicolonToken) {
                    if (token() == SyntaxKind::VarKeyword || token() == SyntaxKind::LetKeyword || token() == SyntaxKind::ConstKeyword) {
                        initializer = parseVariableDeclarationList(/*inForStatementInitializer*/ true);
                    }
                    else {
                        initializer = disallowInAnd(parseExpression);
                    }
                }

                auto IterationStatement node;
                if (awaitToken ? parseExpected(SyntaxKind::OfKeyword) : parseOptional(SyntaxKind::OfKeyword)) {
                    auto expression = allowInAnd(parseAssignmentExpressionOrHigher);
                    parseExpected(SyntaxKind::CloseParenToken);
                    node = factory.createForOfStatement(awaitToken, initializer, expression, parseStatement());
                }
                else if (parseOptional(SyntaxKind::InKeyword)) {
                    auto expression = allowInAnd(parseExpression);
                    parseExpected(SyntaxKind::CloseParenToken);
                    node = factory.createForInStatement(initializer, expression, parseStatement());
                }
                else {
                    parseExpected(SyntaxKind::SemicolonToken);
                    auto condition = token() != SyntaxKind::SemicolonToken && token() != SyntaxKind::CloseParenToken
                        ? allowInAnd(parseExpression)
                        : undefined;
                    parseExpected(SyntaxKind::SemicolonToken);
                    auto incrementor = token() != SyntaxKind::CloseParenToken
                        ? allowInAnd(parseExpression)
                        : undefined;
                    parseExpected(SyntaxKind::CloseParenToken);
                    node = factory.createForStatement(initializer, condition, incrementor, parseStatement());
                }

                return finishNode(node, pos);
            }

            auto parseBreakOrContinueStatement(SyntaxKind kind) -> BreakOrContinueStatement {
                auto pos = getNodePos();

                parseExpected(kind == SyntaxKind::BreakStatement ? SyntaxKind::BreakKeyword : SyntaxKind::ContinueKeyword);
                auto label = canParseSemicolon() ? undefined : parseIdentifier();

                parseSemicolon();
                auto node = kind == SyntaxKind::BreakStatement
                    ? factory.createBreakStatement(label)
                    : factory.createContinueStatement(label);
                return finishNode(node, pos);
            }

            auto parseReturnStatement() -> ReturnStatement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::ReturnKeyword);
                auto expression = canParseSemicolon() ? undefined : allowInAnd(parseExpression);
                parseSemicolon();
                return finishNode(factory.createReturnStatement(expression), pos);
            }

            auto parseWithStatement() -> WithStatement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::WithKeyword);
                parseExpected(SyntaxKind::OpenParenToken);
                auto expression = allowInAnd(parseExpression);
                parseExpected(SyntaxKind::CloseParenToken);
                auto statement = doInsideOfContext(NodeFlags::InWithStatement, parseStatement);
                return finishNode(factory.createWithStatement(expression, statement), pos);
            }

            auto parseCaseClause() -> CaseClause {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::CaseKeyword);
                auto expression = allowInAnd(parseExpression);
                parseExpected(SyntaxKind::ColonToken);
                auto statements = parseList(ParsingContext::SwitchClauseStatements, parseStatement);
                return finishNode(factory.createCaseClause(expression, statements), pos);
            }

            auto parseDefaultClause() -> DefaultClause {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::DefaultKeyword);
                parseExpected(SyntaxKind::ColonToken);
                auto statements = parseList(ParsingContext::SwitchClauseStatements, parseStatement);
                return finishNode(factory.createDefaultClause(statements), pos);
            }

            auto parseCaseOrDefaultClause() -> CaseOrDefaultClause {
                return token() == SyntaxKind::CaseKeyword ? parseCaseClause() : parseDefaultClause();
            }

            auto parseCaseBlock() -> CaseBlock {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::OpenBraceToken);
                auto clauses = parseList(ParsingContext::SwitchClauses, parseCaseOrDefaultClause);
                parseExpected(SyntaxKind::CloseBraceToken);
                return finishNode(factory.createCaseBlock(clauses), pos);
            }

            auto parseSwitchStatement() -> SwitchStatement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::SwitchKeyword);
                parseExpected(SyntaxKind::OpenParenToken);
                auto expression = allowInAnd(parseExpression);
                parseExpected(SyntaxKind::CloseParenToken);
                auto caseBlock = parseCaseBlock();
                return finishNode(factory.createSwitchStatement(expression, caseBlock), pos);
            }

            auto parseThrowStatement() -> ThrowStatement {
                // ThrowStatement[Yield] :
                //      throw [no LineTerminator here]Expression[In, ?Yield];

                auto pos = getNodePos();
                parseExpected(SyntaxKind::ThrowKeyword);

                // Because of automatic semicolon insertion, we need to report error if this
                // throw could be terminated with a semicolon.  we Note can't call 'parseExpression'
                // directly.as<that>() might consume an expression on the following line.
                // Instead, we create a "missing" identifier, but don't report an error. The actual error
                // will be reported in the grammar walker.
                auto expression = scanner.hasPrecedingLineBreak() ? undefined : allowInAnd(parseExpression);
                if (expression == undefined) {
                    identifierCount++;
                    expression = finishNode(factory.createIdentifier(string()), getNodePos());
                }
                parseSemicolon();
                return finishNode(factory.createThrowStatement(expression), pos);
            }

            // Review TODO for error recovery
            auto parseTryStatement() -> TryStatement {
                auto pos = getNodePos();

                parseExpected(SyntaxKind::TryKeyword);
                auto tryBlock = parseBlock(/*ignoreMissingOpenBrace*/ false);
                auto catchClause = token() == SyntaxKind::CatchKeyword ? parseCatchClause() : undefined;

                // If we don't have a catch clause, then we must have a finally clause.  Try to parse
                // one out no matter what.
                auto Block finallyBlock;
                if (!catchClause || token() == SyntaxKind::FinallyKeyword) {
                    parseExpected(SyntaxKind::FinallyKeyword);
                    finallyBlock = parseBlock(/*ignoreMissingOpenBrace*/ false);
                }

                return finishNode(factory.createTryStatement(tryBlock, catchClause, finallyBlock), pos);
            }

            auto parseCatchClause() -> CatchClause {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::CatchKeyword);

                auto variableDeclaration;
                if (parseOptional(SyntaxKind::OpenParenToken)) {
                    variableDeclaration = parseVariableDeclaration();
                    parseExpected(SyntaxKind::CloseParenToken);
                }
                else {
                    // Keep shape of node to avoid degrading performance.
                    variableDeclaration = undefined;
                }

                auto block = parseBlock(/*ignoreMissingOpenBrace*/ false);
                return finishNode(factory.createCatchClause(variableDeclaration, block), pos);
            }

            auto parseDebuggerStatement() -> Statement {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::DebuggerKeyword);
                parseSemicolon();
                return finishNode(factory.createDebuggerStatement(), pos);
            }

            auto parseExpressionOrLabeledStatement() -> Node {
                // Avoiding having to do the lookahead for a labeled statement by just trying to parse
                // out an expression, seeing if it is identifier and then seeing if it is followed by
                // a colon.
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                auto ExpressionStatement node | LabeledStatement;
                auto hasParen = token() == SyntaxKind::OpenParenToken;
                auto expression = allowInAnd(parseExpression);
                if (ts.isIdentifier(expression) && parseOptional(SyntaxKind::ColonToken)) {
                    node = factory.createLabeledStatement(expression, parseStatement());
                }
                else {
                    parseSemicolon();
                    node = factory.createExpressionStatement(expression);
                    if (hasParen) {
                        // do not parse the same jsdoc twice
                        hasJSDoc = false;
                    }
                }
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto nextTokenIsIdentifierOrKeywordOnSameLine() -> boolean {
                nextToken();
                return scanner.tokenIsIdentifierOrKeyword(token()) && !scanner.hasPrecedingLineBreak();
            }

            auto nextTokenIsClassKeywordOnSameLine() -> boolean {
                nextToken();
                return token() == SyntaxKind::ClassKeyword && !scanner.hasPrecedingLineBreak();
            }

            auto nextTokenIsFunctionKeywordOnSameLine() -> boolean {
                nextToken();
                return token() == SyntaxKind::FunctionKeyword && !scanner.hasPrecedingLineBreak();
            }

            auto nextTokenIsIdentifierOrKeywordOrLiteralOnSameLine() {
                nextToken();
                return (scanner.tokenIsIdentifierOrKeyword(token()) || token() == SyntaxKind::NumericLiteral || token() == SyntaxKind::BigIntLiteral || token() == SyntaxKind::StringLiteral) && !scanner.hasPrecedingLineBreak();
            }

            auto isDeclaration() -> boolean {
                while (true) {
                    switch (token()) {
                        case SyntaxKind::VarKeyword:
                        case SyntaxKind::LetKeyword:
                        case SyntaxKind::ConstKeyword:
                        case SyntaxKind::FunctionKeyword:
                        case SyntaxKind::ClassKeyword:
                        case SyntaxKind::EnumKeyword:
                            return true;

                        // 'declare', 'module', 'namespace', 'interface'* and 'type' are all legal JavaScript identifiers;
                        // however, an identifier cannot be followed by another identifier on the same line. This is what we
                        // count on to parse out the respective declarations. For instance, we exploit this to say that
                        //
                        //    namespace n
                        //
                        // can be none other than the beginning of a namespace declaration, but need to respect that JavaScript sees
                        //
                        //    namespace
                        //    n
                        //
                        //.as<the>() identifier 'namespace' on one line followed by the identifier 'n' on another.
                        // We need to look one token ahead to see if it permissible to try parsing a declaration.
                        //
                        // *Note*: 'interface' is actually a strict mode reserved word. So while
                        //
                        //   "use strict"
                        //   interface
                        //   I {}
                        //
                        // could be legal, it would add complexity for very little gain.
                        case SyntaxKind::InterfaceKeyword:
                        case SyntaxKind::TypeKeyword:
                            return nextTokenIsIdentifierOnSameLine();
                        case SyntaxKind::ModuleKeyword:
                        case SyntaxKind::NamespaceKeyword:
                            return nextTokenIsIdentifierOrStringLiteralOnSameLine();
                        case SyntaxKind::AbstractKeyword:
                        case SyntaxKind::AsyncKeyword:
                        case SyntaxKind::DeclareKeyword:
                        case SyntaxKind::PrivateKeyword:
                        case SyntaxKind::ProtectedKeyword:
                        case SyntaxKind::PublicKeyword:
                        case SyntaxKind::ReadonlyKeyword:
                            nextToken();
                            // ASI takes effect for this modifier.
                            if (scanner.hasPrecedingLineBreak()) {
                                return false;
                            }
                            continue;

                        case SyntaxKind::GlobalKeyword:
                            nextToken();
                            return token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::Identifier || token() == SyntaxKind::ExportKeyword;

                        case SyntaxKind::ImportKeyword:
                            nextToken();
                            return token() == SyntaxKind::StringLiteral || token() == SyntaxKind::AsteriskToken ||
                                token() == SyntaxKind::OpenBraceToken || scanner.tokenIsIdentifierOrKeyword(token());
                        case SyntaxKind::ExportKeyword:
                            auto currentToken = nextToken();
                            if (currentToken == SyntaxKind::TypeKeyword) {
                                currentToken = lookAhead<boolean>(std::bind(&Parser::nextToken, this));
                            }
                            if (currentToken == SyntaxKind::EqualsToken || currentToken == SyntaxKind::AsteriskToken ||
                                currentToken == SyntaxKind::OpenBraceToken || currentToken == SyntaxKind::DefaultKeyword ||
                                currentToken == SyntaxKind::AsKeyword) {
                                return true;
                            }
                            continue;

                        case SyntaxKind::StaticKeyword:
                            nextToken();
                            continue;
                        default:
                            return false;
                    }
                }
            }

            auto isStartOfDeclaration() -> boolean {
                return lookAhead<boolean>(std::bind(&Parser::isDeclaration, this));
            }

            auto isStartOfStatement() -> boolean {
                switch (token()) {
                    case SyntaxKind::AtToken:
                    case SyntaxKind::SemicolonToken:
                    case SyntaxKind::OpenBraceToken:
                    case SyntaxKind::VarKeyword:
                    case SyntaxKind::LetKeyword:
                    case SyntaxKind::FunctionKeyword:
                    case SyntaxKind::ClassKeyword:
                    case SyntaxKind::EnumKeyword:
                    case SyntaxKind::IfKeyword:
                    case SyntaxKind::DoKeyword:
                    case SyntaxKind::WhileKeyword:
                    case SyntaxKind::ForKeyword:
                    case SyntaxKind::ContinueKeyword:
                    case SyntaxKind::BreakKeyword:
                    case SyntaxKind::ReturnKeyword:
                    case SyntaxKind::WithKeyword:
                    case SyntaxKind::SwitchKeyword:
                    case SyntaxKind::ThrowKeyword:
                    case SyntaxKind::TryKeyword:
                    case SyntaxKind::DebuggerKeyword:
                    // 'catch' and 'finally' do not actually indicate that the code is part of a statement,
                    // however, we say they are here so that we may gracefully parse them and error later.
                    // falls through
                    case SyntaxKind::CatchKeyword:
                    case SyntaxKind::FinallyKeyword:
                        return true;

                    case SyntaxKind::ImportKeyword:
                        return isStartOfDeclaration() || lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParenOrLessThanOrDot, this));

                    case SyntaxKind::ConstKeyword:
                    case SyntaxKind::ExportKeyword:
                        return isStartOfDeclaration();

                    case SyntaxKind::AsyncKeyword:
                    case SyntaxKind::DeclareKeyword:
                    case SyntaxKind::InterfaceKeyword:
                    case SyntaxKind::ModuleKeyword:
                    case SyntaxKind::NamespaceKeyword:
                    case SyntaxKind::TypeKeyword:
                    case SyntaxKind::GlobalKeyword:
                        // When these don't start a declaration, they're an identifier in an expression statement
                        return true;

                    case SyntaxKind::PublicKeyword:
                    case SyntaxKind::PrivateKeyword:
                    case SyntaxKind::ProtectedKeyword:
                    case SyntaxKind::StaticKeyword:
                    case SyntaxKind::ReadonlyKeyword:
                        // When these don't start a declaration, they may be the start of a class member if an identifier
                        // immediately follows. Otherwise they're an identifier in an expression statement.
                        return isStartOfDeclaration() || !lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOnSameLine, this));

                    default:
                        return isStartOfExpression();
                }
            }

            auto nextTokenIsIdentifierOrStartOfDestructuring() {
                nextToken();
                return isIdentifier() || token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::OpenBracketToken;
            }

            auto isLetDeclaration() {
                // In ES6 'let' always starts a lexical declaration if followed by an identifier or {
                // or [.
                return lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrStartOfDestructuring, this));
            }

            auto parseStatement() -> Statement {
                switch (token()) {
                    case SyntaxKind::SemicolonToken:
                        return parseEmptyStatement();
                    case SyntaxKind::OpenBraceToken:
                        return parseBlock(/*ignoreMissingOpenBrace*/ false);
                    case SyntaxKind::VarKeyword:
                        return parseVariableStatement(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined);
                    case SyntaxKind::LetKeyword:
                        if (isLetDeclaration()) {
                            return parseVariableStatement(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined);
                        }
                        break;
                    case SyntaxKind::FunctionKeyword:
                        return parseFunctionDeclaration(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined);
                    case SyntaxKind::ClassKeyword:
                        return parseClassDeclaration(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined);
                    case SyntaxKind::IfKeyword:
                        return parseIfStatement();
                    case SyntaxKind::DoKeyword:
                        return parseDoStatement();
                    case SyntaxKind::WhileKeyword:
                        return parseWhileStatement();
                    case SyntaxKind::ForKeyword:
                        return parseForOrForInOrForOfStatement();
                    case SyntaxKind::ContinueKeyword:
                        return parseBreakOrContinueStatement(SyntaxKind::ContinueStatement);
                    case SyntaxKind::BreakKeyword:
                        return parseBreakOrContinueStatement(SyntaxKind::BreakStatement);
                    case SyntaxKind::ReturnKeyword:
                        return parseReturnStatement();
                    case SyntaxKind::WithKeyword:
                        return parseWithStatement();
                    case SyntaxKind::SwitchKeyword:
                        return parseSwitchStatement();
                    case SyntaxKind::ThrowKeyword:
                        return parseThrowStatement();
                    case SyntaxKind::TryKeyword:
                    // Include 'catch' and 'finally' for error recovery.
                    // falls through
                    case SyntaxKind::CatchKeyword:
                    case SyntaxKind::FinallyKeyword:
                        return parseTryStatement();
                    case SyntaxKind::DebuggerKeyword:
                        return parseDebuggerStatement();
                    case SyntaxKind::AtToken:
                        return parseDeclaration();
                    case SyntaxKind::AsyncKeyword:
                    case SyntaxKind::InterfaceKeyword:
                    case SyntaxKind::TypeKeyword:
                    case SyntaxKind::ModuleKeyword:
                    case SyntaxKind::NamespaceKeyword:
                    case SyntaxKind::DeclareKeyword:
                    case SyntaxKind::ConstKeyword:
                    case SyntaxKind::EnumKeyword:
                    case SyntaxKind::ExportKeyword:
                    case SyntaxKind::ImportKeyword:
                    case SyntaxKind::PrivateKeyword:
                    case SyntaxKind::ProtectedKeyword:
                    case SyntaxKind::PublicKeyword:
                    case SyntaxKind::AbstractKeyword:
                    case SyntaxKind::StaticKeyword:
                    case SyntaxKind::ReadonlyKeyword:
                    case SyntaxKind::GlobalKeyword:
                        if (isStartOfDeclaration()) {
                            return parseDeclaration();
                        }
                        break;
                }
                return parseExpressionOrLabeledStatement();
            }

            auto isDeclareModifier(Modifier modifier) {
                return modifier.kind == SyntaxKind::DeclareKeyword;
            }

            auto parseDeclaration() -> Statement {
                // Can TODO we hold onto the parsed decorators/modifiers and advance the scanner
                //       if we can't reuse the declaration, so that we don't do this work twice?
                //
                // `parseListElement` attempted to get the reused node at this position,
                // but the ambient context flag was not yet set, so the node appeared
                // not reusable in that context.
                auto isAmbient = some(lookAhead(() => (parseDecorators(), parseModifiers())), isDeclareModifier);
                if (isAmbient) {
                    auto node = tryReuseAmbientDeclaration();
                    if (node) {
                        return node;
                    }
                }

                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                auto decorators = parseDecorators();
                auto modifiers = parseModifiers();
                if (isAmbient) {
                    for (auto m of modifiers!) {
                        (m.as<Mutable>()<Node>).flags |= NodeFlags::Ambient;
                    }
                    return doInsideOfContext(NodeFlags::Ambient, () => parseDeclarationWorker(pos, hasJSDoc, decorators, modifiers));
                }
                else {
                    return parseDeclarationWorker(pos, hasJSDoc, decorators, modifiers);
                }
            }

            auto tryReuseAmbientDeclaration() -> Statement {
                return doInsideOfContext(NodeFlags::Ambient, () => {
                    auto node = currentNode(parsingContext);
                    if (node) {
                        return consumeNode(node).as<Statement>();
                    }
                });
            }

            auto parseDeclarationWorker(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> Statement {
                switch (token()) {
                    case SyntaxKind::VarKeyword:
                    case SyntaxKind::LetKeyword:
                    case SyntaxKind::ConstKeyword:
                        return parseVariableStatement(pos, hasJSDoc, decorators, modifiers);
                    case SyntaxKind::FunctionKeyword:
                        return parseFunctionDeclaration(pos, hasJSDoc, decorators, modifiers);
                    case SyntaxKind::ClassKeyword:
                        return parseClassDeclaration(pos, hasJSDoc, decorators, modifiers);
                    case SyntaxKind::InterfaceKeyword:
                        return parseInterfaceDeclaration(pos, hasJSDoc, decorators, modifiers);
                    case SyntaxKind::TypeKeyword:
                        return parseTypeAliasDeclaration(pos, hasJSDoc, decorators, modifiers);
                    case SyntaxKind::EnumKeyword:
                        return parseEnumDeclaration(pos, hasJSDoc, decorators, modifiers);
                    case SyntaxKind::GlobalKeyword:
                    case SyntaxKind::ModuleKeyword:
                    case SyntaxKind::NamespaceKeyword:
                        return parseModuleDeclaration(pos, hasJSDoc, decorators, modifiers);
                    case SyntaxKind::ImportKeyword:
                        return parseImportDeclarationOrImportEqualsDeclaration(pos, hasJSDoc, decorators, modifiers);
                    case SyntaxKind::ExportKeyword:
                        nextToken();
                        switch (token()) {
                            case SyntaxKind::DefaultKeyword:
                            case SyntaxKind::EqualsToken:
                                return parseExportAssignment(pos, hasJSDoc, decorators, modifiers);
                            case SyntaxKind::AsKeyword:
                                return parseNamespaceExportDeclaration(pos, hasJSDoc, decorators, modifiers);
                            default:
                                return parseExportDeclaration(pos, hasJSDoc, decorators, modifiers);
                        }
                    default:
                        if (decorators || modifiers) {
                            // We reached this point because we encountered decorators and/or modifiers and assumed a declaration
                            // would follow. For recovery and error reporting purposes, return an incomplete declaration.
                            auto missing = createMissingNode<MissingDeclaration>(SyntaxKind::MissingDeclaration, /*reportAtCurrentPosition*/ true, Diagnostics::Declaration_expected);
                            setTextRangePos(missing, pos);
                            missing.decorators = decorators;
                            missing.modifiers = modifiers;
                            return missing;
                        }
                        return undefined!; // GH TODO#18217
                }
            }

            auto nextTokenIsIdentifierOrStringLiteralOnSameLine() {
                nextToken();
                return !scanner.hasPrecedingLineBreak() && (isIdentifier() || token() == SyntaxKind::StringLiteral);
            }

            auto parseFunctionBlockOrSemicolon(SignatureFlags flags, DiagnosticMessage diagnosticMessage) -> Block {
                if (token() != SyntaxKind::OpenBraceToken && canParseSemicolon()) {
                    parseSemicolon();
                    return;
                }

                return parseFunctionBlock(flags, diagnosticMessage);
            }

            // DECLARATIONS

            auto parseArrayBindingElement() -> ArrayBindingElement {
                auto pos = getNodePos();
                if (token() == SyntaxKind::CommaToken) {
                    return finishNode(factory.createOmittedExpression(), pos);
                }
                auto dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
                auto name = parseIdentifierOrPattern();
                auto initializer = parseInitializer();
                return finishNode(factory.createBindingElement(dotDotDotToken, /*propertyName*/ undefined, name, initializer), pos);
            }

            auto parseObjectBindingElement() -> BindingElement {
                auto pos = getNodePos();
                auto dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
                auto tokenIsIdentifier = isBindingIdentifier();
                auto PropertyName propertyName = parsePropertyName();
                auto BindingName name;
                if (tokenIsIdentifier && token() != SyntaxKind::ColonToken) {
                    name = propertyName.as<Identifier>();
                    propertyName = undefined;
                }
                else {
                    parseExpected(SyntaxKind::ColonToken);
                    name = parseIdentifierOrPattern();
                }
                auto initializer = parseInitializer();
                return finishNode(factory.createBindingElement(dotDotDotToken, propertyName, name, initializer), pos);
            }

            auto parseObjectBindingPattern() -> ObjectBindingPattern {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::OpenBraceToken);
                auto elements = parseDelimitedList(ParsingContext::ObjectBindingElements, parseObjectBindingElement);
                parseExpected(SyntaxKind::CloseBraceToken);
                return finishNode(factory.createObjectBindingPattern(elements), pos);
            }

            auto parseArrayBindingPattern() -> ArrayBindingPattern {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::OpenBracketToken);
                auto elements = parseDelimitedList(ParsingContext::ArrayBindingElements, parseArrayBindingElement);
                parseExpected(SyntaxKind::CloseBracketToken);
                return finishNode(factory.createArrayBindingPattern(elements), pos);
            }

            auto isBindingIdentifierOrPrivateIdentifierOrPattern() -> boolean {
                return token() == SyntaxKind::OpenBraceToken
                    || token() == SyntaxKind::OpenBracketToken
                    || token() == SyntaxKind::PrivateIdentifier
                    || isBindingIdentifier();
            }

            auto parseIdentifierOrPattern(DiagnosticMessage privateIdentifierDiagnosticMessage) -> Node {
                if (token() == SyntaxKind::OpenBracketToken) {
                    return parseArrayBindingPattern();
                }
                if (token() == SyntaxKind::OpenBraceToken) {
                    return parseObjectBindingPattern();
                }
                return parseBindingIdentifier(privateIdentifierDiagnosticMessage);
            }

            auto parseVariableDeclarationAllowExclamation() {
                return parseVariableDeclaration(/*allowExclamation*/ true);
            }

            auto parseVariableDeclaration(boolean allowExclamation) -> VariableDeclaration {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                auto name = parseIdentifierOrPattern(Diagnostics::Private_identifiers_are_not_allowed_in_variable_declarations);
                auto ExclamationToken exclamationToken;
                if (allowExclamation && name.kind == SyntaxKind::Identifier &&
                    token() == SyntaxKind::ExclamationToken && !scanner.hasPrecedingLineBreak()) {
                    exclamationToken = parseTokenNode<Token<SyntaxKind::ExclamationToken>>();
                }
                auto type = parseTypeAnnotation();
                auto initializer = isInOrOfKeyword(token()) ? undefined : parseInitializer();
                auto node = factory.createVariableDeclaration(name, exclamationToken, type, initializer);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseVariableDeclarationList(boolean inForStatementInitializer) -> VariableDeclarationList {
                auto pos = getNodePos();

                auto NodeFlags flags = 0;
                switch (token()) {
                    case SyntaxKind::VarKeyword:
                        break;
                    case SyntaxKind::LetKeyword:
                        flags |= NodeFlags::Let;
                        break;
                    case SyntaxKind::ConstKeyword:
                        flags |= NodeFlags::Const;
                        break;
                    default:
                        Debug::fail();
                }

                nextToken();

                // The user may have written the following:
                //
                //    for (auto of X) { }
                //
                // In this case, we want to parse an empty declaration list, and then parse 'of'
                //.as<a>() keyword. The reason this is not automatic is that 'of' is a valid identifier.
                // So we need to look ahead to determine if 'of' should be treated.as<a>() keyword in
                // this context.
                // The checker will then give an error that there is an empty declaration list.
                auto declarations;
                if (token() == SyntaxKind::OfKeyword && lookAhead<boolean>(std::bind(&Parser::canFollowContextualOfKeyword, this))) {
                    declarations = createMissingList<VariableDeclaration>();
                }
                else {
                    auto savedDisallowIn = inDisallowInContext();
                    setDisallowInContext(inForStatementInitializer);

                    declarations = parseDelimitedList(ParsingContext::VariableDeclarations,
                        inForStatementInitializer ? parseVariableDeclaration : parseVariableDeclarationAllowExclamation);

                    setDisallowInContext(savedDisallowIn);
                }

                return finishNode(factory.createVariableDeclarationList(declarations, flags), pos);
            }

            auto canFollowContextualOfKeyword() -> boolean {
                return nextTokenIsIdentifier() && nextToken() == SyntaxKind::CloseParenToken;
            }

            auto parseVariableStatement(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> VariableStatement {
                auto declarationList = parseVariableDeclarationList(/*inForStatementInitializer*/ false);
                parseSemicolon();
                auto node = factory.createVariableStatement(modifiers, declarationList);
                // Decorators are not allowed on a variable statement, so we keep track of them to report them in the grammar checker.
                node.decorators = decorators;
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseFunctionDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> FunctionDeclaration {
                auto savedAwaitContext = inAwaitContext();
                auto modifierFlags = modifiersToFlags(modifiers);
                parseExpected(SyntaxKind::FunctionKeyword);
                auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
                // We don't parse the name here in await context, instead we will report a grammar error in the checker.
                auto name = modifierFlags & ModifierFlags.Default ? parseOptionalBindingIdentifier() : parseBindingIdentifier();
                auto isGenerator = asteriskToken ? SignatureFlags.Yield : SignatureFlags.None;
                auto isAsync = modifierFlags & ModifierFlags.Async ? SignatureFlags.Await : SignatureFlags.None;
                auto typeParameters = parseTypeParameters();
                if (modifierFlags & ModifierFlags.Export) setAwaitContext(/*value*/ true);
                auto parameters = parseParameters(isGenerator | isAsync);
                auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
                auto body = parseFunctionBlockOrSemicolon(isGenerator | isAsync, Diagnostics::or_expected);
                setAwaitContext(savedAwaitContext);
                auto node = factory.createFunctionDeclaration(decorators, modifiers, asteriskToken, name, typeParameters, parameters, type, body);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseConstructorName() {
                if (token() == SyntaxKind::ConstructorKeyword) {
                    return parseExpected(SyntaxKind::ConstructorKeyword);
                }
                if (token() == SyntaxKind::StringLiteral && lookAhead<SyntaxKind>(std::bind(&Parser::nextToken, this)) == SyntaxKind::OpenParenToken) {
                    return tryParse(() => {
                        auto literalNode = parseLiteralNode();
                        return literalNode.text == "constructor" ? literalNode : undefined;
                    });
                }
            }

            auto tryParseConstructorDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> ConstructorDeclaration {
                return tryParse(() => {
                    if (parseConstructorName()) {
                        auto typeParameters = parseTypeParameters();
                        auto parameters = parseParameters(SignatureFlags.None);
                        auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
                        auto body = parseFunctionBlockOrSemicolon(SignatureFlags.None, Diagnostics::or_expected);
                        auto node = factory.createConstructorDeclaration(decorators, modifiers, parameters, body);
                        // Attach `typeParameters` and `type` if they exist so that we can report them in the grammar checker.
                        node.typeParameters = typeParameters;
                        node.type = type;
                        return withJSDoc(finishNode(node, pos), hasJSDoc);
                    }
                });
            }

            auto parseMethodDeclaration(
                number pos,
                boolean hasJSDoc,
                NodeArray<Decorator> decorators,
                NodeArray<Modifier> modifiers,
                SyntaxKind asteriskToken,
                PropertyName name,
                SyntaxKind questionToken,
                SyntaxKind exclamationToken,
                DiagnosticMessage diagnosticMessage
            ) -> MethodDeclaration {
                auto isGenerator = asteriskToken ? SignatureFlags.Yield : SignatureFlags.None;
                auto isAsync = some(modifiers, isAsyncModifier) ? SignatureFlags.Await : SignatureFlags.None;
                auto typeParameters = parseTypeParameters();
                auto parameters = parseParameters(isGenerator | isAsync);
                auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
                auto body = parseFunctionBlockOrSemicolon(isGenerator | isAsync, diagnosticMessage);
                auto node = factory.createMethodDeclaration(
                    decorators,
                    modifiers,
                    asteriskToken,
                    name,
                    questionToken,
                    typeParameters,
                    parameters,
                    type,
                    body
                );
                // An exclamation token on a method is invalid syntax and will be handled by the grammar checker
                node.exclamationToken = exclamationToken;
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parsePropertyDeclaration(
                number pos,
                boolean hasJSDoc,
                NodeArray<Decorator> decorators,
                NodeArray<Modifier> modifiers,
                PropertyName name,
                SyntaxKind questionToken
            ) -> PropertyDeclaration {
                auto exclamationToken = !questionToken && !scanner.hasPrecedingLineBreak() ? parseOptionalToken(SyntaxKind::ExclamationToken) : undefined;
                auto type = parseTypeAnnotation();
                auto initializer = doOutsideOfContext(NodeFlags::YieldContext | NodeFlags::AwaitContext | NodeFlags::DisallowInContext, parseInitializer);
                parseSemicolon();
                auto node = factory.createPropertyDeclaration(decorators, modifiers, name, questionToken || exclamationToken, type, initializer);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parsePropertyOrMethodDeclaration(
                number pos,
                boolean hasJSDoc,
                NodeArray<Decorator> decorators,
                NodeArray<Modifier> modifiers
            ) -> Node {
                auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
                auto name = parsePropertyName();
                // this Note is not legal.as<per>() the grammar.  But we allow it in the parser and
                // report an error in the grammar checker.
                auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
                if (asteriskToken || token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken) {
                    return parseMethodDeclaration(pos, hasJSDoc, decorators, modifiers, asteriskToken, name, questionToken, /*exclamationToken*/ undefined, Diagnostics::or_expected);
                }
                return parsePropertyDeclaration(pos, hasJSDoc, decorators, modifiers, name, questionToken);
            }

            auto parseAccessorDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers, SyntaxKind kind) -> AccessorDeclaration {
                auto name = parsePropertyName();
                auto typeParameters = parseTypeParameters();
                auto parameters = parseParameters(SignatureFlags.None);
                auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
                auto body = parseFunctionBlockOrSemicolon(SignatureFlags.None);
                auto node = kind == SyntaxKind::GetAccessor
                    ? factory.createGetAccessorDeclaration(decorators, modifiers, name, parameters, type, body)
                    : factory.createSetAccessorDeclaration(decorators, modifiers, name, parameters, body);
                // Keep track of `typeParameters` (for both) and `type` (for setters) if they were parsed those indicate grammar errors
                node.typeParameters = typeParameters;
                if (type && node.kind == SyntaxKind::SetAccessor) (node.as<Mutable>()<SetAccessorDeclaration>).type = type;
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto isClassMemberStart() -> boolean {
                auto SyntaxKind idToken;

                if (token() == SyntaxKind::AtToken) {
                    return true;
                }

                // Eat up all modifiers, but hold on to the last one in case it is actually an identifier.
                while (isModifierKind(token())) {
                    idToken = token();
                    // If the idToken is a class modifier (protected, private, public, and static), it is
                    // certain that we are starting to parse class member. This allows better error recovery
                    // Example:
                    //      public foo() ...     // true
                    //      public @dec blah ... // true; we will then report an error later
                    //      public ...    // true; we will then report an error later
                    if (isClassMemberModifier(idToken)) {
                        return true;
                    }

                    nextToken();
                }

                if (token() == SyntaxKind::AsteriskToken) {
                    return true;
                }

                // Try to get the first property-like token following all modifiers.
                // This can either be an identifier or the 'get' or 'set' keywords.
                if (isLiteralPropertyName()) {
                    idToken = token();
                    nextToken();
                }

                // Index signatures and computed properties are class members; we can parse.
                if (token() == SyntaxKind::OpenBracketToken) {
                    return true;
                }

                // If we were able to get any potential identifier...
                if (idToken != undefined) {
                    // If we have a non-keyword identifier, or if we have an accessor, then it's safe to parse.
                    if (!isKeyword(idToken) || idToken == SyntaxKind::SetKeyword || idToken == SyntaxKind::GetKeyword) {
                        return true;
                    }

                    // If it *is* a keyword, but not an accessor, check a little farther along
                    // to see if it should actually be parsed.as<a>() class member.
                    switch (token()) {
                        case SyntaxKind::OpenParenToken:     // Method declaration
                        case SyntaxKind::LessThanToken:      // Generic Method declaration
                        case SyntaxKind::ExclamationToken:   // Non-null assertion on property name
                        case SyntaxKind::ColonToken:         // Type Annotation for declaration
                        case SyntaxKind::EqualsToken:        // Initializer for declaration
                        case SyntaxKind::QuestionToken:      // Not valid, but permitted so that it gets caught later on.
                            return true;
                        default:
                            // Covers
                            //  - Semicolons     (declaration termination)
                            //  - Closing braces (end-of-class, must be declaration)
                            //  - End-of-files   (not valid, but permitted so that it gets caught later on)
                            //  - Line-breaks    (enabling *automatic semicolon insertion*)
                            return canParseSemicolon();
                    }
                }

                return false;
            }

            auto parseDecoratorExpression() {
                if (inAwaitContext() && token() == SyntaxKind::AwaitKeyword) {
                    // `@await` is is disallowed in an [Await] context, but can cause parsing to go off the rails
                    // This simply parses the missing identifier and moves on.
                    auto pos = getNodePos();
                    auto awaitExpression = parseIdentifier(Diagnostics::Expression_expected);
                    nextToken();
                    auto memberExpression = parseMemberExpressionRest(pos, awaitExpression, /*allowOptionalChain*/ true);
                    return parseCallExpressionRest(pos, memberExpression);
                }
                return parseLeftHandSideExpressionOrHigher();
            }

            auto tryParseDecorator() -> Decorator {
                auto pos = getNodePos();
                if (!parseOptional(SyntaxKind::AtToken)) {
                    return undefined;
                }
                auto expression = doInDecoratorContext(parseDecoratorExpression);
                return finishNode(factory.createDecorator(expression), pos);
            }

            auto parseDecorators() -> NodeArray<Decorator> {
                auto pos = getNodePos();
                auto list, decorator;
                while (decorator = tryParseDecorator()) {
                    list = append(list, decorator);
                }
                return list && createNodeArray(list, pos);
            }

            auto tryParseModifier(boolean permitInvalidConstAsModifier) -> Modifier {
                auto pos = getNodePos();
                auto kind = token();

                if (token() == SyntaxKind::ConstKeyword && permitInvalidConstAsModifier) {
                    // We need to ensure that any subsequent modifiers appear on the same line
                    // so that when 'const' is a standalone declaration, we don't issue an error.
                    if (!tryParse<boolean>(std::bind(&Parser::nextTokenIsOnSameLineAndCanFollowModifier, this))) {
                        return undefined;
                    }
                }
                else {
                    if (!parseAnyContextualModifier()) {
                        return undefined;
                    }
                }

                return finishNode(factory.createToken(kind.as<Modifier>()["kind"]), pos);
            }

            /*
             * There are situations in which a modifier like 'const' will appear unexpectedly, such.as<on>() a class member.
             * In those situations, if we are entirely sure that 'const' is not valid on its own (such.as<when>() ASI takes effect
             * and turns it into a standalone declaration), then it is better to parse it and report an error later.
             *
             * In such situations, 'permitInvalidConstAsModifier' should be set to true.
             */
            auto parseModifiers(boolean permitInvalidConstAsModifier) -> NodeArray<Modifier> {
                auto pos = getNodePos();
                auto list, modifier;
                while (modifier = tryParseModifier(permitInvalidConstAsModifier)) {
                    list = append(list, modifier);
                }
                return list && createNodeArray(list, pos);
            }

            auto parseModifiersForArrowFunction() -> NodeArray<Modifier> {
                auto NodeArray<Modifier> modifiers;
                if (token() == SyntaxKind::AsyncKeyword) {
                    auto pos = getNodePos();
                    nextToken();
                    auto modifier = finishNode(factory.createToken(SyntaxKind::AsyncKeyword), pos);
                    modifiers = createNodeArray<Modifier>([modifier], pos);
                }
                return modifiers;
            }

            auto parseClassElement() -> ClassElement {
                auto pos = getNodePos();
                if (token() == SyntaxKind::SemicolonToken) {
                    nextToken();
                    return finishNode(factory.createSemicolonClassElement(), pos);
                }

                auto hasJSDoc = hasPrecedingJSDocComment();
                auto decorators = parseDecorators();
                auto modifiers = parseModifiers(/*permitInvalidConstAsModifier*/ true);

                if (parseContextualModifier(SyntaxKind::GetKeyword)) {
                    return parseAccessorDeclaration(pos, hasJSDoc, decorators, modifiers, SyntaxKind::GetAccessor);
                }

                if (parseContextualModifier(SyntaxKind::SetKeyword)) {
                    return parseAccessorDeclaration(pos, hasJSDoc, decorators, modifiers, SyntaxKind::SetAccessor);
                }

                if (token() == SyntaxKind::ConstructorKeyword || token() == SyntaxKind::StringLiteral) {
                    auto constructorDeclaration = tryParseConstructorDeclaration(pos, hasJSDoc, decorators, modifiers);
                    if (constructorDeclaration) {
                        return constructorDeclaration;
                    }
                }

                if (isIndexSignature()) {
                    return parseIndexSignatureDeclaration(pos, hasJSDoc, decorators, modifiers);
                }

                // It is very important that we check this *after* checking indexers because
                // the [ token can start an index signature or a computed property name
                if (scanner.tokenIsIdentifierOrKeyword(token()) ||
                    token() == SyntaxKind::StringLiteral ||
                    token() == SyntaxKind::NumericLiteral ||
                    token() == SyntaxKind::AsteriskToken ||
                    token() == SyntaxKind::OpenBracketToken) {
                    auto isAmbient = some(modifiers, isDeclareModifier);
                    if (isAmbient) {
                        for (auto m of modifiers!) {
                            (m.as<Mutable>()<Node>).flags |= NodeFlags::Ambient;
                        }
                        return doInsideOfContext(NodeFlags::Ambient, () => parsePropertyOrMethodDeclaration(pos, hasJSDoc, decorators, modifiers));
                    }
                    else {
                        return parsePropertyOrMethodDeclaration(pos, hasJSDoc, decorators, modifiers);
                    }
                }

                if (decorators || modifiers) {
                    // treat this.as<a>() property declaration with a missing name.
                    auto name = createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ true, Diagnostics::Declaration_expected);
                    return parsePropertyDeclaration(pos, hasJSDoc, decorators, modifiers, name, /*questionToken*/ undefined);
                }

                // 'isClassMemberStart' should have hinted not to attempt parsing.
                return Debug::fail("Should not have attempted to parse class member declaration.");
            }

            auto parseClassExpression() -> ClassExpression {
                return <ClassExpression>parseClassDeclarationOrExpression(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined, SyntaxKind::ClassExpression);
            }

            auto parseClassDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> ClassDeclaration {
                return <ClassDeclaration>parseClassDeclarationOrExpression(pos, hasJSDoc, decorators, modifiers, SyntaxKind::ClassDeclaration);
            }

            auto parseClassDeclarationOrExpression(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers, SyntaxKind kind) -> ClassLikeDeclaration {
                auto savedAwaitContext = inAwaitContext();
                parseExpected(SyntaxKind::ClassKeyword);
                // We don't parse the name here in await context, instead we will report a grammar error in the checker.
                auto name = parseNameOfClassDeclarationOrExpression();
                auto typeParameters = parseTypeParameters();
                if (some(modifiers, isExportModifier)) setAwaitContext(/*value*/ true);
                auto heritageClauses = parseHeritageClauses();

                auto members;
                if (parseExpected(SyntaxKind::OpenBraceToken)) {
                    // ClassTail[Yield,Await] : (Modified) See 14.5
                    //      ClassHeritage[?Yield,?Await]opt { ClassBody[?Yield,?Await]opt }
                    members = parseClassMembers();
                    parseExpected(SyntaxKind::CloseBraceToken);
                }
                else {
                    members = createMissingList<ClassElement>();
                }
                setAwaitContext(savedAwaitContext);
                auto node = kind == SyntaxKind::ClassDeclaration
                    ? factory.createClassDeclaration(decorators, modifiers, name, typeParameters, heritageClauses, members)
                    : factory.createClassExpression(decorators, modifiers, name, typeParameters, heritageClauses, members);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseNameOfClassDeclarationOrExpression() -> Identifier {
                // implements is a future reserved word so
                // 'class implements' might mean either
                // - class expression with omitted name, 'implements' starts heritage clause
                // - class with name 'implements'
                // 'isImplementsClause' helps to disambiguate between these two cases
                return isBindingIdentifier() && !isImplementsClause()
                    ? createIdentifier(isBindingIdentifier())
                    : undefined;
            }

            auto isImplementsClause() {
                return token() == SyntaxKind::ImplementsKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeyword, this));
            }

            auto parseHeritageClauses() -> NodeArray<HeritageClause> {
                // ClassTail[Yield,Await] : (Modified) See 14.5
                //      ClassHeritage[?Yield,?Await]opt { ClassBody[?Yield,?Await]opt }

                if (isHeritageClause()) {
                    return parseList(ParsingContext::HeritageClauses, parseHeritageClause);
                }

                return undefined;
            }

            auto parseHeritageClause() -> HeritageClause {
                auto pos = getNodePos();
                auto tok = token();
                Debug::_assert(tok == SyntaxKind::ExtendsKeyword || tok == SyntaxKind::ImplementsKeyword); // isListElement() should ensure this.
                nextToken();
                auto types = parseDelimitedList(ParsingContext::HeritageClauseElement, parseExpressionWithTypeArguments);
                return finishNode(factory.createHeritageClause(tok, types), pos);
            }

            auto parseExpressionWithTypeArguments() -> ExpressionWithTypeArguments {
                auto pos = getNodePos();
                auto expression = parseLeftHandSideExpressionOrHigher();
                auto typeArguments = tryParseTypeArguments();
                return finishNode(factory.createExpressionWithTypeArguments(expression, typeArguments), pos);
            }

            auto tryParseTypeArguments() -> NodeArray<TypeNode> {
                return token() == SyntaxKind::LessThanToken ?
                    parseBracketedList(ParsingContext::TypeArguments, parseType, SyntaxKind::LessThanToken, SyntaxKind::GreaterThanToken) : undefined;
            }

            auto isHeritageClause() -> boolean {
                return token() == SyntaxKind::ExtendsKeyword || token() == SyntaxKind::ImplementsKeyword;
            }

            auto parseClassMembers() -> NodeArray<ClassElement> {
                return parseList(ParsingContext::ClassMembers, parseClassElement);
            }

            auto parseInterfaceDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> InterfaceDeclaration {
                parseExpected(SyntaxKind::InterfaceKeyword);
                auto name = parseIdentifier();
                auto typeParameters = parseTypeParameters();
                auto heritageClauses = parseHeritageClauses();
                auto members = parseObjectTypeMembers();
                auto node = factory.createInterfaceDeclaration(decorators, modifiers, name, typeParameters, heritageClauses, members);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseTypeAliasDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> TypeAliasDeclaration {
                parseExpected(SyntaxKind::TypeKeyword);
                auto name = parseIdentifier();
                auto typeParameters = parseTypeParameters();
                parseExpected(SyntaxKind::EqualsToken);
                auto type = token() == SyntaxKind::IntrinsicKeyword && tryParse<boolean>(std::bind(&Parser::parseKeywordAndNoDot, this)) || parseType();
                parseSemicolon();
                auto node = factory.createTypeAliasDeclaration(decorators, modifiers, name, typeParameters, type);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            // In an ambient declaration, the grammar only allows integer literals.as<initializers>().
            // In a non-ambient declaration, the grammar allows uninitialized members only in a
            // ConstantEnumMemberSection, which starts at the beginning of an enum declaration
            // or any time an integer literal initializer is encountered.
            auto parseEnumMember() -> EnumMember {
                auto pos = getNodePos();
                auto hasJSDoc = hasPrecedingJSDocComment();
                auto name = parsePropertyName();
                auto initializer = allowInAnd(parseInitializer);
                return withJSDoc(finishNode(factory.createEnumMember(name, initializer), pos), hasJSDoc);
            }

            auto parseEnumDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> EnumDeclaration {
                parseExpected(SyntaxKind::EnumKeyword);
                auto name = parseIdentifier();
                auto members;
                if (parseExpected(SyntaxKind::OpenBraceToken)) {
                    members = doOutsideOfYieldAndAwaitContext(() => parseDelimitedList(ParsingContext::EnumMembers, parseEnumMember));
                    parseExpected(SyntaxKind::CloseBraceToken);
                }
                else {
                    members = createMissingList<EnumMember>();
                }
                auto node = factory.createEnumDeclaration(decorators, modifiers, name, members);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseModuleBlock() -> ModuleBlock {
                auto pos = getNodePos();
                auto statements;
                if (parseExpected(SyntaxKind::OpenBraceToken)) {
                    statements = parseList(ParsingContext::BlockStatements, parseStatement);
                    parseExpected(SyntaxKind::CloseBraceToken);
                }
                else {
                    statements = createMissingList<Statement>();
                }
                return finishNode(factory.createModuleBlock(statements), pos);
            }

            auto parseModuleOrNamespaceDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers, NodeFlags flags) -> ModuleDeclaration {
                // If we are parsing a dotted namespace name, we want to
                // propagate the 'Namespace' flag across the names if set.
                auto namespaceFlag = flags & NodeFlags::Namespace;
                auto name = parseIdentifier();
                auto body = parseOptional(SyntaxKind::DotToken)
                    ? <NamespaceDeclaration>parseModuleOrNamespaceDeclaration(getNodePos(), /*hasJSDoc*/ false, /*decorators*/ undefined, /*modifiers*/ undefined, NodeFlags::NestedNamespace | namespaceFlag)
                    : parseModuleBlock();
                auto node = factory.createModuleDeclaration(decorators, modifiers, name, body, flags);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseAmbientExternalModuleDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> ModuleDeclaration {
                auto flags = NodeFlags::None;
                Node name;
                if (token() == SyntaxKind::GlobalKeyword) {
                    // parse 'global'.as<name>() of global scope augmentation
                    name = parseIdentifier();
                    flags |= NodeFlags::GlobalAugmentation;
                }
                else {
                    name = parseLiteralNode().as<StringLiteral>();
                    name.text = internIdentifier(name.text);
                }
                ModuleBlock body;
                if (token() == SyntaxKind::OpenBraceToken) {
                    body = parseModuleBlock();
                }
                else {
                    parseSemicolon();
                }
                auto node = factory.createModuleDeclaration(decorators, modifiers, name, body, flags);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseModuleDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> ModuleDeclaration {
                auto NodeFlags flags = 0;
                if (token() == SyntaxKind::GlobalKeyword) {
                    // global augmentation
                    return parseAmbientExternalModuleDeclaration(pos, hasJSDoc, decorators, modifiers);
                }
                else if (parseOptional(SyntaxKind::NamespaceKeyword)) {
                    flags |= NodeFlags::Namespace;
                }
                else {
                    parseExpected(SyntaxKind::ModuleKeyword);
                    if (token() == SyntaxKind::StringLiteral) {
                        return parseAmbientExternalModuleDeclaration(pos, hasJSDoc, decorators, modifiers);
                    }
                }
                return parseModuleOrNamespaceDeclaration(pos, hasJSDoc, decorators, modifiers, flags);
            }

            auto isExternalModuleReference() -> boolean {
                return token() == SyntaxKind::RequireKeyword &&
                    lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParen, this));
            }

            auto nextTokenIsOpenParen() -> boolean {
                return nextToken() == SyntaxKind::OpenParenToken;
            }

            auto nextTokenIsSlash() -> boolean {
                return nextToken() == SyntaxKind::SlashToken;
            }

            auto parseNamespaceExportDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> NamespaceExportDeclaration {
                parseExpected(SyntaxKind::AsKeyword);
                parseExpected(SyntaxKind::NamespaceKeyword);
                auto name = parseIdentifier();
                parseSemicolon();
                auto node = factory.createNamespaceExportDeclaration(name);
                // NamespaceExportDeclaration nodes cannot have decorators or modifiers, so we attach them here so we can report them in the grammar checker
                node.decorators = decorators;
                node.modifiers = modifiers;
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseImportDeclarationOrImportEqualsDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> Node {
                parseExpected(SyntaxKind::ImportKeyword);

                auto afterImportPos = scanner.getStartPos();

                // We don't parse the identifier here in await context, instead we will report a grammar error in the checker.
                auto Identifier identifier;
                if (isIdentifier()) {
                    identifier = parseIdentifier();
                }

                auto isTypeOnly = false;
                if (token() != SyntaxKind::FromKeyword &&
                    identifier?.escapedText == "type" &&
                    (isIdentifier() || tokenAfterImportDefinitelyProducesImportDeclaration())
                ) {
                    isTypeOnly = true;
                    identifier = isIdentifier() ? parseIdentifier() : undefined;
                }

                if (identifier && !tokenAfterImportedIdentifierDefinitelyProducesImportDeclaration()) {
                    return parseImportEqualsDeclaration(pos, hasJSDoc, decorators, modifiers, identifier, isTypeOnly);
                }

                // ImportDeclaration:
                //  import ImportClause from ModuleSpecifier ;
                //  import ModuleSpecifier;
                auto ImportClause importClause;
                if (identifier || // import id
                    token() == SyntaxKind::AsteriskToken || // import *
                    token() == SyntaxKind::OpenBraceToken    // import {
                ) {
                    importClause = parseImportClause(identifier, afterImportPos, isTypeOnly);
                    parseExpected(SyntaxKind::FromKeyword);
                }

                auto moduleSpecifier = parseModuleSpecifier();
                parseSemicolon();
                auto node = factory.createImportDeclaration(decorators, modifiers, importClause, moduleSpecifier);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto tokenAfterImportDefinitelyProducesImportDeclaration() {
                return token() == SyntaxKind::AsteriskToken || token() == SyntaxKind::OpenBraceToken;
            }

            auto tokenAfterImportedIdentifierDefinitelyProducesImportDeclaration() {
                // In `import id ___`, the current token decides whether to produce
                // an ImportDeclaration or ImportEqualsDeclaration.
                return token() == SyntaxKind::CommaToken || token() == SyntaxKind::FromKeyword;
            }

            auto parseImportEqualsDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers, Identifier identifier, boolean isTypeOnly) -> ImportEqualsDeclaration {
                parseExpected(SyntaxKind::EqualsToken);
                auto moduleReference = parseModuleReference();
                parseSemicolon();
                auto node = factory.createImportEqualsDeclaration(decorators, modifiers, isTypeOnly, identifier, moduleReference);
                auto finished = withJSDoc(finishNode(node, pos), hasJSDoc);
                return finished;
            }

            auto parseImportClause(Identifier identifier, number pos, boolean isTypeOnly) {
                // ImportClause:
                //  ImportedDefaultBinding
                //  NameSpaceImport
                //  NamedImports
                //  ImportedDefaultBinding, NameSpaceImport
                //  ImportedDefaultBinding, NamedImports

                // If there was no default import or if there is comma token after default import
                // parse namespace or named imports
                auto NamespaceImport namedBindings | NamedImports;
                if (!identifier ||
                    parseOptional(SyntaxKind::CommaToken)) {
                    namedBindings = token() == SyntaxKind::AsteriskToken ? parseNamespaceImport() : parseNamedImportsOrExports(SyntaxKind::NamedImports);
                }

                return finishNode(factory.createImportClause(isTypeOnly, identifier, namedBindings), pos);
            }

            auto parseModuleReference() {
                return isExternalModuleReference()
                    ? parseExternalModuleReference()
                    : parseEntityName(/*allowReservedWords*/ false);
            }

            auto parseExternalModuleReference() {
                auto pos = getNodePos();
                parseExpected(SyntaxKind::RequireKeyword);
                parseExpected(SyntaxKind::OpenParenToken);
                auto expression = parseModuleSpecifier();
                parseExpected(SyntaxKind::CloseParenToken);
                return finishNode(factory.createExternalModuleReference(expression), pos);
            }

            auto parseModuleSpecifier() -> Expression {
                if (token() == SyntaxKind::StringLiteral) {
                    auto result = parseLiteralNode();
                    result.text = internIdentifier(result.text);
                    return result;
                }
                else {
                    // We allow arbitrary expressions here, even though the grammar only allows string
                    // literals.  We check to ensure that it is only a string literal later in the grammar
                    // check pass.
                    return parseExpression();
                }
            }

            auto parseNamespaceImport() -> NamespaceImport {
                // NameSpaceImport:
                //  *.as<ImportedBinding>()
                auto pos = getNodePos();
                parseExpected(SyntaxKind::AsteriskToken);
                parseExpected(SyntaxKind::AsKeyword);
                auto name = parseIdentifier();
                return finishNode(factory.createNamespaceImport(name), pos);
            }

            auto parseNamedImportsOrExports(SyntaxKind kind) -> NamedImportsOrExports {
                auto pos = getNodePos();

                // NamedImports:
                //  { }
                //  { ImportsList }
                //  { ImportsList, }

                // ImportsList:
                //  ImportSpecifier
                //  ImportsList, ImportSpecifier
                auto node = kind == SyntaxKind::NamedImports
                    ? factory.createNamedImports(parseBracketedList(ParsingContext::ImportOrExportSpecifiers, parseImportSpecifier, SyntaxKind::OpenBraceToken, SyntaxKind::CloseBraceToken))
                    : factory.createNamedExports(parseBracketedList(ParsingContext::ImportOrExportSpecifiers, parseExportSpecifier, SyntaxKind::OpenBraceToken, SyntaxKind::CloseBraceToken));
                return finishNode(node, pos);
            }

            auto parseExportSpecifier() {
                return parseImportOrExportSpecifier(SyntaxKind::ExportSpecifier).as<ExportSpecifier>();
            }

            auto parseImportSpecifier() {
                return parseImportOrExportSpecifier(SyntaxKind::ImportSpecifier).as<ImportSpecifier>();
            }

            auto parseImportOrExportSpecifier(SyntaxKind kind) -> ImportOrExportSpecifier {
                auto pos = getNodePos();
                // ImportSpecifier:
                //   BindingIdentifier
                //   IdentifierName.as<BindingIdentifier>()
                // ExportSpecifier:
                //   IdentifierName
                //   IdentifierName.as<IdentifierName>()
                auto checkIdentifierIsKeyword = isKeyword(token()) && !isIdentifier();
                auto checkIdentifierStart = scanner.getTokenPos();
                auto checkIdentifierEnd = scanner.getTextPos();
                auto identifierName = parseIdentifierName();
                auto Identifier propertyName;
                auto Identifier name;
                if (token() == SyntaxKind::AsKeyword) {
                    propertyName = identifierName;
                    parseExpected(SyntaxKind::AsKeyword);
                    checkIdentifierIsKeyword = isKeyword(token()) && !isIdentifier();
                    checkIdentifierStart = scanner.getTokenPos();
                    checkIdentifierEnd = scanner.getTextPos();
                    name = parseIdentifierName();
                }
                else {
                    name = identifierName;
                }
                if (kind == SyntaxKind::ImportSpecifier && checkIdentifierIsKeyword) {
                    parseErrorAt(checkIdentifierStart, checkIdentifierEnd, Diagnostics::Identifier_expected);
                }
                auto node = kind == SyntaxKind::ImportSpecifier
                    ? factory.createImportSpecifier(propertyName, name)
                    : factory.createExportSpecifier(propertyName, name);
                return finishNode(node, pos);
            }

            auto parseNamespaceExport(number pos) -> NamespaceExport {
                return finishNode(factory.createNamespaceExport(parseIdentifierName()), pos);
            }

            auto parseExportDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> ExportDeclaration {
                auto savedAwaitContext = inAwaitContext();
                setAwaitContext(/*value*/ true);
                auto NamedExportBindings exportClause;
                auto Expression moduleSpecifier;
                auto isTypeOnly = parseOptional(SyntaxKind::TypeKeyword);
                auto namespaceExportPos = getNodePos();
                if (parseOptional(SyntaxKind::AsteriskToken)) {
                    if (parseOptional(SyntaxKind::AsKeyword)) {
                        exportClause = parseNamespaceExport(namespaceExportPos);
                    }
                    parseExpected(SyntaxKind::FromKeyword);
                    moduleSpecifier = parseModuleSpecifier();
                }
                else {
                    exportClause = parseNamedImportsOrExports(SyntaxKind::NamedExports);
                    // It is not uncommon to accidentally omit the 'from' keyword. Additionally, in editing scenarios,
                    // the 'from' keyword can be parsed.as<a>() named when the clause is unterminated (i.e. `{ from "moduleName";`)
                    // If we don't have a 'from' keyword, see if we have a string literal such that ASI won't take effect.
                    if (token() == SyntaxKind::FromKeyword || (token() == SyntaxKind::StringLiteral && !scanner.hasPrecedingLineBreak())) {
                        parseExpected(SyntaxKind::FromKeyword);
                        moduleSpecifier = parseModuleSpecifier();
                    }
                }
                parseSemicolon();
                setAwaitContext(savedAwaitContext);
                auto node = factory.createExportDeclaration(decorators, modifiers, isTypeOnly, exportClause, moduleSpecifier);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto parseExportAssignment(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> ExportAssignment {
                auto savedAwaitContext = inAwaitContext();
                setAwaitContext(/*value*/ true);
                auto boolean isExportEquals;
                if (parseOptional(SyntaxKind::EqualsToken)) {
                    isExportEquals = true;
                }
                else {
                    parseExpected(SyntaxKind::DefaultKeyword);
                }
                auto expression = parseAssignmentExpressionOrHigher();
                parseSemicolon();
                setAwaitContext(savedAwaitContext);
                auto node = factory.createExportAssignment(decorators, modifiers, isExportEquals, expression);
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            auto setExternalModuleIndicator(SourceFile sourceFile) -> void {
                // Try to use the first top-level import/when available, then
                // fall back to looking for an 'import.meta' somewhere in the tree if necessary.
                sourceFile.externalModuleIndicator =
                        forEach(sourceFile.statements, isAnExternalModuleIndicatorNode) ||
                        getImportMetaIfNecessary(sourceFile);
            }

            auto isAnExternalModuleIndicatorNode(Node node) {
                return hasModifierOfKind(node, SyntaxKind::ExportKeyword)
                    || isImportEqualsDeclaration(node) && ts.isExternalModuleReference(node.moduleReference)
                    || isImportDeclaration(node)
                    || isExportAssignment(node)
                    || isExportDeclaration(node) ? node : undefined;
            }

            auto getImportMetaIfNecessary(SourceFile sourceFile) {
                return sourceFile.flags & NodeFlags::PossiblyContainsImportMeta ?
                    walkTreeForExternalModuleIndicators(sourceFile) :
                    undefined;
            }

            auto walkTreeForExternalModuleIndicators(Node node) -> Node {
                return isImportMeta(node) ? node : forEachChild(node, walkTreeForExternalModuleIndicators);
            }

            /** Do not use hasModifier inside the parser; it relies on parent pointers. Use this instead. */
            auto hasModifierOfKind(Node node, SyntaxKind kind) {
                return some(node.modifiers, m => m.kind == kind);
            }

            auto isImportMeta(Node node) -> boolean {
                return isMetaProperty(node) && node.keywordToken == SyntaxKind::ImportKeyword && node.name.escapedText == "meta";
            }

            // [[[ namespace JSDocParser ]]]

            auto parseJSDocTypeExpressionForTests(string content, number start, number length) -> Undefined<NodeWithDiagnostics> {
                initializeState("file.js", content, ScriptTarget::Latest, /*_syntaxCursor:*/ undefined, ScriptKind::JS);
                scanner.setText(content, start, length);
                currentToken = scanner.scan();
                auto jsDocTypeExpression = parseJSDocTypeExpression();

                auto sourceFile = createSourceFile("file.js", ScriptTarget::Latest, ScriptKind::JS, /*isDeclarationFile*/ false, [], factory.createToken(SyntaxKind::EndOfFileToken), NodeFlags::None);
                auto diagnostics = attachFileToDiagnostics(parseDiagnostics, sourceFile);
                if (jsDocDiagnostics) {
                    sourceFile.jsDocDiagnostics = attachFileToDiagnostics(jsDocDiagnostics, sourceFile);
                }

                clearState();

                return jsDocTypeExpression ? NodeWithDiagnostics{ jsDocTypeExpression, diagnostics } : undefined;
            }

            // Parses out a JSDoc type expression.
            auto parseJSDocTypeExpression(boolean mayOmitBraces) -> JSDocTypeExpression {
                auto pos = getNodePos();
                auto hasBrace = (mayOmitBraces ? parseOptional : parseExpected)(SyntaxKind::OpenBraceToken);
                auto type = doInsideOfContext(NodeFlags::JSDoc, parseJSDocType);
                if (!mayOmitBraces || hasBrace) {
                    parseExpectedJSDoc(SyntaxKind::CloseBraceToken);
                }

                auto result = factory.createJSDocTypeExpression(type);
                fixupParentReferences(result);
                return finishNode(result, pos);
            }

            auto parseJSDocNameReference() -> JSDocNameReference {
                auto pos = getNodePos();
                auto hasBrace = parseOptional(SyntaxKind::OpenBraceToken);
                auto entityName = parseEntityName(/* allowReservedWords*/ false);
                if (hasBrace) {
                    parseExpectedJSDoc(SyntaxKind::CloseBraceToken);
                }

                auto result = factory.createJSDocNameReference(entityName);
                fixupParentReferences(result);
                return finishNode(result, pos);
            }

            auto parseIsolatedJSDocComment(string content, number start, number length) -> Undefined<NodeWithDiagnostics> {
                initializeState(string(), content, ScriptTarget::Latest, /*_syntaxCursor:*/ undefined, ScriptKind::JS);
                auto jsDoc = doInsideOfContext(NodeFlags::JSDoc, () => parseJSDocCommentWorker(start, length));

                auto sourceFile = <SourceFile>{ LanguageVariant::Standard languageVariant, content text };
                auto diagnostics = attachFileToDiagnostics(parseDiagnostics, sourceFile);
                clearState();

                return jsDoc ? NodeWithDiagnostics{ jsDoc, diagnostics } : undefined;
            }

            auto parseJSDocComment(SyntaxKind parent, number start, number length) -> JSDoc {
                auto saveToken = currentToken;
                auto saveParseDiagnosticsLength = parseDiagnostics::size();
                auto saveParseErrorBeforeNextFinishedNode = parseErrorBeforeNextFinishedNode;

                auto comment = doInsideOfContext(NodeFlags::JSDoc, () => parseJSDocCommentWorker(start, length));
                setParent(comment, parent);

                if (contextFlags & NodeFlags::JavaScriptFile) {
                    if (!jsDocDiagnostics) {
                        jsDocDiagnostics = [];
                    }
                    jsDocDiagnostics::push_back(...parseDiagnostics);
                }
                currentToken = saveToken;
                parseDiagnostics::size() = saveParseDiagnosticsLength;
                parseErrorBeforeNextFinishedNode = saveParseErrorBeforeNextFinishedNode;
                return comment;
            }

            enum class JSDocState : number {
                BeginningOfLine,
                SawAsterisk,
                SavingComments,
                SavingBackticks, // Only NOTE used when parsing tag comments
            };

            enum class PropertyLikeParse : number {
                Property = 1 << 0,
                Parameter = 1 << 1,
                CallbackParameter = 1 << 2,
            };

            auto parseJSDocCommentWorker(number start = 0, number length = -1) -> JSDoc {
                auto content = sourceText;
                auto end = length == -1 ? content.size() : start + length;
                length = end - start;

                Debug::_assert(start >= 0);
                Debug::_assert(start <= end);
                Debug::_assert(end <= content.size());

                // Check for /** (JSDoc opening part)
                if (!isJSDocLikeText(content, start)) {
                    return undefined;
                }

                auto std::vector<JSDocTag> tags;
                auto number tagsPos;
                auto number tagsEnd;
                auto std::vector<string> = [] comments;

                // + 3 for leading /**, - 5 in total for /** */
                return scanner.scanRange(start + 3, length - 5, () => {
                    // Initially we can parse out a tag.  We also have seen a starting asterisk.
                    // This is so that /** * @type */ doesn't parse.
                    auto state = JSDocState.SawAsterisk;
                    auto number margin;
                    // + 4 for leading '/** '
                    // + 1 because the last index of \n is always one index before the first character in the line and coincidentally, if there is no \n before start, it is -1, which is also one index before the first character
                    auto indent = start - (content.lastIndexOf("\n", start) + 1) + 4;
                    auto pushComment(string text) {
                        if (!margin) {
                            margin = indent;
                        }
                        comments.push_back(text);
                        indent += text.size();
                    }

                    nextTokenJSDoc();
                    while (parseOptionalJsdoc(SyntaxKind::WhitespaceTrivia));
                    if (parseOptionalJsdoc(SyntaxKind::NewLineTrivia)) {
                        state = JSDocState.BeginningOfLine;
                        indent = 0;
                    }
                    while loop (true) {
                        switch (token()) {
                            case SyntaxKind::AtToken:
                                if (state == JSDocState.BeginningOfLine || state == JSDocState.SawAsterisk) {
                                    removeTrailingWhitespace(comments);
                                    addTag(parseTag(indent));
                                    // According NOTE to usejsdoc.org, a tag goes to end of line, except the last tag.
                                    // Real-world comments may break this rule, so "BeginningOfLine" will not be a real line beginning
                                    // for malformed examples like `/** @param {string} x @returns {number} the length */`
                                    state = JSDocState.BeginningOfLine;
                                    margin = undefined;
                                }
                                else {
                                    pushComment(scanner.getTokenText());
                                }
                                break;
                            case SyntaxKind::NewLineTrivia:
                                comments.push_back(scanner.getTokenText());
                                state = JSDocState.BeginningOfLine;
                                indent = 0;
                                break;
                            case SyntaxKind::AsteriskToken:
                                auto asterisk = scanner.getTokenText();
                                if (state == JSDocState.SawAsterisk || state == JSDocState.SavingComments) {
                                    // If we've already seen an asterisk, then we can no longer parse a tag on this line
                                    state = JSDocState.SavingComments;
                                    pushComment(asterisk);
                                }
                                else {
                                    // Ignore the first asterisk on a line
                                    state = JSDocState.SawAsterisk;
                                    indent += asterisk.size();
                                }
                                break;
                            case SyntaxKind::WhitespaceTrivia:
                                // only collect whitespace if we're already saving comments or have just crossed the comment indent margin
                                auto whitespace = scanner.getTokenText();
                                if (state == JSDocState.SavingComments) {
                                    comments.push(whitespace);
                                }
                                else if (margin != undefined && indent + whitespace.size() > margin) {
                                    comments.push(whitespace.slice(margin - indent));
                                }
                                indent += whitespace.size();
                                break;
                            case SyntaxKind::EndOfFileToken:
                                break loop;
                            default:
                                // Anything else is doc comment text. We just save it. Because it
                                // wasn't a tag, we can no longer parse a tag on this line until we hit the next
                                // line break.
                                state = JSDocState.SavingComments;
                                pushComment(scanner.getTokenText());
                                break;
                        }
                        nextTokenJSDoc();
                    }
                    removeLeadingNewlines(comments);
                    removeTrailingWhitespace(comments);
                    return createJSDocComment();
                });

                auto removeLeadingNewlines(std::vector<string> comments) {
                    while (comments.size() && (comments[0] == "\n" || comments[0] == "\r")) {
                        comments.shift();
                    }
                }

                auto removeTrailingWhitespace(std::vector<string> comments) {
                    while (comments.size() && comments[comments.size() - 1].trim() == string()) {
                        comments.pop();
                    }
                }

                auto createJSDocComment() -> JSDoc {
                    auto comment = comments.size() ? comments.join(string()) : undefined;
                    auto tagsArray = tags && createNodeArray(tags, tagsPos, tagsEnd);
                    return finishNode(factory.createJSDocComment(comment, tagsArray), start, end);
                }

                auto isNextNonwhitespaceTokenEndOfFile() -> boolean {
                    // We must use infinite lookahead,.as<there>() could be any number of newlines :(
                    while (true) {
                        nextTokenJSDoc();
                        if (token() == SyntaxKind::EndOfFileToken) {
                            return true;
                        }
                        if (!(token() == SyntaxKind::WhitespaceTrivia || token() == SyntaxKind::NewLineTrivia)) {
                            return false;
                        }
                    }
                }

                auto skipWhitespace() -> void {
                    if (token() == SyntaxKind::WhitespaceTrivia || token() == SyntaxKind::NewLineTrivia) {
                        if (lookAhead<boolean>(std::bind(&Parser::isNextNonwhitespaceTokenEndOfFile, this))) {
                            return; // Don't skip whitespace prior to EoF (or end of comment) - that shouldn't be included in any node's range
                        }
                    }
                    while (token() == SyntaxKind::WhitespaceTrivia || token() == SyntaxKind::NewLineTrivia) {
                        nextTokenJSDoc();
                    }
                }

                auto skipWhitespaceOrAsterisk() -> string {
                    if (token() == SyntaxKind::WhitespaceTrivia || token() == SyntaxKind::NewLineTrivia) {
                        if (lookAhead<boolean>(std::bind(&Parser::isNextNonwhitespaceTokenEndOfFile, this))) {
                            return string(); // Don't skip whitespace prior to EoF (or end of comment) - that shouldn't be included in any node's range
                        }
                    }

                    auto precedingLineBreak = scanner.hasPrecedingLineBreak();
                    auto seenLineBreak = false;
                    auto indentText = string();
                    while ((precedingLineBreak && token() == SyntaxKind::AsteriskToken) || token() == SyntaxKind::WhitespaceTrivia || token() == SyntaxKind::NewLineTrivia) {
                        indentText += scanner.getTokenText();
                        if (token() == SyntaxKind::NewLineTrivia) {
                            precedingLineBreak = true;
                            seenLineBreak = true;
                            indentText = string();
                        }
                        else if (token() == SyntaxKind::AsteriskToken) {
                            precedingLineBreak = false;
                        }
                        nextTokenJSDoc();
                    }
                    return seenLineBreak ? indentText : string();
                }

                auto parseTag(number margin) {
                    Debug::_assert(token() == SyntaxKind::AtToken);
                    auto start = scanner.getTokenPos();
                    nextTokenJSDoc();

                    auto tagName = parseJSDocIdentifierName(/*message*/ undefined);
                    auto indentText = skipWhitespaceOrAsterisk();

                    auto JSDocTag tag;
                    switch (tagName.escapedText) {
                        case "author":
                            tag = parseAuthorTag(start, tagName, margin, indentText);
                            break;
                        case "implements":
                            tag = parseImplementsTag(start, tagName, margin, indentText);
                            break;
                        case "augments":
                        case "extends":
                            tag = parseAugmentsTag(start, tagName, margin, indentText);
                            break;
                        case "class":
                        case "constructor":
                            tag = parseSimpleTag(start, factory.createJSDocClassTag, tagName, margin, indentText);
                            break;
                        case "public":
                            tag = parseSimpleTag(start, factory.createJSDocPublicTag, tagName, margin, indentText);
                            break;
                        case "private":
                            tag = parseSimpleTag(start, factory.createJSDocPrivateTag, tagName, margin, indentText);
                            break;
                        case "protected":
                            tag = parseSimpleTag(start, factory.createJSDocProtectedTag, tagName, margin, indentText);
                            break;
                        case "readonly":
                            tag = parseSimpleTag(start, factory.createJSDocReadonlyTag, tagName, margin, indentText);
                            break;
                        case "deprecated":
                            hasDeprecatedTag = true;
                            tag = parseSimpleTag(start, factory.createJSDocDeprecatedTag, tagName, margin, indentText);
                            break;
                        case "this":
                            tag = parseThisTag(start, tagName, margin, indentText);
                            break;
                        case "enum":
                            tag = parseEnumTag(start, tagName, margin, indentText);
                            break;
                        case "arg":
                        case "argument":
                        case "param":
                            return parseParameterOrPropertyTag(start, tagName, PropertyLikeParse.Parameter, margin);
                        case "return":
                        case "returns":
                            tag = parseReturnTag(start, tagName, margin, indentText);
                            break;
                        case "template":
                            tag = parseTemplateTag(start, tagName, margin, indentText);
                            break;
                        case "type":
                            tag = parseTypeTag(start, tagName, margin, indentText);
                            break;
                        case "typedef":
                            tag = parseTypedefTag(start, tagName, margin, indentText);
                            break;
                        case "callback":
                            tag = parseCallbackTag(start, tagName, margin, indentText);
                            break;
                        case "see":
                            tag = parseSeeTag(start, tagName, margin, indentText);
                            break;
                        default:
                            tag = parseUnknownTag(start, tagName, margin, indentText);
                            break;
                    }
                    return tag;
                }

                auto parseTrailingTagComments(number pos, number end, number margin, string indentText) {
                    // some tags, like typedef and callback, have already parsed their comments earlier
                    if (!indentText) {
                        margin += end - pos;
                    }
                    return parseTagComments(margin, indentText.slice(margin));
                }

                auto parseTagComments(number indent, string initialMargin) -> string {
                    auto std::vector<string> = [] comments;
                    auto state = JSDocState.BeginningOfLine;
                    auto previousWhitespace = true;
                    auto number margin;
                    auto pushComment(string text) {
                        if (!margin) {
                            margin = indent;
                        }
                        comments.push(text);
                        indent += text.size();
                    }
                    if (initialMargin != undefined) {
                        // jump straight to saving comments if there is some initial indentation
                        if (initialMargin != string()) {
                            pushComment(initialMargin);
                        }
                        state = JSDocState.SawAsterisk;
                    }
                    auto tok = token().as<SyntaxKind>();
                    while loop (true) {
                        switch (tok) {
                            case SyntaxKind::NewLineTrivia:
                                state = JSDocState.BeginningOfLine;
                                // don't use pushComment here because we want to keep the margin unchanged
                                comments.push(scanner.getTokenText());
                                indent = 0;
                                break;
                            case SyntaxKind::AtToken:
                                if (state == JSDocState.SavingBackticks || !previousWhitespace && state == JSDocState.SavingComments) {
                                    // @ doesn't start a new tag inside ``, and inside a comment, only after whitespace
                                    comments.push(scanner.getTokenText());
                                    break;
                                }
                                scanner.setTextPos(scanner.getTextPos() - 1);
                                // falls through
                            case SyntaxKind::EndOfFileToken:
                                // Done
                                break loop;
                            case SyntaxKind::WhitespaceTrivia:
                                if (state == JSDocState.SavingComments || state == JSDocState.SavingBackticks) {
                                    pushComment(scanner.getTokenText());
                                }
                                else {
                                    auto whitespace = scanner.getTokenText();
                                    // if the whitespace crosses the margin, take only the whitespace that passes the margin
                                    if (margin != undefined && indent + whitespace.size() > margin) {
                                        comments.push(whitespace.slice(margin - indent));
                                    }
                                    indent += whitespace.size();
                                }
                                break;
                            case SyntaxKind::OpenBraceToken:
                                state = JSDocState.SavingComments;
                                if (lookAhead<boolean>(() => nextTokenJSDoc() == SyntaxKind::AtToken && scanner.tokenIsIdentifierOrKeyword(nextTokenJSDoc()) && scanner.getTokenText() == "link")) {
                                    pushComment(scanner.getTokenText());
                                    nextTokenJSDoc();
                                    pushComment(scanner.getTokenText());
                                    nextTokenJSDoc();
                                }
                                pushComment(scanner.getTokenText());
                                break;
                            case SyntaxKind::BacktickToken:
                                if (state == JSDocState.SavingBackticks) {
                                    state = JSDocState.SavingComments;
                                }
                                else {
                                    state = JSDocState.SavingBackticks;
                                }
                                pushComment(scanner.getTokenText());
                                break;
                            case SyntaxKind::AsteriskToken:
                                if (state == JSDocState.BeginningOfLine) {
                                    // leading asterisks start recording on the *next* (non-whitespace) token
                                    state = JSDocState.SawAsterisk;
                                    indent += 1;
                                    break;
                                }
                                // record the *.as<a>() comment
                                // falls through
                            default:
                                if (state != JSDocState.SavingBackticks) {
                                    state = JSDocState.SavingComments; // leading identifiers start recording.as<well>()
                                }
                                pushComment(scanner.getTokenText());
                                break;
                        }
                        previousWhitespace = token() == SyntaxKind::WhitespaceTrivia;
                        tok = nextTokenJSDoc();
                    }

                    removeLeadingNewlines(comments);
                    removeTrailingWhitespace(comments);
                    return comments.size() == 0 ? undefined : comments.join(string());
                }

                auto parseUnknownTag(number start, Identifier tagName, number indent, string indentText) {
                    auto end = getNodePos();
                    return finishNode(factory.createJSDocUnknownTag(tagName, parseTrailingTagComments(start, end, indent, indentText)), start, end);
                }

                auto addTag(JSDocTag tag) -> void {
                    if (!tag) {
                        return;
                    }
                    if (!tags) {
                        tags = [tag];
                        tagsPos = tag.pos;
                    }
                    else {
                        tags.push(tag);
                    }
                    tagsEnd = tag.end;
                }

                auto tryParseTypeExpression() -> JSDocTypeExpression {
                    skipWhitespaceOrAsterisk();
                    return token() == SyntaxKind::OpenBraceToken ? parseJSDocTypeExpression() : undefined;
                }

                auto parseBracketNameInPropertyAndParamTag() -> { EntityName name, boolean isBracketed } {
                    // Looking for something like '[foo]', 'foo', '[foo.bar]' or 'foo.bar'
                    auto isBracketed = parseOptionalJsdoc(SyntaxKind::OpenBracketToken);
                    if (isBracketed) {
                        skipWhitespace();
                    }
                    // a markdown-quoted name: `arg` is not legal jsdoc, but occurs in the wild
                    auto isBackquoted = parseOptionalJsdoc(SyntaxKind::BacktickToken);
                    auto name = parseJSDocEntityName();
                    if (isBackquoted) {
                        parseExpectedTokenJSDoc(SyntaxKind::BacktickToken);
                    }
                    if (isBracketed) {
                        skipWhitespace();
                        // May have an optional default, e.g. '[foo = 42]'
                        if (parseOptionalToken(SyntaxKind::EqualsToken)) {
                            parseExpression();
                        }

                        parseExpected(SyntaxKind::CloseBracketToken);
                    }

                    return { name, isBracketed };
                }

                auto isObjectOrObjectArrayTypeReference(TypeNode node) -> boolean {
                    switch (node.kind) {
                        case SyntaxKind::ObjectKeyword:
                            return true;
                        case SyntaxKind::ArrayType:
                            return isObjectOrObjectArrayTypeReference(node.as<ArrayTypeNode>().elementType);
                        default:
                            return isTypeReferenceNode(node) && ts.isIdentifier(node.typeName) && node.typeName.escapedText == "Object" && !node.typeArguments;
                    }
                }

                auto parseParameterOrPropertyTag(number start, Identifier tagName, PropertyLikeParse target, number indent) -> JSDocParameterTag | JSDocPropertyTag {
                    auto typeExpression = tryParseTypeExpression();
                    auto isNameFirst = !typeExpression;
                    skipWhitespaceOrAsterisk();

                    auto { name, isBracketed } = parseBracketNameInPropertyAndParamTag();
                    auto indentText = skipWhitespaceOrAsterisk();

                    if (isNameFirst) {
                        typeExpression = tryParseTypeExpression();
                    }

                    auto comment = parseTrailingTagComments(start, getNodePos(), indent, indentText);

                    auto nestedTypeLiteral = target != PropertyLikeParse.CallbackParameter && parseNestedTypeLiteral(typeExpression, name, target, indent);
                    if (nestedTypeLiteral) {
                        typeExpression = nestedTypeLiteral;
                        isNameFirst = true;
                    }
                    auto result = target == PropertyLikeParse.Property
                        ? factory.createJSDocPropertyTag(tagName, name, isBracketed, typeExpression, isNameFirst, comment)
                        : factory.createJSDocParameterTag(tagName, name, isBracketed, typeExpression, isNameFirst, comment);
                    return finishNode(result, start);
                }

                auto parseNestedTypeLiteral(JSDocTypeExpression typeExpression, EntityName name, PropertyLikeParse target, number indent) {
                    if (typeExpression && isObjectOrObjectArrayTypeReference(typeExpression.type)) {
                        auto pos = getNodePos();
                        auto JSDocPropertyLikeTag child | JSDocTypeTag | false;
                        auto std::vector<JSDocPropertyLikeTag> children;
                        while (child = tryParse(() => parseChildParameterOrPropertyTag(target, indent, name))) {
                            if (child.kind == SyntaxKind::JSDocParameterTag || child.kind == SyntaxKind::JSDocPropertyTag) {
                                children = append(children, child);
                            }
                        }
                        if (children) {
                            auto literal = finishNode(factory.createJSDocTypeLiteral(children, typeExpression.type.kind == SyntaxKind::ArrayType), pos);
                            return finishNode(factory.createJSDocTypeExpression(literal), pos);
                        }
                    }
                }

                auto parseReturnTag(number start, Identifier tagName, number indent, string indentText) -> JSDocReturnTag {
                    if (some(tags, isJSDocReturnTag)) {
                        parseErrorAt(tagName.pos, scanner.getTokenPos(), Diagnostics::_0_tag_already_specified, tagName.escapedText);
                    }

                    auto typeExpression = tryParseTypeExpression();
                    auto end = getNodePos();
                    return finishNode(factory.createJSDocReturnTag(tagName, typeExpression, parseTrailingTagComments(start, end, indent, indentText)), start, end);
                }

                auto parseTypeTag(number start, Identifier tagName, number indent, string indentText) -> JSDocTypeTag {
                    if (some(tags, isJSDocTypeTag)) {
                        parseErrorAt(tagName.pos, scanner.getTokenPos(), Diagnostics::_0_tag_already_specified, tagName.escapedText);
                    }

                    auto typeExpression = parseJSDocTypeExpression(/*mayOmitBraces*/ true);
                    auto end = getNodePos();
                    auto comments = indent != undefined && indentText != undefined ? parseTrailingTagComments(start, end, indent, indentText) : undefined;
                    return finishNode(factory.createJSDocTypeTag(tagName, typeExpression, comments), start, end);
                }

                auto parseSeeTag(number start, Identifier tagName, number indent, string indentText) -> JSDocSeeTag {
                    auto nameExpression = parseJSDocNameReference();
                    auto end = getNodePos();
                    auto comments = indent != undefined && indentText != undefined ? parseTrailingTagComments(start, end, indent, indentText) : undefined;
                    return finishNode(factory.createJSDocSeeTag(tagName, nameExpression, comments), start, end);
                }

                auto parseAuthorTag(number start, Identifier tagName, number indent, string indentText) -> JSDocAuthorTag {
                    auto comments = parseAuthorNameAndEmail() + (parseTrailingTagComments(start, end, indent, indentText) || string());
                    return finishNode(factory.createJSDocAuthorTag(tagName, comments || undefined), start);
                }

                auto parseAuthorNameAndEmail() -> string {
                    auto std::vector<string> = [] comments;
                    auto inEmail = false;
                    auto token = scanner.getToken();
                    while (token != SyntaxKind::EndOfFileToken && token != SyntaxKind::NewLineTrivia) {
                        if (token == SyntaxKind::LessThanToken) {
                            inEmail = true;
                        }
                        else if (token == SyntaxKind::AtToken && !inEmail) {
                            break;
                        }
                        else if (token == SyntaxKind::GreaterThanToken && inEmail) {
                            comments.push(scanner.getTokenText());
                            scanner.setTextPos(scanner.getTokenPos() + 1);
                            break;
                        }
                        comments.push(scanner.getTokenText());
                        token = nextTokenJSDoc();
                    }

                    return comments.join(string());
                }

                auto parseImplementsTag(number start, Identifier tagName, number margin, string indentText) -> JSDocImplementsTag {
                    auto className = parseExpressionWithTypeArgumentsForAugments();
                    auto end = getNodePos();
                    return finishNode(factory.createJSDocImplementsTag(tagName, className, parseTrailingTagComments(start, end, margin, indentText)), start, end);
                }

                auto parseAugmentsTag(number start, Identifier tagName, number margin, string indentText) -> JSDocAugmentsTag {
                    auto className = parseExpressionWithTypeArgumentsForAugments();
                    auto end = getNodePos();
                    return finishNode(factory.createJSDocAugmentsTag(tagName, className, parseTrailingTagComments(start, end, margin, indentText)), start, end);
                }

                auto parseExpressionWithTypeArgumentsForAugments() -> ExpressionWithTypeArguments & { Identifier expression | PropertyAccessEntityNameExpression } {
                    auto usedBrace = parseOptional(SyntaxKind::OpenBraceToken);
                    auto pos = getNodePos();
                    auto expression = parsePropertyAccessEntityNameExpression();
                    auto typeArguments = tryParseTypeArguments();
                    auto node = factory.createExpressionWithTypeArguments(expression, typeArguments).as<ExpressionWithTypeArguments>() & { Identifier expression | PropertyAccessEntityNameExpression };
                    auto res = finishNode(node, pos);
                    if (usedBrace) {
                        parseExpected(SyntaxKind::CloseBraceToken);
                    }
                    return res;
                }

                auto parsePropertyAccessEntityNameExpression() {
                    auto pos = getNodePos();
                    auto Identifier node | PropertyAccessEntityNameExpression = parseJSDocIdentifierName();
                    while (parseOptional(SyntaxKind::DotToken)) {
                        auto name = parseJSDocIdentifierName();
                        node = finishNode(factory.createPropertyAccessExpression(node, name), pos).as<PropertyAccessEntityNameExpression>();
                    }
                    return node;
                }

                auto parseSimpleTag(number start, createTag: (Identifier tagName, string comment) => JSDocTag, Identifier tagName, number margin, string indentText) -> JSDocTag {
                    auto end = getNodePos();
                    return finishNode(createTag(tagName, parseTrailingTagComments(start, end, margin, indentText)), start, end);
                }

                auto parseThisTag(number start, Identifier tagName, number margin, string indentText) -> JSDocThisTag {
                    auto typeExpression = parseJSDocTypeExpression(/*mayOmitBraces*/ true);
                    skipWhitespace();
                    auto end = getNodePos();
                    return finishNode(factory.createJSDocThisTag(tagName, typeExpression, parseTrailingTagComments(start, end, margin, indentText)), start, end);
                }

                auto parseEnumTag(number start, Identifier tagName, number margin, string indentText) -> JSDocEnumTag {
                    auto typeExpression = parseJSDocTypeExpression(/*mayOmitBraces*/ true);
                    skipWhitespace();
                    auto end = getNodePos();
                    return finishNode(factory.createJSDocEnumTag(tagName, typeExpression, parseTrailingTagComments(start, end, margin, indentText)), start, end);
                }

                auto parseTypedefTag(number start, Identifier tagName, number indent, string indentText) -> JSDocTypedefTag {
                    auto JSDocTypeExpression typeExpression | JSDocTypeLiteral = tryParseTypeExpression();
                    skipWhitespaceOrAsterisk();

                    auto fullName = parseJSDocTypeNameWithNamespace();
                    skipWhitespace();
                    auto comment = parseTagComments(indent);

                    auto number end;
                    if (!typeExpression || isObjectOrObjectArrayTypeReference(typeExpression.type)) {
                        auto JSDocTypeTag child | JSDocPropertyTag | false;
                        auto JSDocTypeTag childTypeTag;
                        auto std::vector<JSDocPropertyTag> jsDocPropertyTags;
                        auto hasChildren = false;
                        while (child = tryParse(() => parseChildPropertyTag(indent))) {
                            hasChildren = true;
                            if (child.kind == SyntaxKind::JSDocTypeTag) {
                                if (childTypeTag) {
                                    parseErrorAtCurrentToken(Diagnostics::A_JSDoc_typedef_comment_may_not_contain_multiple_type_tags);
                                    auto lastError = lastOrUndefined(parseDiagnostics);
                                    if (lastError) {
                                        addRelatedInfo(
                                            lastError,
                                            createDetachedDiagnostic(fileName, 0, 0, Diagnostics::The_tag_was_first_specified_here)
                                        );
                                    }
                                    break;
                                }
                                else {
                                    childTypeTag = child;
                                }
                            }
                            else {
                                jsDocPropertyTags = append(jsDocPropertyTags, child);
                            }
                        }
                        if (hasChildren) {
                            auto isArrayType = typeExpression && typeExpression.type.kind == SyntaxKind::ArrayType;
                            auto jsdocTypeLiteral = factory.createJSDocTypeLiteral(jsDocPropertyTags, isArrayType);
                            typeExpression = childTypeTag && childTypeTag.typeExpression && !isObjectOrObjectArrayTypeReference(childTypeTag.typeExpression.type) ?
                                childTypeTag.typeExpression :
                                finishNode(jsdocTypeLiteral, start);
                            end = typeExpression.end;
                        }
                    }

                    // Only include the characters between the name end and the next token if a comment was actually parsed out - otherwise it's just whitespace
                    end = end || comment != undefined ?
                        getNodePos() :
                        (fullName ?? typeExpression ?? tagName).end;

                    if (!comment) {
                        comment = parseTrailingTagComments(start, end, indent, indentText);
                    }

                    auto typedefTag = factory.createJSDocTypedefTag(tagName, typeExpression, fullName, comment);
                    return finishNode(typedefTag, start, end);
                }

                auto parseJSDocTypeNameWithNamespace(boolean nested) {
                    auto pos = scanner.getTokenPos();
                    if (!scanner.tokenIsIdentifierOrKeyword(token())) {
                        return undefined;
                    }
                    auto typeNameOrNamespaceName = parseJSDocIdentifierName();
                    if (parseOptional(SyntaxKind::DotToken)) {
                        auto body = parseJSDocTypeNameWithNamespace(/*nested*/ true);
                        auto jsDocNamespaceNode = factory.createModuleDeclaration(
                            /*decorators*/ undefined,
                            /*modifiers*/ undefined,
                            typeNameOrNamespaceName,
                            body,
                            nested ? NodeFlags::NestedNamespace : undefined
                        ).as<JSDocNamespaceDeclaration>();
                        return finishNode(jsDocNamespaceNode, pos);
                    }

                    if (nested) {
                        typeNameOrNamespaceName.isInJSDocNamespace = true;
                    }
                    return typeNameOrNamespaceName;
                }


                auto parseCallbackTagParameters(number indent) {
                    auto pos = getNodePos();
                    auto JSDocParameterTag child | false;
                    auto parameters;
                    while (child = tryParse(() => parseChildParameterOrPropertyTag(PropertyLikeParse.CallbackParameter, indent).as<JSDocParameterTag>())) {
                        parameters = append(parameters, child);
                    }
                    return createNodeArray(parameters || [], pos);
                }

                auto parseCallbackTag(number start, Identifier tagName, number indent, string indentText) -> JSDocCallbackTag {
                    auto fullName = parseJSDocTypeNameWithNamespace();
                    skipWhitespace();
                    auto comment = parseTagComments(indent);
                    auto parameters = parseCallbackTagParameters(indent);
                    auto returnTag = tryParse(() => {
                        if (parseOptionalJsdoc(SyntaxKind::AtToken)) {
                            auto tag = parseTag(indent);
                            if (tag && tag.kind == SyntaxKind::JSDocReturnTag) {
                                return tag.as<JSDocReturnTag>();
                            }
                        }
                    });
                    auto typeExpression = finishNode(factory.createJSDocSignature(/*typeParameters*/ undefined, parameters, returnTag), start);
                    auto end = getNodePos();
                    if (!comment) {
                        comment = parseTrailingTagComments(start, end, indent, indentText);
                    }
                    return finishNode(factory.createJSDocCallbackTag(tagName, typeExpression, fullName, comment), start, end);
                }

                auto escapedTextsEqual(EntityName a, EntityName b) -> boolean {
                    while (!ts.isIdentifier(a) || !ts.isIdentifier(b)) {
                        if (!ts.isIdentifier(a) && !ts.isIdentifier(b) && a.right.escapedText == b.right.escapedText) {
                            a = a.left;
                            b = b.left;
                        }
                        else {
                            return false;
                        }
                    }
                    return a.escapedText == b.escapedText;
                }

                auto parseChildPropertyTag(number indent) {
                    return parseChildParameterOrPropertyTag(PropertyLikeParse.Property, indent).as<JSDocTypeTag>() | JSDocPropertyTag | false;
                }

                auto parseChildParameterOrPropertyTag(PropertyLikeParse target, number indent, EntityName name) -> JSDocTypeTag | JSDocPropertyTag | JSDocParameterTag | false {
                    auto canParseTag = true;
                    auto seenAsterisk = false;
                    while (true) {
                        switch (nextTokenJSDoc()) {
                            case SyntaxKind::AtToken:
                                if (canParseTag) {
                                    auto child = tryParseChildTag(target, indent);
                                    if (child && (child.kind == SyntaxKind::JSDocParameterTag || child.kind == SyntaxKind::JSDocPropertyTag) &&
                                        target != PropertyLikeParse.CallbackParameter &&
                                        name && (ts.isIdentifier(child.name) || !escapedTextsEqual(name, child.name.left))) {
                                        return false;
                                    }
                                    return child;
                                }
                                seenAsterisk = false;
                                break;
                            case SyntaxKind::NewLineTrivia:
                                canParseTag = true;
                                seenAsterisk = false;
                                break;
                            case SyntaxKind::AsteriskToken:
                                if (seenAsterisk) {
                                    canParseTag = false;
                                }
                                seenAsterisk = true;
                                break;
                            case SyntaxKind::Identifier:
                                canParseTag = false;
                                break;
                            case SyntaxKind::EndOfFileToken:
                                return false;
                        }
                    }
                }

                auto tryParseChildTag(PropertyLikeParse target, number indent) -> JSDocTypeTag | JSDocPropertyTag | JSDocParameterTag | false {
                    Debug::_assert(token() == SyntaxKind::AtToken);
                    auto start = scanner.getStartPos();
                    nextTokenJSDoc();

                    auto tagName = parseJSDocIdentifierName();
                    skipWhitespace();
                    auto PropertyLikeParse t;
                    switch (tagName.escapedText) {
                        case "type":
                            return target == PropertyLikeParse.Property && parseTypeTag(start, tagName);
                        case "prop":
                        case "property":
                            t = PropertyLikeParse.Property;
                            break;
                        case "arg":
                        case "argument":
                        case "param":
                            t = PropertyLikeParse.Parameter | PropertyLikeParse.CallbackParameter;
                            break;
                        default:
                            return false;
                    }
                    if (!(target & t)) {
                        return false;
                    }
                    return parseParameterOrPropertyTag(start, tagName, target, indent);
                }

                auto parseTemplateTagTypeParameter() {
                    auto typeParameterPos = getNodePos();
                    auto name = parseJSDocIdentifierName(Diagnostics::Unexpected_token_A_type_parameter_name_was_expected_without_curly_braces);
                    if (nodeIsMissing(name)) {
                        return undefined;
                    }
                    return finishNode(factory.createTypeParameterDeclaration(name, /*constraint*/ undefined, /*defaultType*/ undefined), typeParameterPos);
                }

                auto parseTemplateTagTypeParameters() {
                    auto pos = getNodePos();
                    auto typeParameters = [];
                    do {
                        skipWhitespace();
                        auto node = parseTemplateTagTypeParameter();
                        if (node != undefined) {
                            typeParameters.push(node);
                        }
                        skipWhitespaceOrAsterisk();
                    } while (parseOptionalJsdoc(SyntaxKind::CommaToken));
                    return createNodeArray(typeParameters, pos);
                }

                auto parseTemplateTag(number start, Identifier tagName, number indent, string indentText) -> JSDocTemplateTag {
                    // The template tag looks like one of the following:
                    //   @template T,U,V
                    //   @template {Constraint} T
                    //
                    // According to the [closure docs](https://github.com/google/closure-compiler/wiki/Generic-Types#multiple-bounded-template-types) ->
                    //   > Multiple bounded generics cannot be declared on the same line. For the sake of clarity, if multiple templates share the same
                    //   > type bound they must be declared on separate lines.
                    //
                    // Determine TODO whether we should enforce this in the checker.
                    // Consider TODO moving the `constraint` to the first type parameter.as<we>() could then remove `getEffectiveConstraintOfTypeParameter`.
                    // Consider TODO only parsing a single type parameter if there is a constraint.
                    auto constraint = token() == SyntaxKind::OpenBraceToken ? parseJSDocTypeExpression() : undefined;
                    auto typeParameters = parseTemplateTagTypeParameters();
                    auto end = getNodePos();
                    return finishNode(factory.createJSDocTemplateTag(tagName, constraint, typeParameters, parseTrailingTagComments(start, end, indent, indentText)), start, end);
                }

                auto parseOptionalJsdoc(SyntaxKind t) -> boolean {
                    if (token() == t) {
                        nextTokenJSDoc();
                        return true;
                    }
                    return false;
                }

                auto parseJSDocEntityName() -> EntityName {
                    auto EntityName entity = parseJSDocIdentifierName();
                    if (parseOptional(SyntaxKind::OpenBracketToken)) {
                        parseExpected(SyntaxKind::CloseBracketToken);
                        // Note that y[] is accepted.as<an>() entity name, but the postfix brackets are not saved for checking.
                        // Technically usejsdoc.org requires them for specifying a property of a type equivalent to Array<{ ... x}>
                        // but it's not worth it to enforce that restriction.
                    }
                    while (parseOptional(SyntaxKind::DotToken)) {
                        auto name = parseJSDocIdentifierName();
                        if (parseOptional(SyntaxKind::OpenBracketToken)) {
                            parseExpected(SyntaxKind::CloseBracketToken);
                        }
                        entity = createQualifiedName(entity, name);
                    }
                    return entity;
                }

                auto parseJSDocIdentifierName(DiagnosticMessage message) -> Identifier {
                    if (!scanner.tokenIsIdentifierOrKeyword(token())) {
                        return createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ !message, message || Diagnostics::Identifier_expected);
                    }

                    identifierCount++;
                    auto pos = scanner.getTokenPos();
                    auto end = scanner.getTextPos();
                    auto originalKeywordKind = token();
                    auto text = internIdentifier(scanner.getTokenValue());
                    auto result = finishNode(factory.createIdentifier(text, /*typeArguments*/ undefined, originalKeywordKind), pos, end);
                    nextTokenJSDoc();
                    return result;
                }
            }

            // End JSDoc namespace
        };

        namespace IncrementalParser {
            auto updateSourceFile(SourceFile sourceFile, string newText, TextChangeRange textChangeRange, boolean aggressiveChecks) -> SourceFile {
                aggressiveChecks = aggressiveChecks || Debug::shouldAssert(AssertionLevel::Aggressive);

                checkChangeRange(sourceFile, newText, textChangeRange, aggressiveChecks);
                if (textChangeRangeIsUnchanged(textChangeRange)) {
                    // if the text didn't change, then we can just return our current source file as-is.
                    return sourceFile;
                }

                if (sourceFile.statements.size() == 0) {
                    // If we don't have any statements in the current source file, then there's no real
                    // way to incrementally parse.  So just do a full parse instead.
                    return Parser::parseSourceFile(sourceFile.fileName, newText, sourceFile.languageVersion, undefined, /*setParentNodes*/ true, sourceFile.scriptKind);
                }

                // Make sure we're not trying to incrementally update a source file more than once.  Once
                // we do an update the original source file is considered unusable from that point onwards.
                //
                // This is because we do incremental parsing in-place.  i.e. we take nodes from the old
                // tree and give them new positions and parents.  From that point on, trusting the old
                // tree at all is not possible.as<far>() too much of it may violate invariants.
                auto incrementalSourceFile = <IncrementalNode>sourceFile.as<Node>();
                Debug::_assert(!incrementalSourceFile.hasBeenIncrementallyParsed);
                incrementalSourceFile.hasBeenIncrementallyParsed = true;
                Parser::fixupParentReferences(incrementalSourceFile);
                auto oldText = sourceFile.text;
                auto syntaxCursor = createSyntaxCursor(sourceFile);

                // Make the actual change larger so that we know to reparse anything whose lookahead
                // might have intersected the change.
                auto changeRange = extendToAffectedRange(sourceFile, textChangeRange);
                checkChangeRange(sourceFile, newText, changeRange, aggressiveChecks);

                // Ensure that extending the affected range only moved the start of the change range
                // earlier in the file.
                Debug::_assert(changeRange.span.start <= textChangeRange.span.start);
                Debug::_assert(textSpanEnd(changeRange.span) == textSpanEnd(textChangeRange.span));
                Debug::_assert(textSpanEnd(textChangeRangeNewSpan(changeRange)) == textSpanEnd(textChangeRangeNewSpan(textChangeRange)));

                // The is the amount the nodes after the edit range need to be adjusted.  It can be
                // positive (if the edit added characters), negative (if the edit deleted characters)
                // or zero (if this was a pure overwrite with nothing added/removed).
                auto delta = textChangeRangeNewSpan(changeRange).size() - changeRange.span.size();

                // If we added or removed characters during the edit, then we need to go and adjust all
                // the nodes after the edit.  Those nodes may move forward (if we inserted chars) or they
                // may move backward (if we deleted chars).
                //
                // Doing this helps us out in two ways.  First, it means that any nodes/tokens we want
                // to reuse are already at the appropriate position in the new text.  That way when we
                // reuse them, we don't have to figure out if they need to be adjusted.  Second, it makes
                // it very easy to determine if we can reuse a node.  If the node's position is at where
                // we are in the text, then we can reuse it.  Otherwise we can't.  If the node's position
                // is ahead of us, then we'll need to rescan tokens.  If the node's position is behind
                // us, then we'll need to skip it or crumble it.as<appropriate>()
                //
                // We will also adjust the positions of nodes that intersect the change range.as<well>().
                // By doing this, we ensure that all the positions in the old tree are consistent, not
                // just the positions of nodes entirely before/after the change range.  By being
                // consistent, we can then easily map from positions to nodes in the old tree easily.
                //
                // Also, mark any syntax elements that intersect the changed span.  We know, up front,
                // that we cannot reuse these elements.
                updateTokenPositionsAndMarkElements(incrementalSourceFile,
                    changeRange.span.start, textSpanEnd(changeRange.span), textSpanEnd(textChangeRangeNewSpan(changeRange)), delta, oldText, newText, aggressiveChecks);

                // Now that we've set up our internal incremental state just proceed and parse the
                // source file in the normal fashion.  When possible the parser will retrieve and
                // reuse nodes from the old tree.
                //
                // passing Note in 'true' for setNodeParents is very important.  When incrementally
                // parsing, we will be reusing nodes from the old tree, and placing it into new
                // parents.  If we don't set the parents now, we'll end up with an observably
                // inconsistent tree.  Setting the parents on the new tree should be very fast.  We
                // will immediately bail out of walking any subtrees when we can see that their parents
                // are already correct.
                auto result = Parser::parseSourceFile(sourceFile.fileName, newText, sourceFile.languageVersion, syntaxCursor, /*setParentNodes*/ true, sourceFile.scriptKind);
                result.commentDirectives = getNewCommentDirectives(
                    sourceFile.commentDirectives,
                    result.commentDirectives,
                    changeRange.span.start,
                    textSpanEnd(changeRange.span),
                    delta,
                    oldText,
                    newText,
                    aggressiveChecks
                );
                return result;
            }

            auto getNewCommentDirectives(
                std::vector<CommentDirective> oldDirectives,
                std::vector<CommentDirective> newDirectives,
                number changeStart,
                number changeRangeOldEnd,
                number delta,
                safe_string oldText,
                safe_string newText,
                boolean aggressiveChecks
            ) -> std::vector<CommentDirective> {
                if (!oldDirectives) return newDirectives;
                auto std::vector<CommentDirective> commentDirectives;
                auto addedNewlyScannedDirectives = false;

                auto addNewlyScannedDirectives = [&]() {
                    if (addedNewlyScannedDirectives) return;
                    addedNewlyScannedDirectives = true;
                    if (!commentDirectives) {
                        commentDirectives = newDirectives;
                    }
                    else if (newDirectives) {
                        commentDirectives.push(...newDirectives);
                    }
                };

                for (auto directive of oldDirectives) {
                    auto { range, type } = directive;
                    // Range before the change
                    if (range.end < changeStart) {
                        commentDirectives = append(commentDirectives, directive);
                    }
                    else if (range.pos > changeRangeOldEnd) {
                        addNewlyScannedDirectives();
                        // Node is entirely past the change range.  We need to move both its pos and
                        // end, forward or backward appropriately.
                        auto CommentDirective updatedDirective = {
                            range: { range.pos pos + delta, range.end end + delta },
                            type
                        };
                        commentDirectives = append(commentDirectives, updatedDirective);
                        if (aggressiveChecks) {
                            Debug::_assert(oldText.substring(range.pos, range.end) == newText.substring(updatedDirective.range.pos, updatedDirective.range.end));
                        }
                    }
                    // Ignore ranges that fall in change range
                }
                addNewlyScannedDirectives();
                return commentDirectives;
            }

            auto moveElementEntirelyPastChangeRange(IncrementalElement element, boolean isArray, number delta, string oldText, string newText, boolean aggressiveChecks) {
                if (isArray) {
                    visitArray(element.as<IncrementalNodeArray>());
                }
                else {
                    visitNode(element.as<IncrementalNode>());
                }
                return;

                auto visitNode(IncrementalNode node) {
                    auto text = string();
                    if (aggressiveChecks && shouldCheckNode(node)) {
                        text = oldText.substring(node->pos, node->end);
                    }

                    // Ditch any existing LS children we may have created.  This way we can avoid
                    // moving them forward.
                    if (node._children) {
                        node._children = undefined;
                    }

                    setTextRangePosEnd(node, node->pos + delta, node->end + delta);

                    if (aggressiveChecks && shouldCheckNode(node)) {
                        Debug::_assert(text == newText.substring(node->pos, node->end));
                    }

                    forEachChild(node, visitNode, visitArray);
                    if (hasJSDocNodes(node)) {
                        for (auto jsDocComment of node.jsDoc!) {
                            visitNode(<IncrementalNode>jsDocComment.as<Node>());
                        }
                    }
                    checkNodePositions(node, aggressiveChecks);
                }

                auto visitArray(IncrementalNodeArray array) {
                    array._children = undefined;
                    setTextRangePosEnd(array, array.pos + delta, array.end + delta);

                    for (auto node of array) {
                        visitNode(node);
                    }
                }
            }

            auto shouldCheckNode(Node node) {
                switch (node.kind) {
                    case SyntaxKind::StringLiteral:
                    case SyntaxKind::NumericLiteral:
                    case SyntaxKind::Identifier:
                        return true;
                }

                return false;
            }

            auto adjustIntersectingElement(IncrementalElement element, number changeStart, number changeRangeOldEnd, number changeRangeNewEnd, number delta) {
                Debug::_assert(element.end >= changeStart, "Adjusting an element that was entirely before the change range");
                Debug::_assert(element.pos <= changeRangeOldEnd, "Adjusting an element that was entirely after the change range");
                Debug::_assert(element.pos <= element.end);

                // We have an element that intersects the change range in some way.  It may have its
                // start, or its end (or both) in the changed range.  We want to adjust any part
                // that intersects such that the final tree is in a consistent state.  i.e. all
                // children have spans within the span of their parent, and all siblings are ordered
                // properly.

                // We may need to update both the 'pos' and the 'end' of the element.

                // If the 'pos' is before the start of the change, then we don't need to touch it.
                // If it isn't, then the 'pos' must be inside the change.  How we update it will
                // depend if delta is positive or negative. If delta is positive then we have
                // something like:
                //
                //  -------------------AAA-----------------
                //  -------------------BBBCCCCCCC-----------------
                //
                // In this case, we consider any node that started in the change range to still be
                // starting at the same position.
                //
                // however, if the delta is negative, then we instead have something like this:
                //
                //  -------------------XXXYYYYYYY-----------------
                //  -------------------ZZZ-----------------
                //
                // In this case, any element that started in the 'X' range will keep its position.
                // However any element that started after that will have their pos adjusted to be
                // at the end of the new range.  i.e. any node that started in the 'Y' range will
                // be adjusted to have their start at the end of the 'Z' range.
                //
                // The element will keep its position if possible.  Or Move backward to the new-end
                // if it's in the 'Y' range.
                auto pos = Math.min(element.pos, changeRangeNewEnd);

                // If the 'end' is after the change range, then we always adjust it by the delta
                // amount.  However, if the end is in the change range, then how we adjust it
                // will depend on if delta is positive or negative.  If delta is positive then we
                // have something like:
                //
                //  -------------------AAA-----------------
                //  -------------------BBBCCCCCCC-----------------
                //
                // In this case, we consider any node that ended inside the change range to keep its
                // end position.
                //
                // however, if the delta is negative, then we instead have something like this:
                //
                //  -------------------XXXYYYYYYY-----------------
                //  -------------------ZZZ-----------------
                //
                // In this case, any element that ended in the 'X' range will keep its position.
                // However any element that ended after that will have their pos adjusted to be
                // at the end of the new range.  i.e. any node that ended in the 'Y' range will
                // be adjusted to have their end at the end of the 'Z' range.
                auto end = element.end >= changeRangeOldEnd ?
                    // Element ends after the change range.  Always adjust the end pos.
                    element.end + delta :
                    // Element ends in the change range.  The element will keep its position if
                    // possible. Or Move backward to the new-end if it's in the 'Y' range.
                    Math.min(element.end, changeRangeNewEnd);

                Debug::_assert(pos <= end);
                if (element.parent) {
                    Debug::_assertGreaterThanOrEqual(pos, element.parent.pos);
                    Debug::_assertLessThanOrEqual(end, element.parent.end);
                }

                setTextRangePosEnd(element, pos, end);
            }

            auto checkNodePositions(Node node, boolean aggressiveChecks) {
                if (aggressiveChecks) {
                    auto pos = node->pos;
                    auto visitNode = (Node child) => {
                        Debug::_assert(child.pos >= pos);
                        pos = child.end;
                    };
                    if (hasJSDocNodes(node)) {
                        for (auto jsDocComment of node.jsDoc!) {
                            visitNode(jsDocComment);
                        }
                    }
                    forEachChild(node, visitNode);
                    Debug::_assert(pos <= node->end);
                }
            }

            auto updateTokenPositionsAndMarkElements(
                IncrementalNode sourceFile,
                number changeStart,
                number changeRangeOldEnd,
                number changeRangeNewEnd,
                number delta,
                string oldText,
                string newText,
                boolean aggressiveChecks) -> void {

                visitNode(sourceFile);
                return;

                auto visitNode(IncrementalNode child) {
                    Debug::_assert(child.pos <= child.end);
                    if (child.pos > changeRangeOldEnd) {
                        // Node is entirely past the change range.  We need to move both its pos and
                        // end, forward or backward appropriately.
                        moveElementEntirelyPastChangeRange(child, /*isArray*/ false, delta, oldText, newText, aggressiveChecks);
                        return;
                    }

                    // Check if the element intersects the change range.  If it does, then it is not
                    // reusable.  Also, we'll need to recurse to see what constituent portions we may
                    // be able to use.
                    auto fullEnd = child.end;
                    if (fullEnd >= changeStart) {
                        child.intersectsChange = true;
                        child._children = undefined;

                        // Adjust the pos or end (or both) of the intersecting element accordingly.
                        adjustIntersectingElement(child, changeStart, changeRangeOldEnd, changeRangeNewEnd, delta);
                        forEachChild(child, visitNode, visitArray);
                        if (hasJSDocNodes(child)) {
                            for (auto jsDocComment of child.jsDoc!) {
                                visitNode(<IncrementalNode>jsDocComment.as<Node>());
                            }
                        }
                        checkNodePositions(child, aggressiveChecks);
                        return;
                    }

                    // Otherwise, the node is entirely before the change range.  No need to do anything with it.
                    Debug::_assert(fullEnd < changeStart);
                }

                auto visitArray(IncrementalNodeArray array) {
                    Debug::_assert(array.pos <= array.end);
                    if (array.pos > changeRangeOldEnd) {
                        // Array is entirely after the change range.  We need to move it, and move any of
                        // its children.
                        moveElementEntirelyPastChangeRange(array, /*isArray*/ true, delta, oldText, newText, aggressiveChecks);
                        return;
                    }

                    // Check if the element intersects the change range.  If it does, then it is not
                    // reusable.  Also, we'll need to recurse to see what constituent portions we may
                    // be able to use.
                    auto fullEnd = array.end;
                    if (fullEnd >= changeStart) {
                        array.intersectsChange = true;
                        array._children = undefined;

                        // Adjust the pos or end (or both) of the intersecting array accordingly.
                        adjustIntersectingElement(array, changeStart, changeRangeOldEnd, changeRangeNewEnd, delta);
                        for (auto node of array) {
                            visitNode(node);
                        }
                        return;
                    }

                    // Otherwise, the array is entirely before the change range.  No need to do anything with it.
                    Debug::_assert(fullEnd < changeStart);
                }
            }

            auto extendToAffectedRange(SourceFile sourceFile, TextChangeRange changeRange) -> TextChangeRange {
                // Consider the following code:
                //      void foo() { /; }
                //
                // If the text changes with an insertion of / just before the semicolon then we end up with:
                //      void foo() { //; }
                //
                // If we were to just use the changeRange a is, then we would not rescan the { token
                // (as it does not intersect the actual original change range).  Because an edit may
                // change the token touching it, we actually need to look back *at least* one token so
                // that the prior token sees that change.
                auto maxLookahead = 1;

                auto start = changeRange.span.start;

                // the first iteration aligns us with the change start. subsequent iteration move us to
                // the left by maxLookahead tokens.  We only need to do this.as<long>().as<we>()'re not at the
                // start of the tree.
                for (auto i = 0; start > 0 && i <= maxLookahead; i++) {
                    auto nearestNode = findNearestNodeStartingBeforeOrAtPosition(sourceFile, start);
                    Debug::_assert(nearestNode.pos <= start);
                    auto position = nearestNode.pos;

                    start = Math.max(0, position - 1);
                }

                auto finalSpan = createTextSpanFromBounds(start, textSpanEnd(changeRange.span));
                auto finalLength = changeRange.newLength + (changeRange.span.start - start);

                return createTextChangeRange(finalSpan, finalLength);
            }

            auto findNearestNodeStartingBeforeOrAtPosition(SourceFile sourceFile, number position) -> Node {
                auto Node bestResult = sourceFile;
                auto Node lastNodeEntirelyBeforePosition;

                forEachChild(sourceFile, visit);

                if (lastNodeEntirelyBeforePosition) {
                    auto lastChildOfLastEntireNodeBeforePosition = getLastDescendant(lastNodeEntirelyBeforePosition);
                    if (lastChildOfLastEntireNodeBeforePosition.pos > bestResult.pos) {
                        bestResult = lastChildOfLastEntireNodeBeforePosition;
                    }
                }

                return bestResult;

                auto getLastDescendant(Node node) -> Node {
                    while (true) {
                        auto lastChild = getLastChild(node);
                        if (lastChild) {
                            node = lastChild;
                        }
                        else {
                            return node;
                        }
                    }
                }

                auto visit(Node child) {
                    if (nodeIsMissing(child)) {
                        // Missing nodes are effectively invisible to us.  We never even consider them
                        // When trying to find the nearest node before us.
                        return;
                    }

                    // If the child intersects this position, then this node is currently the nearest
                    // node that starts before the position.
                    if (child.pos <= position) {
                        if (child.pos >= bestResult.pos) {
                            // This node starts before the position, and is closer to the position than
                            // the previous best node we found.  It is now the new best node.
                            bestResult = child;
                        }

                        // Now, the node may overlap the position, or it may end entirely before the
                        // position.  If it overlaps with the position, then either it, or one of its
                        // children must be the nearest node before the position.  So we can just
                        // recurse into this child to see if we can find something better.
                        if (position < child.end) {
                            // The nearest node is either this child, or one of the children inside
                            // of it.  We've already marked this child.as<the>() best so far.  Recurse
                            // in case one of the children is better.
                            forEachChild(child, visit);

                            // Once we look at the children of this node, then there's no need to
                            // continue any further.
                            return true;
                        }
                        else {
                            Debug::_assert(child.end <= position);
                            // The child ends entirely before this position.  Say you have the following
                            // (where $ is the position)
                            //
                            //      <complex expr 1> ? <complex expr 2> $ : <...> <...>
                            //
                            // We would want to find the nearest preceding node in "complex expr 2".
                            // To support that, we keep track of this node, and once we're done searching
                            // for a best node, we recurse down this node to see if we can find a good
                            // result in it.
                            //
                            // This approach allows us to quickly skip over nodes that are entirely
                            // before the position, while still allowing us to find any nodes in the
                            // last one that might be what we want.
                            lastNodeEntirelyBeforePosition = child;
                        }
                    }
                    else {
                        Debug::_assert(child.pos > position);
                        // We're now at a node that is entirely past the position we're searching for.
                        // This node (and all following nodes) could never contribute to the result,
                        // so just skip them by returning 'true' here.
                        return true;
                    }
                }
            }

            static auto checkChangeRange(SourceFile sourceFile, string newText, TextChangeRange textChangeRange, boolean aggressiveChecks) {
                auto oldText = sourceFile.text;
                if (textChangeRange) {
                    Debug::_assert((oldText.size() - textChangeRange.span.size() + textChangeRange.newLength) == newText.size());

                    if (aggressiveChecks || Debug::shouldAssert(AssertionLevel::VeryAggressive)) {
                        auto oldTextPrefix = oldText.substr(0, textChangeRange.span.start);
                        auto newTextPrefix = newText.substr(0, textChangeRange.span.start);
                        Debug::_assert(oldTextPrefix == newTextPrefix);

                        auto oldTextSuffix = oldText.substring(textSpanEnd(textChangeRange.span), oldText.size());
                        auto newTextSuffix = newText.substring(textSpanEnd(textChangeRangeNewSpan(textChangeRange)), newText.size());
                        Debug::_assert(oldTextSuffix == newTextSuffix);
                    }
                }
            }

            auto createSyntaxCursor(SourceFile sourceFile) -> SyntaxCursor {
                auto NodeArray<Node> currentArray = sourceFile.statements;
                auto currentArrayIndex = 0;

                Debug::_assert(currentArrayIndex < currentArray.size());
                auto current = currentArray[currentArrayIndex];
                auto lastQueriedPosition = InvalidPosition.Value;

                return {
                    currentNode(number position) {
                        // Only compute the current node if the position is different than the last time
                        // we were asked.  The parser commonly asks for the node at the same position
                        // twice.  Once to know if can read an appropriate list element at a certain point,
                        // and then to actually read and consume the node.
                        if (position != lastQueriedPosition) {
                            // Much of the time the parser will need the very next node in the array that
                            // we just returned a node from.So just simply check for that case and move
                            // forward in the array instead of searching for the node again.
                            if (current && current.end == position && currentArrayIndex < (currentArray.size() - 1)) {
                                currentArrayIndex++;
                                current = currentArray[currentArrayIndex];
                            }

                            // If we don't have a node, or the node we have isn't in the right position,
                            // then try to find a viable node at the position requested.
                            if (!current || current.pos != position) {
                                findHighestListElementThatStartsAtPosition(position);
                            }
                        }

                        // Cache this query so that we don't do any extra work if the parser calls back
                        // into us.  this Note is very common.as<the>() parser will make pairs of calls like
                        // 'isListElement -> parseListElement'.  If we were unable to find a node when
                        // called with 'isListElement', we don't want to redo the work when parseListElement
                        // is called immediately after.
                        lastQueriedPosition = position;

                        // Either we don'd have a node, or we have a node at the position being asked for.
                        Debug::_assert(!current || current.pos == position);
                        return current.as<IncrementalNode>();
                    }
                };

                // Finds the highest element in the tree we can find that starts at the provided position.
                // The element must be a direct child of some node list in the tree.  This way after we
                // return it, we can easily return its next sibling in the list.
                auto findHighestListElementThatStartsAtPosition(number position) {
                    // Clear out any cached state about the last node we found.
                    currentArray = undefined!;
                    currentArrayIndex = InvalidPosition.Value;
                    current = undefined!;

                    // Recurse into the source file to find the highest node at this position.
                    forEachChild(sourceFile, visitNode, visitArray);
                    return;

                    auto visitNode(Node node) {
                        if (position >= node->pos && position < node->end) {
                            // Position was within this node.  Keep searching deeper to find the node.
                            forEachChild(node, visitNode, visitArray);

                            // don't proceed any further in the search.
                            return true;
                        }

                        // position wasn't in this node, have to keep searching.
                        return false;
                    }

                    auto visitArray(NodeArray<Node> array) {
                        if (position >= array.pos && position < array.end) {
                            // position was in this array.  Search through this array to see if we find a
                            // viable element.
                            for (auto i = 0; i < array.size(); i++) {
                                auto child = array[i];
                                if (child) {
                                    if (child.pos == position) {
                                        // Found the right node.  We're done.
                                        currentArray = array;
                                        currentArrayIndex = i;
                                        current = child;
                                        return true;
                                    }
                                    else {
                                        if (child.pos < position && position < child.end) {
                                            // Position in somewhere within this child.  Search in it and
                                            // stop searching in this array.
                                            forEachChild(child, visitNode, visitArray);
                                            return true;
                                        }
                                    }
                                }
                            }
                        }

                        // position wasn't in this array, have to keep searching.
                        return false;
                    }
                }
            }

            enum class InvalidPosition : number {
                Value = -1
            };
        };


        auto createSourceFile(string fileName, string sourceText, ScriptTarget languageVersion, boolean setParentNodes = false, ScriptKind scriptKind = ScriptKind::Unknown) -> SourceFile {
            SourceFile result;
            if (languageVersion == ScriptTarget::JSON) {
                result = Parser::parseSourceFile(fileName, sourceText, languageVersion, undefined /*syntaxCursor*/, setParentNodes, ScriptKind::JSON);
            }
            else {
                result = Parser::parseSourceFile(fileName, sourceText, languageVersion, undefined /*syntaxCursor*/, setParentNodes, scriptKind);
            }

            return result;
        }

        auto parseIsolatedEntityName(string text, ScriptTarget languageVersion) -> EntityName {
            return Parser::parseIsolatedEntityName(text, languageVersion);
        }

        /**
         * Parse json text into SyntaxTree and return node and parse errors if any
         * @param fileName
         * @param sourceText
         */
        auto parseJsonText(string fileName, string sourceText) -> JsonSourceFile {
            return Parser::parseJsonText(fileName, sourceText);
        }

        // See also `isExternalOrCommonJsModule` in utilities.ts
        auto isExternalModule(SourceFile file) -> boolean {
            return !!file.externalModuleIndicator;
        }

        // Produces a new SourceFile for the 'newText' provided. The 'textChangeRange' parameter
        // indicates what changed between the 'text' that this SourceFile has and the 'newText'.
        // The SourceFile will be created with the compiler attempting to reuse.as<many>() nodes from
        // this file.as<possible>().
        //
        // this Note auto mutates nodes from this SourceFile. That means any existing nodes
        // from this SourceFile that are being held onto may change.as<a>() result (including
        // becoming detached from any SourceFile).  It is recommended that this SourceFile not
        // be used once 'update' is called on it.
        auto updateSourceFile(SourceFile sourceFile, string newText, TextChangeRange textChangeRange, boolean aggressiveChecks = false) -> SourceFile {
            auto newSourceFile = IncrementalParser::updateSourceFile(sourceFile, newText, textChangeRange, aggressiveChecks);
            // Because new source file node is created, it may not have the flag PossiblyContainDynamicImport. This is the case if there is no new edit to add dynamic import.
            // We will manually port the flag to the new source file.
            newSourceFile.flags |= (sourceFile.flags & NodeFlags::PermanentlySetIncrementalFlags);
            return newSourceFile;
        }

        /* @internal */
        auto parseIsolatedJSDocComment(string content, number start, number length) {
            auto result = Parser::JSDocParser::parseIsolatedJSDocComment(content, start, length);
            if (result && result.jsDoc) {
                // because the jsDocComment was parsed out of the source file, it might
                // not be covered by the fixupParentReferences.
                Parser::fixupParentReferences(result.jsDoc);
            }

            return result;
        }

        /* @internal */
        // Exposed only for testing.
        auto parseJSDocTypeExpressionForTests(string content, number start, number length) {
            return Parser::JSDocParser::parseJSDocTypeExpressionForTests(content, start, length);
        }

        /*@internal*/
        auto processCommentPragmas(SourceFile context, string sourceText) -> void {
            auto std::vector<PragmaPseudoMapEntry> pragmas;

            for (auto range of getLeadingCommentRanges(sourceText, 0) || emptyArray) {
                auto comment = sourceText.substring(range.pos, range.end);
                extractPragmas(pragmas, range, comment);
            }

            context.pragmas = new Map().as<PragmaMap>();
            for (auto pragma of pragmas) {
                if (context.pragmas.has(pragma.name)) {
                    auto currentValue = context.pragmas.at(pragma.name);
                    if (currentValue instanceof Array) {
                        currentValue.push(pragma.args);
                    }
                    else {
                        context.pragmas.set(pragma.name, [currentValue, pragma.args]);
                    }
                    continue;
                }
                context.pragmas.set(pragma.name, pragma.args);
            }
        }

        /*@internal*/
        auto processPragmasIntoFields(SourceFile context, PragmaDiagnosticReporter reportDiagnostic) -> void {
            context.checkJsDirective = undefined;
            context.referencedFiles = [];
            context.typeReferenceDirectives = [];
            context.libReferenceDirectives = [];
            context.amdDependencies = [];
            context.hasNoDefaultLib = false;
            context.pragmas.forEach((entryOrList, key) => { // GH TODO#18217
                // The TODO below should be strongly type-guarded and not need casts/explicit annotations, since entryOrList is related to
                // key and key is constrained to a union; but it's not (see GH#21483 for at least partial fix) :(
                switch (key) {
                    case "reference": {
                        auto referencedFiles = context.referencedFiles;
                        auto typeReferenceDirectives = context.typeReferenceDirectives;
                        auto libReferenceDirectives = context.libReferenceDirectives;
                        forEach(toArray(entryOrList).as<PragmaPseudoMap>()["reference"][], arg => {
                            auto { types, lib, path } = arg.arguments;
                            if (arg.arguments["no-default-lib"]) {
                                context.hasNoDefaultLib = true;
                            }
                            else if (types) {
                                typeReferenceDirectives.push({ types.pos pos, types.end end, types.value fileName });
                            }
                            else if (lib) {
                                libReferenceDirectives.push({ lib.pos pos, lib.end end, lib.value fileName });
                            }
                            else if (path) {
                                referencedFiles.push({ path.pos pos, path.end end, path.value fileName });
                            }
                            else {
                                reportDiagnostic(arg.range.pos, arg.range.end - arg.range.pos, Diagnostics::Invalid_reference_directive_syntax);
                            }
                        });
                        break;
                    }
                    case "amd-dependency": {
                        context.amdDependencies = map(
                            toArray(entryOrList).as<PragmaPseudoMap>()["amd-dependency"][],
                            x => ({ x.arguments.name name, x.arguments.path path }));
                        break;
                    }
                    case "amd-module": {
                        if (entryOrList instanceof Array) {
                            for (auto entry of entryOrList) {
                                if (context.moduleName) {
                                    // It TODO's probably fine to issue this diagnostic on all instances of the pragma
                                    reportDiagnostic(entry.range.pos, entry.range.end - entry.range.pos, Diagnostics::An_AMD_module_cannot_have_multiple_name_assignments);
                                }
                                context.moduleName = (entry.as<PragmaPseudoMap>()["amd-module"]).arguments.name;
                            }
                        }
                        else {
                            context.moduleName = (entryOrList.as<PragmaPseudoMap>()["amd-module"]).arguments.name;
                        }
                        break;
                    }
                    case "ts-nocheck":
                    case "ts-check": {
                        // _last_ of either nocheck or check in a file is the "winner"
                        forEach(toArray(entryOrList), entry => {
                            if (!context.checkJsDirective || entry.range.pos > context.checkJsDirective.pos) {
                                context.checkJsDirective = {
                                    key enabled == "ts-check",
                                    entry.range.end end,
                                    entry.range.pos pos
                                };
                            }
                        });
                        break;
                    }
                    case "jsx":
                    case "jsxfrag":
                    case "jsximportsource":
                    case "jsxruntime":
                        return; // Accessed directly
                    Debug::fail default("Unhandled pragma kind"); // Can this be made into an assertNever in the future?
                }
            });
        }

        auto namedArgRegExCache = new Map<string, RegExp>();
        auto getNamedArgRegEx(string name) -> RegExp {
            if (namedArgRegExCache.has(name)) {
                return namedArgRegExCache.at(name)!;
            }
            auto result = new RegExp(`(\\s${name}\\s*=\\s*)('|")(.+?)\\2`, "im");
            namedArgRegExCache.set(name, result);
            return result;
        }

        auto tripleSlashXMLCommentStartRegEx = /^\/\/\/\s*<(\S+)\s.*?\/>/im;
        auto singleLinePragmaRegEx = /^\/\/\/?\s*@(\S+)\s*(.*)\s*$/im;
        auto extractPragmas(std::vector<PragmaPseudoMapEntry> pragmas, CommentRange range, string text) {
            auto tripleSlash = range.kind == SyntaxKind::SingleLineCommentTrivia && tripleSlashXMLCommentStartRegEx.exec(text);
            if (tripleSlash) {
                auto name = tripleSlash[1].toLowerCase().as<keyof>() PragmaPseudoMap; // Technically unsafe cast, but we do it so the below check to make it safe typechecks
                auto pragma = commentPragmas[name].as<PragmaDefinition>();
                if (!pragma || !(pragma.kind! & PragmaKindFlags.TripleSlashXML)) {
                    return;
                }
                if (pragma.args) {
                    auto argument: {[string index]: string | {string value, number pos, number end}} = {};
                    for (auto arg of pragma.args) {
                        auto matcher = getNamedArgRegEx(arg.name);
                        auto matchResult = matcher.exec(text);
                        if (!matchResult && !arg.optional) {
                            return; // Missing required argument, don't parse
                        }
                        else if (matchResult) {
                            if (arg.captureSpan) {
                                auto startPos = range.pos + matchResult.index + matchResult[1].size() + matchResult[2].size();
                                argument[arg.name] = {
                                    matchResult[3] value,
                                    startPos pos,
                                    startPos end + matchResult[3].size()
                                };
                            }
                            else {
                                argument[arg.name] = matchResult[3];
                            }
                        }
                    }
                    pragmas.push({ name, args: { argument arguments, range } }.as<PragmaPseudoMapEntry>());
                }
                else {
                    pragmas.push({ name, args: { arguments: {}, range } }.as<PragmaPseudoMapEntry>());
                }
                return;
            }

            auto singleLine = range.kind == SyntaxKind::SingleLineCommentTrivia && singleLinePragmaRegEx.exec(text);
            if (singleLine) {
                return addPragmaForMatch(pragmas, range, PragmaKindFlags.SingleLine, singleLine);
            }

            if (range.kind == SyntaxKind::MultiLineCommentTrivia) {
                auto multiLinePragmaRegEx = /\s*@(\S+)\s*(.*)\s*$/gim; // Defined inline since it uses the "g" flag, which keeps a persistent index (for iterating)
                auto RegExpExecArray multiLineMatch | null;
                while (multiLineMatch = multiLinePragmaRegEx.exec(text)) {
                    addPragmaForMatch(pragmas, range, PragmaKindFlags.MultiLine, multiLineMatch);
                }
            }
        }

        auto addPragmaForMatch(std::vector<PragmaPseudoMapEntry> pragmas, CommentRange range, PragmaKindFlags kind, RegExpExecArray match) {
            if (!match) return;
            auto name = match[1].toLowerCase().as<keyof>() PragmaPseudoMap; // Technically unsafe cast, but we do it so they below check to make it safe typechecks
            auto pragma = commentPragmas[name].as<PragmaDefinition>();
            if (!pragma || !(pragma.kind! & kind)) {
                return;
            }
            auto args = match[2]; // Split on spaces and match up positionally with definition
            auto argument = getNamedPragmaArguments(pragma, args);
            if (argument == "fail") return; // Missing required argument, fail to parse it
            pragmas.push({ name, args: { argument arguments, range } }.as<PragmaPseudoMapEntry>());
            return;
        }

        auto getNamedPragmaArguments(PragmaDefinition pragma, string text) -> {[string index]: string} | "fail" {
            if (!text) return {};
            if (!pragma.args) return {};
            auto args = text.split(/\s+/);
            auto argMap: {[string index]: string} = {};
            for (auto i = 0; i < pragma.args.size(); i++) {
                auto argument = pragma.args[i];
                if (!args[i] && !argument.optional) {
                    return "fail";
                }
                if (argument.captureSpan) {
                    return Debug::fail("Capture spans not yet implemented for non-xml pragmas");
                }
                argMap[argument.name] = args[i];
            }
            return argMap;
        }

        /** @internal */
        auto tagNamesAreEquivalent(JsxTagNameExpression lhs, JsxTagNameExpression rhs) -> boolean {
            if (lhs.kind != rhs.kind) {
                return false;
            }

            if (lhs.kind == SyntaxKind::Identifier) {
                return lhs.escapedText == (rhs.as<Identifier>()).escapedText;
            }

            if (lhs.kind == SyntaxKind::ThisKeyword) {
                return true;
            }

            // If we are at this statement then we must have PropertyAccessExpression and because tag name in Jsx element can only
            // take forms of JsxTagNameExpression which includes an identifier, "this" expression, or another propertyAccessExpression
            // it is safe to case the expression property.as<such>(). See parseJsxElementName for how we parse tag name in Jsx element
            return (lhs.as<PropertyAccessExpression>()).name.escapedText == (rhs.as<PropertyAccessExpression>()).name.escapedText &&
                tagNamesAreEquivalent((<PropertyAccessExpression>lhs).expression.as<JsxTagNameExpression>(), (<PropertyAccessExpression>rhs).expression.as<JsxTagNameExpression>());
        }

    } // namespace
}
