#ifndef NEW_PARSER_FWRD_TYPES2_H
#define NEW_PARSER_FWRD_TYPES2_H

#include "config.h"
#include "enums.h"
#include "scanner_enums.h"
#include "undefined.h"

#include <functional>
#include <memory>
#include <type_traits>

#define REF_NAME(x) x##Ref
#define REF_TYPE(x) std::shared_ptr<x>

#define FORWARD_DECLARATION(x)                                                                                         \
    struct x;                                                                                                          \
    using REF_NAME(x) = REF_TYPE(x);

#define FORWARD_DECLARATION_T(x)                                                                                       \
    template <typename T> struct x;                                                                                    \
    template <typename T> using REF_NAME(x) = REF_TYPE(x<T>);

#define FORWARD_DECLARATION_VAR(x, v)                                                                                  \
    template <v TKind> struct x;                                                                                       \
    template <v TKind> using REF_NAME(x) = REF_TYPE(x<TKind>);

#define FORWARD_DECLARATION_VARS(x, v)                                                                                 \
    template <v... TKind> struct x;                                                                                    \
    template <v... TKind> using REF_NAME(x) = REF_TYPE(x<TKind...>);

#define NODE_REF(x)                                                                                                    \
    using x = Node;                                                                                                    \
    using REF_NAME(x) = REF_NAME(Node);

#define CLASS_REF(x, c)                                                                                                \
    using x = c;                                                                                                       \
    using REF_NAME(x) = REF_NAME(c);

#define TMPL_REF(x, n, t)                                                                                              \
    using x = n<t>;                                                                                                    \
    using REF_NAME(x) = REF_NAME(n)<t>;

#define TMPL_REF2(x, n, t1, t2)                                                                                        \
    using x = n<t1>;                                                                                                   \
    using REF_NAME(x) = REF_NAME(n)<t1, t2>;

#define PTR(x) ptr<x>
#define POINTER(x) using x = PTR(data::x);
#define POINTER_T(x) template <typename T> using x = PTR(data::x<T>);
#define POINTER_VAR(x, v) template <v TKind> using x = PTR(data::x<TKind>);
#define POINTER_TARGS(x, v) template <v... TKind> using x = PTR(data::x<TKind...>);

template <typename T> struct ptr
{
    typedef T data;

    ptr() : instance(nullptr){}

    ptr(undefined_t) : instance(nullptr){}

    ptr(const T &data) : instance(std::make_shared<T>(data)){}

    template <typename U> ptr(ptr<U> otherPtr) : instance(std::static_pointer_cast<T>(otherPtr.instance)){}

    template <typename U> ptr(REF_TYPE(U) & otherInstance) : instance(std::static_pointer_cast<T>(otherInstance)){}

    ~ptr() = default;

    inline auto operator->()
    {
        return instance.operator->();
    }

    auto operator=(undefined_t) -> ptr &
    {
        instance.reset();
        return *this;
    }

    inline auto operator!()
    {
        return !instance;
    }

    inline operator bool()
    {
        return instance ? instance->_kind != SyntaxKind::Unknown : false;
    }

    inline operator SyntaxKind()
    {
        return instance ? instance->_kind : SyntaxKind::Unknown;
    }

    inline operator const T &() const
    {
        return *instance;
    }

    inline auto operator==(const ptr<T> &otherPtr) -> boolean
    {
        return instance == otherPtr.instance;
    }

    inline auto operator!=(const ptr<T> &otherPtr) -> boolean
    {
        return instance != otherPtr.instance;
    }

    template <typename U> inline auto operator==(const ptr<U> &otherPtr) -> boolean
    {
        return instance == otherPtr.instance;
    }

    template <typename U> inline auto operator!=(const ptr<U> &otherPtr) -> boolean
    {
        return instance != otherPtr.instance;
    }

    inline friend auto operator==(const ptr<T> &inst, const ptr<T> &otherPtr) -> boolean
    {
        return inst.instance == otherPtr.instance;
    }

    inline friend auto operator!=(const ptr<T> &inst, const ptr<T> &otherPtr) -> boolean
    {
        return inst.instance != otherPtr.instance;
    }

    template <typename U> inline friend auto operator==(const ptr<T> &inst, const ptr<U> &otherPtr) -> boolean
    {
        return inst.instance == otherPtr.instance;
    }

    template <typename U> inline friend auto operator!=(const ptr<T> &inst, const ptr<U> &otherPtr) -> boolean
    {
        return inst.instance != otherPtr.instance;
    }

    inline auto operator==(SyntaxKind kind) -> boolean
    {
        return this->operator SyntaxKind() == kind;
    }

    inline auto operator!=(SyntaxKind kind) -> boolean
    {
        return this->operator SyntaxKind() != kind;
    }

    inline auto operator==(undefined_t) -> boolean
    {
        return !instance;
    }

    inline auto operator!=(undefined_t) -> boolean
    {
        return !!instance;
    }

    inline auto operator==(std::nullptr_t) -> boolean
    {
        return instance == nullptr;
    }

    inline auto operator!=(std::nullptr_t) -> boolean
    {
        return instance != nullptr;
    }

    inline auto operator||(bool rhs) -> bool
    {
        return this->operator bool() || rhs;
    }

    template <typename L> inline auto operator||(L rhs) -> ptr<T>
    {
        if (this->operator bool())
        {
            return *this;
        }

        return rhs().template as<ptr<T>>();
    }

    template <typename U> inline auto as() -> U
    {
        if (!*this)
        {
            return U();
        }

        return U(instance);
    }

    template <typename U> inline auto asMutable() -> U
    {
        if (!*this)
        {
            return U();
        }

        return U(instance);
    }

    template <typename U, typename D = typename U::data> inline auto is() -> boolean
    {
        // TODO: review and simplify using Kind
        return !!std::dynamic_pointer_cast<D>(instance);
    }

    REF_TYPE(T) instance;
};

namespace ts
{
namespace data
{
FORWARD_DECLARATION(TextRange)
FORWARD_DECLARATION(ReadonlyTextRange)
FORWARD_DECLARATION_T(ReadonlyArray)
FORWARD_DECLARATION_T(NodeArray)
FORWARD_DECLARATION(Symbol)
FORWARD_DECLARATION(Type)
FORWARD_DECLARATION(Node)
FORWARD_DECLARATION(ImportAttribute)
FORWARD_DECLARATION(ImportAttributes)
FORWARD_DECLARATION_VARS(Token, SyntaxKind)
FORWARD_DECLARATION(JSDocContainer)
FORWARD_DECLARATION(EndOfFileToken)
FORWARD_DECLARATION_VAR(PunctuationToken, SyntaxKind)
FORWARD_DECLARATION_VAR(KeywordToken, SyntaxKind)
FORWARD_DECLARATION_VAR(ModifierToken, SyntaxKind)
FORWARD_DECLARATION_VAR(LiteralToken, SyntaxKind)
FORWARD_DECLARATION(QualifiedName)
FORWARD_DECLARATION(Declaration)
FORWARD_DECLARATION(NamedDeclaration)
FORWARD_DECLARATION(DynamicNamedDeclaration)
FORWARD_DECLARATION(DynamicNamedBinaryExpression)
FORWARD_DECLARATION(LateBoundDeclaration)
FORWARD_DECLARATION(LateBoundBinaryExpressionDeclaration)
FORWARD_DECLARATION(LateBoundElementAccessExpression)
FORWARD_DECLARATION(DeclarationStatement)
FORWARD_DECLARATION(ComputedPropertyName)
FORWARD_DECLARATION(PrivateIdentifier)
FORWARD_DECLARATION(LateBoundName)
FORWARD_DECLARATION(Decorator)
FORWARD_DECLARATION(TypeParameterDeclaration)
FORWARD_DECLARATION(SignatureDeclarationBase)
FORWARD_DECLARATION(CallSignatureDeclaration)
FORWARD_DECLARATION(ConstructSignatureDeclaration)
FORWARD_DECLARATION(VariableDeclaration)
FORWARD_DECLARATION(InitializedVariableDeclaration)
FORWARD_DECLARATION(VariableDeclarationList)
FORWARD_DECLARATION(ParameterDeclaration)
FORWARD_DECLARATION(BindingElement)
FORWARD_DECLARATION(PropertySignature)
FORWARD_DECLARATION(PropertyDeclaration)
FORWARD_DECLARATION(PrivateIdentifierPropertyDeclaration)
FORWARD_DECLARATION(InitializedPropertyDeclaration)
FORWARD_DECLARATION(ObjectLiteralElement)
FORWARD_DECLARATION(PropertyAssignment)
FORWARD_DECLARATION(ShorthandPropertyAssignment)
FORWARD_DECLARATION(SpreadAssignment)
FORWARD_DECLARATION(PropertyLikeDeclaration)
FORWARD_DECLARATION(ObjectBindingPattern)
FORWARD_DECLARATION(ArrayBindingPattern)
FORWARD_DECLARATION(FunctionLikeDeclarationBase)
FORWARD_DECLARATION(FunctionDeclaration)
FORWARD_DECLARATION(MethodSignature)
FORWARD_DECLARATION(MethodDeclaration)
FORWARD_DECLARATION(ConstructorDeclaration)
FORWARD_DECLARATION(SemicolonClassElement)
FORWARD_DECLARATION(AccessorDeclaration)
FORWARD_DECLARATION(GetAccessorDeclaration)
FORWARD_DECLARATION(SetAccessorDeclaration)
FORWARD_DECLARATION(IndexSignatureDeclaration)
FORWARD_DECLARATION(TypeNode)
FORWARD_DECLARATION_VAR(KeywordTypeNode, SyntaxKind)
FORWARD_DECLARATION(ImportTypeNode)
FORWARD_DECLARATION(LiteralImportTypeNode)
FORWARD_DECLARATION(ThisTypeNode)
FORWARD_DECLARATION(FunctionOrConstructorTypeNodeBase)
FORWARD_DECLARATION(FunctionTypeNode)
FORWARD_DECLARATION(ConstructorTypeNode)
FORWARD_DECLARATION(NodeWithTypeArguments)
FORWARD_DECLARATION(TypeReferenceNode)
FORWARD_DECLARATION(TypePredicateNode)
FORWARD_DECLARATION(TypeQueryNode)
FORWARD_DECLARATION(TypeLiteralNode)
FORWARD_DECLARATION(ArrayTypeNode)
FORWARD_DECLARATION(TupleTypeNode)
FORWARD_DECLARATION(NamedTupleMember)
FORWARD_DECLARATION(OptionalTypeNode)
FORWARD_DECLARATION(RestTypeNode)
FORWARD_DECLARATION(UnionTypeNode)
FORWARD_DECLARATION(IntersectionTypeNode)
FORWARD_DECLARATION(ConditionalTypeNode)
FORWARD_DECLARATION(InferTypeNode)
FORWARD_DECLARATION(ParenthesizedTypeNode)
FORWARD_DECLARATION(TypeOperatorNode)
FORWARD_DECLARATION(UniqueTypeOperatorNode)
FORWARD_DECLARATION(IndexedAccessTypeNode)
FORWARD_DECLARATION(MappedTypeNode)
FORWARD_DECLARATION(LiteralTypeNode)
FORWARD_DECLARATION(StringLiteral)
FORWARD_DECLARATION(TemplateLiteralTypeNode)
FORWARD_DECLARATION(TemplateLiteralTypeSpan)
FORWARD_DECLARATION(Expression)
FORWARD_DECLARATION(OmittedExpression)
FORWARD_DECLARATION(PartiallyEmittedExpression)
FORWARD_DECLARATION(UnaryExpression)
FORWARD_DECLARATION(UpdateExpression)
FORWARD_DECLARATION(PrefixUnaryExpression)
FORWARD_DECLARATION(PostfixUnaryExpression)
FORWARD_DECLARATION(LeftHandSideExpression)
FORWARD_DECLARATION(MemberExpression)
FORWARD_DECLARATION(PrimaryExpression)
FORWARD_DECLARATION(Identifier)
FORWARD_DECLARATION(TransientIdentifier)
FORWARD_DECLARATION(GeneratedIdentifier)
FORWARD_DECLARATION(NullLiteral)
FORWARD_DECLARATION(TrueLiteral)
FORWARD_DECLARATION(FalseLiteral)
FORWARD_DECLARATION(ThisExpression)
FORWARD_DECLARATION(SuperExpression)
FORWARD_DECLARATION(ImportExpression)
FORWARD_DECLARATION(DeleteExpression)
FORWARD_DECLARATION(TypeOfExpression)
FORWARD_DECLARATION(VoidExpression)
FORWARD_DECLARATION(AwaitExpression)
FORWARD_DECLARATION(YieldExpression)
FORWARD_DECLARATION(SyntheticExpression)
FORWARD_DECLARATION(BinaryExpression)
FORWARD_DECLARATION_T(AssignmentExpression)
FORWARD_DECLARATION(ObjectDestructuringAssignment)
FORWARD_DECLARATION(ArrayDestructuringAssignment)
FORWARD_DECLARATION(ConditionalExpression)
FORWARD_DECLARATION(FunctionExpression)
FORWARD_DECLARATION(ArrowFunction)
FORWARD_DECLARATION(LiteralLikeNode)
FORWARD_DECLARATION(TemplateLiteralLikeNode)
FORWARD_DECLARATION(LiteralExpression)
FORWARD_DECLARATION(RegularExpressionLiteral)
FORWARD_DECLARATION(NoSubstitutionTemplateLiteral)
FORWARD_DECLARATION(NumericLiteral)
FORWARD_DECLARATION(BigIntLiteral)
FORWARD_DECLARATION(PseudoBigInt)
FORWARD_DECLARATION(TemplateHead)
FORWARD_DECLARATION(TemplateMiddle)
FORWARD_DECLARATION(TemplateTail)
FORWARD_DECLARATION(TemplateExpression)
FORWARD_DECLARATION(TemplateSpan)
FORWARD_DECLARATION(ParenthesizedExpression)
FORWARD_DECLARATION(ArrayLiteralExpression)
FORWARD_DECLARATION(SpreadElement)
FORWARD_DECLARATION_T(ObjectLiteralExpressionBase)
FORWARD_DECLARATION(ObjectLiteralExpression)
FORWARD_DECLARATION(PropertyAccessExpression)
FORWARD_DECLARATION(PrivateIdentifierPropertyAccessExpression)
FORWARD_DECLARATION(PropertyAccessChain)
FORWARD_DECLARATION(PropertyAccessChainRoot)
FORWARD_DECLARATION(SuperPropertyAccessExpression)
FORWARD_DECLARATION(PropertyAccessEntityNameExpression)
FORWARD_DECLARATION(ElementAccessExpression)
FORWARD_DECLARATION(ElementAccessChain)
FORWARD_DECLARATION(ElementAccessChainRoot)
FORWARD_DECLARATION(SuperElementAccessExpression)
FORWARD_DECLARATION(CallExpression)
FORWARD_DECLARATION(CallChain)
FORWARD_DECLARATION(CallChainRoot)
FORWARD_DECLARATION(BindableObjectDefinePropertyCall)
FORWARD_DECLARATION(LiteralLikeElementAccessExpression)
FORWARD_DECLARATION(BindableStaticElementAccessExpression)
FORWARD_DECLARATION(BindableElementAccessExpression)
FORWARD_DECLARATION(BindableStaticPropertyAssignmentExpression)
FORWARD_DECLARATION(BindablePropertyAssignmentExpression)
FORWARD_DECLARATION(SuperCall)
FORWARD_DECLARATION(ImportCall)
FORWARD_DECLARATION(ExpressionWithTypeArguments)
FORWARD_DECLARATION(NewExpression)
FORWARD_DECLARATION(TaggedTemplateExpression)
FORWARD_DECLARATION(AsExpression)
FORWARD_DECLARATION(TypeAssertion)
FORWARD_DECLARATION(SatisfiesExpression)
FORWARD_DECLARATION(NonNullExpression)
FORWARD_DECLARATION(NonNullChain)
FORWARD_DECLARATION(MetaProperty)
FORWARD_DECLARATION(ImportMetaProperty)
FORWARD_DECLARATION(JsxElement)
FORWARD_DECLARATION(JsxTagNamePropertyAccess)
FORWARD_DECLARATION(JsxAttributes)
FORWARD_DECLARATION(JsxNamespacedName)
FORWARD_DECLARATION(JsxOpeningElement)
FORWARD_DECLARATION(JsxSelfClosingElement)
FORWARD_DECLARATION(JsxFragment)
FORWARD_DECLARATION(JsxOpeningFragment)
FORWARD_DECLARATION(JsxClosingFragment)
FORWARD_DECLARATION(JsxAttribute)
FORWARD_DECLARATION(JsxSpreadAttribute)
FORWARD_DECLARATION(JsxClosingElement)
FORWARD_DECLARATION(JsxExpression)
FORWARD_DECLARATION(JsxText)
FORWARD_DECLARATION(Statement)
FORWARD_DECLARATION(BreakOrContinueStatement)
FORWARD_DECLARATION(NotEmittedStatement)
FORWARD_DECLARATION(EndOfDeclarationMarker)
FORWARD_DECLARATION(CommaListExpression)
FORWARD_DECLARATION(MergeDeclarationMarker)
FORWARD_DECLARATION(SyntheticReferenceExpression)
FORWARD_DECLARATION(EmptyStatement)
FORWARD_DECLARATION(DebuggerStatement)
FORWARD_DECLARATION(MissingDeclaration)
FORWARD_DECLARATION(Block)
FORWARD_DECLARATION(VariableStatement)
FORWARD_DECLARATION(ExpressionStatement)
FORWARD_DECLARATION(PrologueDirective)
FORWARD_DECLARATION(IfStatement)
FORWARD_DECLARATION(IterationStatement)
FORWARD_DECLARATION(DoStatement)
FORWARD_DECLARATION(WhileStatement)
FORWARD_DECLARATION(ForStatement)
FORWARD_DECLARATION(ForInStatement)
FORWARD_DECLARATION(ForOfStatement)
FORWARD_DECLARATION(BreakStatement)
FORWARD_DECLARATION(ContinueStatement)
FORWARD_DECLARATION(ReturnStatement)
FORWARD_DECLARATION(WithStatement)
FORWARD_DECLARATION(SwitchStatement)
FORWARD_DECLARATION(CaseBlock)
FORWARD_DECLARATION(CaseOrDefaultClause)
FORWARD_DECLARATION(CaseClause)
FORWARD_DECLARATION(DefaultClause)
FORWARD_DECLARATION(LabeledStatement)
FORWARD_DECLARATION(ThrowStatement)
FORWARD_DECLARATION(TryStatement)
FORWARD_DECLARATION(CatchClause)
FORWARD_DECLARATION(ClassLikeDeclaration)
FORWARD_DECLARATION(ClassDeclaration)
FORWARD_DECLARATION(ClassExpression)
FORWARD_DECLARATION(ClassElement)
FORWARD_DECLARATION(ClassStaticBlockDeclaration)
FORWARD_DECLARATION(TypeElement)
FORWARD_DECLARATION(InterfaceDeclaration)
FORWARD_DECLARATION(HeritageClause)
FORWARD_DECLARATION(TypeAliasDeclaration)
FORWARD_DECLARATION(EnumMember)
FORWARD_DECLARATION(EnumDeclaration)
FORWARD_DECLARATION(AmbientModuleDeclaration)
FORWARD_DECLARATION(ModuleBody)
FORWARD_DECLARATION(ModuleDeclaration)
FORWARD_DECLARATION(NamespaceDeclaration)
FORWARD_DECLARATION(JSDocNamespaceDeclaration)
FORWARD_DECLARATION(ModuleBlock)
FORWARD_DECLARATION(ImportEqualsDeclaration)
FORWARD_DECLARATION(ExternalModuleReference)
FORWARD_DECLARATION(ImportDeclaration)
FORWARD_DECLARATION(ImportClause)
FORWARD_DECLARATION(NamespaceImport)
FORWARD_DECLARATION(NamespaceExport)
FORWARD_DECLARATION(NamespaceExportDeclaration)
FORWARD_DECLARATION(ExportDeclaration)
FORWARD_DECLARATION(NamedImportsOrExports)
FORWARD_DECLARATION(NamedImports)
FORWARD_DECLARATION(NamedExports)
FORWARD_DECLARATION(ImportOrExportSpecifier)
FORWARD_DECLARATION(ImportSpecifier)
FORWARD_DECLARATION(ExportSpecifier)
FORWARD_DECLARATION(ExportAssignment)
FORWARD_DECLARATION(FileReference)
FORWARD_DECLARATION(CheckJsDirective)
FORWARD_DECLARATION(CommentRange)
FORWARD_DECLARATION(SynthesizedComment)
FORWARD_DECLARATION(JSDocTypeExpression)
FORWARD_DECLARATION(JSDocNameReference)
FORWARD_DECLARATION(JSDocMemberName)
FORWARD_DECLARATION(JSDocType)
FORWARD_DECLARATION(JSDocAllType)
FORWARD_DECLARATION(JSDocUnknownType)
FORWARD_DECLARATION(JSDocNonNullableType)
FORWARD_DECLARATION(JSDocNullableType)
FORWARD_DECLARATION(JSDocOptionalType)
FORWARD_DECLARATION(JSDocFunctionType)
FORWARD_DECLARATION(JSDocVariadicType)
FORWARD_DECLARATION(JSDocNamepathType)
FORWARD_DECLARATION(JSDoc)
FORWARD_DECLARATION(JSDocTag)
FORWARD_DECLARATION(JSDocUnknownTag)
FORWARD_DECLARATION(JSDocAugmentsTag)
FORWARD_DECLARATION(JSDocImplementsTag)
FORWARD_DECLARATION(JSDocAuthorTag)
FORWARD_DECLARATION(JSDocDeprecatedTag)
FORWARD_DECLARATION(JSDocClassTag)
FORWARD_DECLARATION(JSDocPublicTag)
FORWARD_DECLARATION(JSDocPrivateTag)
FORWARD_DECLARATION(JSDocProtectedTag)
FORWARD_DECLARATION(JSDocReadonlyTag)
FORWARD_DECLARATION(JSDocEnumTag)
FORWARD_DECLARATION(JSDocThisTag)
FORWARD_DECLARATION(JSDocTemplateTag)
FORWARD_DECLARATION(JSDocSeeTag)
FORWARD_DECLARATION(JSDocReturnTag)
FORWARD_DECLARATION(JSDocTypeTag)
FORWARD_DECLARATION(JSDocTypedefTag)
FORWARD_DECLARATION(JSDocCallbackTag)
FORWARD_DECLARATION(JSDocSignature)
FORWARD_DECLARATION(JSDocPropertyLikeTag)
FORWARD_DECLARATION(JSDocPropertyTag)
FORWARD_DECLARATION(JSDocParameterTag)
FORWARD_DECLARATION(JSDocTypeLiteral)
FORWARD_DECLARATION(AmdDependency)
FORWARD_DECLARATION(CommentDirective)
FORWARD_DECLARATION(PragmaPseudoMapEntry)
FORWARD_DECLARATION(SourceFileLike)
FORWARD_DECLARATION(RedirectInfo)
FORWARD_DECLARATION(DiagnosticMessage)
FORWARD_DECLARATION(DiagnosticMessageChain)
FORWARD_DECLARATION(DiagnosticRelatedInformation)
FORWARD_DECLARATION(Diagnostic)
FORWARD_DECLARATION(DiagnosticWithLocation)
FORWARD_DECLARATION(DiagnosticWithDetachedLocation)
FORWARD_DECLARATION(ResolvedModule)
FORWARD_DECLARATION(PackageId)
FORWARD_DECLARATION(ResolvedModuleFull)
FORWARD_DECLARATION(ResolvedTypeReferenceDirective)
FORWARD_DECLARATION(PatternAmbientModule)
FORWARD_DECLARATION(SourceFile)
FORWARD_DECLARATION(UnparsedSource)
FORWARD_DECLARATION(UnparsedSourceText)
FORWARD_DECLARATION(UnparsedNode)
FORWARD_DECLARATION(UnparsedSection)
FORWARD_DECLARATION(UnparsedPrologue)
FORWARD_DECLARATION(UnparsedPrepend)
FORWARD_DECLARATION(UnparsedTextLike)
FORWARD_DECLARATION(UnparsedSyntheticReference)
FORWARD_DECLARATION(EmitHelper)
FORWARD_DECLARATION(InputFiles)
FORWARD_DECLARATION(SyntaxList)
FORWARD_DECLARATION(Bundle)
FORWARD_DECLARATION(PropertyDescriptorAttributes)
FORWARD_DECLARATION(CallBinding)
FORWARD_DECLARATION_T(Push)
FORWARD_DECLARATION_T(VisitResult)
FORWARD_DECLARATION(NodeWithDiagnostics)
FORWARD_DECLARATION(JsonObjectExpressionStatement)

NODE_REF(ModifierLike)
NODE_REF(Modifier)

NODE_REF(EntityName)
NODE_REF(PropertyName)
NODE_REF(MemberName)
NODE_REF(DeclarationName)

NODE_REF(EntityNameExpression)
NODE_REF(EntityNameOrEntityNameExpression)
NODE_REF(AccessExpression)

NODE_REF(BindingName)
NODE_REF(SignatureDeclaration)
NODE_REF(BindingPattern)
NODE_REF(ArrayBindingElement)

CLASS_REF(FunctionBody, Block)
NODE_REF(ConciseBody)

NODE_REF(ObjectTypeDeclaration)
NODE_REF(DeclarationWithTypeParameters)
NODE_REF(DeclarationWithTypeParameterChildren)

TMPL_REF(DotToken, PunctuationToken, SyntaxKind::DotToken)
TMPL_REF(DotDotDotToken, PunctuationToken, SyntaxKind::DotDotDotToken)
TMPL_REF(QuestionToken, PunctuationToken, SyntaxKind::QuestionToken)
TMPL_REF(ExclamationToken, PunctuationToken, SyntaxKind::ExclamationToken)
TMPL_REF(ColonToken, PunctuationToken, SyntaxKind::ColonToken)
TMPL_REF(EqualsToken, PunctuationToken, SyntaxKind::EqualsToken)
TMPL_REF(AsteriskToken, PunctuationToken, SyntaxKind::AsteriskToken)
TMPL_REF(EqualsGreaterThanToken, PunctuationToken, SyntaxKind::EqualsGreaterThanToken)
TMPL_REF(PlusToken, PunctuationToken, SyntaxKind::PlusToken)
TMPL_REF(MinusToken, PunctuationToken, SyntaxKind::MinusToken)
TMPL_REF(QuestionDotToken, PunctuationToken, SyntaxKind::QuestionDotToken)

TMPL_REF(AssertsKeyword, KeywordToken, SyntaxKind::AssertsKeyword)
TMPL_REF(AwaitKeyword, KeywordToken, SyntaxKind::AwaitKeyword)

TMPL_REF2(BinaryOperatorToken, Token, SyntaxKind::AsteriskAsteriskToken, SyntaxKind::CommaToken)

CLASS_REF(AwaitKeywordToken, AwaitKeyword)
CLASS_REF(AssertsToken, AssertsKeyword)

TMPL_REF(AbstractKeyword, ModifierToken, SyntaxKind::AbstractKeyword)
TMPL_REF(AsyncKeyword, ModifierToken, SyntaxKind::AsyncKeyword)
TMPL_REF(ConstKeyword, ModifierToken, SyntaxKind::ConstKeyword)
TMPL_REF(DeclareKeyword, ModifierToken, SyntaxKind::DeclareKeyword)
TMPL_REF(DefaultKeyword, ModifierToken, SyntaxKind::DefaultKeyword)
TMPL_REF(ExportKeyword, ModifierToken, SyntaxKind::ExportKeyword)
TMPL_REF(PrivateKeyword, ModifierToken, SyntaxKind::PrivateKeyword)
TMPL_REF(ProtectedKeyword, ModifierToken, SyntaxKind::ProtectedKeyword)
TMPL_REF(PublicKeyword, ModifierToken, SyntaxKind::PublicKeyword)
TMPL_REF(ReadonlyKeyword, ModifierToken, SyntaxKind::ReadonlyKeyword)
TMPL_REF(StaticKeyword, ModifierToken, SyntaxKind::StaticKeyword)

NODE_REF(BindableStaticNameExpression)
NODE_REF(BindableStaticAccessExpression)
NODE_REF(BindableAccessExpression)
NODE_REF(CallLikeExpression)

CLASS_REF(TemplateLiteral, TemplateLiteralLikeNode)

NODE_REF(JsxChild)
NODE_REF(JsxOpeningLikeElement)
NODE_REF(JsxAttributeLike)
NODE_REF(JsxTagNameExpression)

NODE_REF(ForInitializer)

NODE_REF(ModuleName)
NODE_REF(ModuleReference)
NODE_REF(NamespaceBody)
NODE_REF(JSDocNamespaceBody)

NODE_REF(NamedImportBindings)
NODE_REF(NamedExportBindings)

NODE_REF(TypeOnlyCompatibleAliasDeclaration)

NODE_REF(JSDocComment)
NODE_REF(JSDocTypeReferencingNode)
NODE_REF(HasJSDoc)
NODE_REF(DestructuringPattern)
NODE_REF(BindingElementGrandparent)

NODE_REF(ObjectLiteralElementLike)
NODE_REF(StringLiteralLike)
NODE_REF(PropertyNameLiteral)

CLASS_REF(JsonSourceFile, SourceFile)

NODE_REF(UnionOrIntersectionTypeNode)

NODE_REF(BooleanLiteral)
NODE_REF(JsonObjectExpression)

NODE_REF(FunctionOrConstructorTypeNode)
NODE_REF(AssignmentPattern)
} // namespace data

POINTER(TextRange)
POINTER(ReadonlyTextRange)
POINTER(Symbol)
POINTER(Type)
POINTER(Node)
POINTER(ImportAttribute)
POINTER(ImportAttributes)
POINTER(JSDocContainer)
POINTER(EndOfFileToken)
POINTER(QualifiedName)
POINTER(Declaration)
POINTER(NamedDeclaration)
POINTER(DynamicNamedDeclaration)
POINTER(DynamicNamedBinaryExpression)
POINTER(LateBoundDeclaration)
POINTER(LateBoundBinaryExpressionDeclaration)
POINTER(LateBoundElementAccessExpression)
POINTER(DeclarationStatement)
POINTER(ComputedPropertyName)
POINTER(PrivateIdentifier)
POINTER(LateBoundName)
POINTER(ModifierLike)
POINTER(Decorator)
POINTER(Modifier)
POINTER(TypeParameterDeclaration)
POINTER(SignatureDeclarationBase)
POINTER(CallSignatureDeclaration)
POINTER(ConstructSignatureDeclaration)
POINTER(VariableDeclaration)
POINTER(InitializedVariableDeclaration)
POINTER(VariableDeclarationList)
POINTER(ParameterDeclaration)
POINTER(BindingElement)
POINTER(PropertySignature)
POINTER(PropertyDeclaration)
POINTER(PrivateIdentifierPropertyDeclaration)
POINTER(InitializedPropertyDeclaration)
POINTER(ObjectLiteralElement)
POINTER(PropertyAssignment)
POINTER(ShorthandPropertyAssignment)
POINTER(SpreadAssignment)
POINTER(PropertyLikeDeclaration)
POINTER(ObjectBindingPattern)
POINTER(ArrayBindingPattern)
POINTER(FunctionLikeDeclarationBase)
POINTER(FunctionDeclaration)
POINTER(MethodSignature)
POINTER(MethodDeclaration)
POINTER(ConstructorDeclaration)
POINTER(SemicolonClassElement)
POINTER(GetAccessorDeclaration)
POINTER(SetAccessorDeclaration)
POINTER(IndexSignatureDeclaration)
POINTER(TypeNode)
POINTER(ImportTypeNode)
POINTER(LiteralImportTypeNode)
POINTER(ThisTypeNode)
POINTER(FunctionOrConstructorTypeNodeBase)
POINTER(FunctionTypeNode)
POINTER(ConstructorTypeNode)
POINTER(NodeWithTypeArguments)
POINTER(TypeReferenceNode)
POINTER(TypePredicateNode)
POINTER(TypeQueryNode)
POINTER(TypeLiteralNode)
POINTER(ArrayTypeNode)
POINTER(TupleTypeNode)
POINTER(NamedTupleMember)
POINTER(OptionalTypeNode)
POINTER(RestTypeNode)
POINTER(UnionTypeNode)
POINTER(IntersectionTypeNode)
POINTER(ConditionalTypeNode)
POINTER(InferTypeNode)
POINTER(ParenthesizedTypeNode)
POINTER(TypeOperatorNode)
POINTER(UniqueTypeOperatorNode)
POINTER(IndexedAccessTypeNode)
POINTER(MappedTypeNode)
POINTER(LiteralTypeNode)
POINTER(StringLiteral)
POINTER(TemplateLiteralTypeNode)
POINTER(TemplateLiteralTypeSpan)
POINTER(Expression)
POINTER(OmittedExpression)
POINTER(PartiallyEmittedExpression)
POINTER(UnaryExpression)
POINTER(UpdateExpression)
POINTER(PrefixUnaryExpression)
POINTER(PostfixUnaryExpression)
POINTER(LeftHandSideExpression)
POINTER(MemberExpression)
POINTER(PrimaryExpression)
POINTER(Identifier)
POINTER(TransientIdentifier)
POINTER(GeneratedIdentifier)
POINTER(NullLiteral)
POINTER(TrueLiteral)
POINTER(FalseLiteral)
POINTER(ThisExpression)
POINTER(SuperExpression)
POINTER(ImportExpression)
POINTER(DeleteExpression)
POINTER(TypeOfExpression)
POINTER(VoidExpression)
POINTER(AwaitExpression)
POINTER(YieldExpression)
POINTER(SyntheticExpression)
POINTER(BinaryExpression)
POINTER(ObjectDestructuringAssignment)
POINTER(ArrayDestructuringAssignment)
POINTER(ConditionalExpression)
POINTER(FunctionExpression)
POINTER(ArrowFunction)
POINTER(LiteralLikeNode)
POINTER(TemplateLiteralLikeNode)
POINTER(LiteralExpression)
POINTER(RegularExpressionLiteral)
POINTER(NoSubstitutionTemplateLiteral)
POINTER(NumericLiteral)
POINTER(BigIntLiteral)
POINTER(PseudoBigInt)
POINTER(TemplateHead)
POINTER(TemplateMiddle)
POINTER(TemplateTail)
POINTER(TemplateExpression)
POINTER(TemplateSpan)
POINTER(ParenthesizedExpression)
POINTER(ArrayLiteralExpression)
POINTER(SpreadElement)
POINTER(ObjectLiteralExpression)
POINTER(PropertyAccessExpression)
POINTER(PrivateIdentifierPropertyAccessExpression)
POINTER(PropertyAccessChain)
POINTER(PropertyAccessChainRoot)
POINTER(SuperPropertyAccessExpression)
POINTER(PropertyAccessEntityNameExpression)
POINTER(ElementAccessExpression)
POINTER(ElementAccessChain)
POINTER(ElementAccessChainRoot)
POINTER(SuperElementAccessExpression)
POINTER(CallExpression)
POINTER(CallChain)
POINTER(CallChainRoot)
POINTER(BindableObjectDefinePropertyCall)
POINTER(LiteralLikeElementAccessExpression)
POINTER(BindableStaticElementAccessExpression)
POINTER(BindableElementAccessExpression)
POINTER(BindableStaticPropertyAssignmentExpression)
POINTER(BindablePropertyAssignmentExpression)
POINTER(SuperCall)
POINTER(ImportCall)
POINTER(ExpressionWithTypeArguments)
POINTER(NewExpression)
POINTER(TaggedTemplateExpression)
POINTER(AsExpression)
POINTER(TypeAssertion)
POINTER(SatisfiesExpression)
POINTER(NonNullExpression)
POINTER(NonNullChain)
POINTER(MetaProperty)
POINTER(ImportMetaProperty)
POINTER(JsxElement)
POINTER(JsxTagNamePropertyAccess)
POINTER(JsxAttributes)
POINTER(JsxNamespacedName)
POINTER(JsxOpeningElement)
POINTER(JsxSelfClosingElement)
POINTER(JsxFragment)
POINTER(JsxOpeningFragment)
POINTER(JsxClosingFragment)
POINTER(JsxAttribute)
POINTER(JsxSpreadAttribute)
POINTER(JsxClosingElement)
POINTER(JsxExpression)
POINTER(JsxText)
POINTER(Statement)
POINTER(BreakOrContinueStatement)
POINTER(NotEmittedStatement)
POINTER(EndOfDeclarationMarker)
POINTER(CommaListExpression)
POINTER(MergeDeclarationMarker)
POINTER(SyntheticReferenceExpression)
POINTER(EmptyStatement)
POINTER(DebuggerStatement)
POINTER(MissingDeclaration)
POINTER(Block)
POINTER(VariableStatement)
POINTER(ExpressionStatement)
POINTER(PrologueDirective)
POINTER(IfStatement)
POINTER(IterationStatement)
POINTER(DoStatement)
POINTER(WhileStatement)
POINTER(ForStatement)
POINTER(ForInStatement)
POINTER(ForOfStatement)
POINTER(BreakStatement)
POINTER(ContinueStatement)
POINTER(ReturnStatement)
POINTER(WithStatement)
POINTER(SwitchStatement)
POINTER(CaseBlock)
POINTER(CaseClause)
POINTER(DefaultClause)
POINTER(LabeledStatement)
POINTER(ThrowStatement)
POINTER(TryStatement)
POINTER(CatchClause)
POINTER(ClassLikeDeclaration)
POINTER(ClassDeclaration)
POINTER(ClassExpression)
POINTER(ClassStaticBlockDeclaration)
POINTER(ClassElement)
POINTER(TypeElement)
POINTER(InterfaceDeclaration)
POINTER(HeritageClause)
POINTER(TypeAliasDeclaration)
POINTER(EnumMember)
POINTER(EnumDeclaration)
POINTER(AmbientModuleDeclaration)
POINTER(ModuleDeclaration)
POINTER(NamespaceDeclaration)
POINTER(JSDocNamespaceDeclaration)
POINTER(ModuleBlock)
POINTER(ImportEqualsDeclaration)
POINTER(ExternalModuleReference)
POINTER(ImportDeclaration)
POINTER(ImportClause)
POINTER(NamespaceImport)
POINTER(NamespaceExport)
POINTER(NamespaceExportDeclaration)
POINTER(ExportDeclaration)
POINTER(NamedImports)
POINTER(NamedExports)
POINTER(ImportSpecifier)
POINTER(ExportSpecifier)
POINTER(ExportAssignment)
POINTER(FileReference)
POINTER(CheckJsDirective)
POINTER(CommentRange)
POINTER(SynthesizedComment)
POINTER(JSDocTypeExpression)
POINTER(JSDocNameReference)
POINTER(JSDocMemberName)
POINTER(JSDocType)
POINTER(JSDocAllType)
POINTER(JSDocUnknownType)
POINTER(JSDocNonNullableType)
POINTER(JSDocNullableType)
POINTER(JSDocOptionalType)
POINTER(JSDocFunctionType)
POINTER(JSDocVariadicType)
POINTER(JSDocNamepathType)
POINTER(JSDoc)
POINTER(JSDocTag)
POINTER(JSDocUnknownTag)
POINTER(JSDocAugmentsTag)
POINTER(JSDocImplementsTag)
POINTER(JSDocAuthorTag)
POINTER(JSDocDeprecatedTag)
POINTER(JSDocClassTag)
POINTER(JSDocPublicTag)
POINTER(JSDocPrivateTag)
POINTER(JSDocProtectedTag)
POINTER(JSDocReadonlyTag)
POINTER(JSDocEnumTag)
POINTER(JSDocThisTag)
POINTER(JSDocTemplateTag)
POINTER(JSDocSeeTag)
POINTER(JSDocReturnTag)
POINTER(JSDocTypeTag)
POINTER(JSDocTypedefTag)
POINTER(JSDocCallbackTag)
POINTER(JSDocSignature)
POINTER(JSDocPropertyLikeTag)
POINTER(JSDocPropertyTag)
POINTER(JSDocParameterTag)
POINTER(JSDocTypeLiteral)
POINTER(AmdDependency)
POINTER(CommentDirective)
POINTER(PragmaPseudoMapEntry)
POINTER(SourceFileLike)
POINTER(RedirectInfo)
POINTER(DiagnosticMessage)
POINTER(DiagnosticMessageChain)
POINTER(DiagnosticRelatedInformation)
POINTER(Diagnostic)
POINTER(DiagnosticWithLocation)
POINTER(DiagnosticWithDetachedLocation)
POINTER(ResolvedModule)
POINTER(PackageId)
POINTER(ResolvedModuleFull)
POINTER(ResolvedTypeReferenceDirective)
POINTER(PatternAmbientModule)
POINTER(SourceFile)

POINTER(EntityName)
POINTER(PropertyName)
POINTER(MemberName)
POINTER(DeclarationName)

POINTER(EntityNameExpression)
POINTER(EntityNameOrEntityNameExpression)
POINTER(AccessExpression)

POINTER(BindingName)
POINTER(SignatureDeclaration)
POINTER(BindingPattern)
POINTER(ArrayBindingElement)

POINTER(FunctionBody)
POINTER(ConciseBody)

POINTER(ObjectTypeDeclaration)
POINTER(DeclarationWithTypeParameters)
POINTER(DeclarationWithTypeParameterChildren)

POINTER(DotToken)
POINTER(DotDotDotToken)
POINTER(QuestionToken)
POINTER(ExclamationToken)
POINTER(ColonToken)
POINTER(EqualsToken)
POINTER(AsteriskToken)
POINTER(EqualsGreaterThanToken)
POINTER(PlusToken)
POINTER(MinusToken)
POINTER(QuestionDotToken)

POINTER_VAR(LiteralToken, SyntaxKind)
POINTER(BinaryOperatorToken)

POINTER(AssertsKeyword)
POINTER(AwaitKeyword)

POINTER(AwaitKeywordToken)
POINTER(AssertsToken)

POINTER(AbstractKeyword)
POINTER(AsyncKeyword)
POINTER(ConstKeyword)
POINTER(DeclareKeyword)
POINTER(DefaultKeyword)
POINTER(ExportKeyword)
POINTER(PrivateKeyword)
POINTER(ProtectedKeyword)
POINTER(PublicKeyword)
POINTER(ReadonlyKeyword)
POINTER(StaticKeyword)

POINTER(BindableStaticNameExpression)
POINTER(BindableStaticAccessExpression)
POINTER(BindableAccessExpression)
POINTER(CallLikeExpression)

POINTER(TemplateLiteral)

POINTER(JsxChild)
POINTER(JsxOpeningLikeElement)
POINTER(JsxAttributeLike)
POINTER(JsxTagNameExpression)

POINTER(ForInitializer)
POINTER(BreakOrContinueStatement)
POINTER(CaseOrDefaultClause)

POINTER(ModuleName)
POINTER(ModuleBody)
POINTER(ModuleReference)
POINTER(NamespaceBody)
POINTER(JSDocNamespaceBody)

POINTER(NamedImportBindings)
POINTER(NamedExportBindings)

POINTER(ImportOrExportSpecifier)
POINTER(TypeOnlyCompatibleAliasDeclaration)

POINTER(JSDocComment)

POINTER(JSDocTypeReferencingNode)
POINTER(HasJSDoc)
POINTER(DestructuringPattern)
POINTER(BindingElementGrandparent)

POINTER(ObjectLiteralElementLike)
POINTER(StringLiteralLike)
POINTER(PropertyNameLiteral)

POINTER(UnparsedSource)
POINTER(UnparsedSourceText)
POINTER(UnparsedNode)
POINTER(UnparsedPrologue)
POINTER(UnparsedPrepend)
POINTER(UnparsedTextLike)
POINTER(UnparsedSyntheticReference)

POINTER(EmitHelper)
POINTER(InputFiles)
POINTER(SyntaxList)

POINTER(Bundle)
POINTER(PropertyDescriptorAttributes)
POINTER(CallBinding)

POINTER(JsonSourceFile)
POINTER(UnionOrIntersectionTypeNode)
POINTER(AccessorDeclaration)
POINTER(NamedImportsOrExports)

POINTER(NodeWithDiagnostics)
POINTER(BooleanLiteral)
POINTER(JsonObjectExpressionStatement)
POINTER(JsonObjectExpression)
POINTER(FunctionOrConstructorTypeNode)
POINTER(AssignmentPattern)

POINTER_T(Push)
POINTER_T(VisitResult)
POINTER_TARGS(Token, SyntaxKind)

using PrefixUnaryOperator = SyntaxKind;
using PostfixUnaryOperator = SyntaxKind;

template <typename T> using NodeArray = data::NodeArray<T>;

using ModifiersArray = NodeArray<Modifier>;
using ModifiersLikeArray = NodeArray<ModifierLike>;
using DecoratorsArray = NodeArray<Decorator>;

template <typename R = Node, typename T = Node> using ArrayFuncT = std::function<R(NodeArray<T>)>;

template <typename R = Node, typename T = Node> using ArrayFuncWithParentT = std::function<R(NodeArray<T>, T)>;
} // namespace ts

#endif // NEW_PARSER_FWRD_TYPES2_H