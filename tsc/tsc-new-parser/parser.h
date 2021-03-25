#ifndef PARSER_H
#define PARSER_H

#include "undefined.h"
#include "enums.h"
#include "types.h"
#include "scanner.h"
#include "debug.h"

#include <memory>

struct Node;

template <typename T>
using NodeFuncT = std::function<T(Node)>;

template <typename T>
using NodeWithParentFuncT = std::function<T(Node, Node)>;

typedef std::function<Node(SyntaxKind, number, number)> NodeCreateFunc;

typedef std::function<void(number, number, DiagnosticMessage)> PragmaDiagnosticReporter;

template <SyntaxKind kind>
using Token = Node;

struct Node;

template <typename T>
struct NodeArray
{
    TextRange range;

    NodeArray() = default;
    NodeArray(std::initializer_list<T> il) : items(il) {}
    NodeArray(undefined_t) : items() {}

    std::vector<T> items;
    boolean isMissingList;

    operator TextRange()
    {
        return range;
    }

    inline auto operator [](size_t i) const -> T
    {
        return items[i];
    }

    inline auto operator [](size_t i) -> T&
    {
        return items[i];
    }

    inline auto operator !() -> boolean
    {
        return items.size() == 0;
    }    

    inline auto clear() -> void {
        return items.clear();
    }

    inline auto size() -> size_t {
        return items.size();
    }

    auto push_back(T node) -> void
    {
        items.push_back(node);
    }       

    auto pop() -> T
    {
        auto v = items.back();
        items.pop_back();
        return v;
    }

    TextRange* operator ->()
    {
        return &range;
    }

    inline auto operator ==(undefined_t) -> boolean
    {
        return items.size() == 0;
    } 

    inline auto operator !=(undefined_t) -> boolean
    {
        return items.size() != 0;
    }        

    auto begin() -> decltype(items.begin())
    {
        return items.begin();
    }

    auto end() -> decltype(items.end())
    {
        return items.end();
    }
};

template <typename T>
using NodeArrayFuncT = std::function<T(NodeArray<T>)>;

template <typename T>
using NodeWithParentArrayFuncT = std::function<T(NodeArray<T>, Node)>;

struct Decorator;
typedef NodeArray<Decorator> DecoratorsArray;
typedef Node Modifier;
typedef NodeArray<Modifier> ModifiersArray;


#define CLASS_DATA_BASE(x, b) struct x##Data : b##Data { using b##Data::b##Data;    \
    virtual ~x##Data() override {}

#define CLASS_DATA(x) CLASS_DATA_BASE(x, Node)

#define CLASS_DATA_END(x) };                \
    struct x : BaseNode {                   \
        x() {}                              \
        x(undefined_t) {}                   \
        x(Node node) : BaseNode(node) { node.data = std::static_pointer_cast<x##Data>(node.data); }   \
                                            \
        x##Data* operator->()               \
        {                                   \
            return static_cast<x##Data*>(node.operator->());   \
        }                                   \
        operator TextRange()                \
        {                                   \
            return *(TextRange*)node.data.get(); \
        }                                   \
        operator Node()                     \
        {                                   \
            return *this;                   \
        }                                   \
        inline operator SyntaxKind()        \
        {                                   \
            return node.data->kind;         \
        }                                   \
        inline auto operator==(undefined_t) \
        {                                   \
            return !node.data;              \
        }                                   \
        inline auto operator!=(undefined_t) \
        {                                   \
            return !!node.data;             \
        }                                   \
    };  

struct NodeData;

struct NodeRef
{
    std::shared_ptr<NodeData> data;

    operator Node();
    boolean operator !();
    boolean operator ==(const Node& rhv);
    Node operator=(Node& rhv);
    NodeData* operator->();

    size_t size();
    auto begin() -> decltype(((NodeArray<Node>*)nullptr)->begin());
    auto end() -> decltype(((NodeArray<Node>*)nullptr)->end());
};

struct NodeData : TextRange
{   
    NodeData() = default;

    SyntaxKind kind;
    NodeFlags flags;
    string text;
    TransformFlags transformFlags;
    DecoratorsArray decorators;
    ModifiersArray modifiers;
    SyntaxKind originalKeywordKind;
    NodeRef parent;
    NodeRef jsDoc;
    number jsdocDotPos;

    NodeArray<Node> children;

    NodeData(SyntaxKind kind, number start, number end) : kind(kind), TextRange{start, end} {};

    virtual ~NodeData() {}
};

struct Node
{
    std::shared_ptr<NodeData> data;

    Node() {};
    Node(undefined_t) {};
    Node(SyntaxKind kind, number start, number end) : data(std::make_shared<NodeData>(kind, start, end)) {};
    Node(NodeArray<Node> values) : data(std::make_shared<NodeData>(SyntaxKind::Array, -1, -1)) { data->children = values; };
    template <typename T>
    Node(NodeArray<T> values) : data(std::make_shared<NodeData>(SyntaxKind::Array, -1, -1)) { 
        data->children.clear();
        for (auto &item : values)
        {
            data->children.push_back(item); 
        }
    };

/*protected*/
    Node(std::shared_ptr<NodeData> data) : data(data) {};

    NodeData* operator->()
    {
        return data.get();
    }

    operator TextRange()
    {
        return *(TextRange*)data.get();
    }

    template <typename T> 
    auto as() -> T
    {
        return T(*this);
    }

    template <typename T> 
    auto asMutable() -> T
    {
        return T(*this);
    }    

    auto asArray() -> NodeArray<Node>
    {
        return data->children;
    }    

    template <typename T> 
    auto asArray() -> NodeArray<T>
    {
        NodeArray<T> ret;
        for (auto &item : data->children)
        {
            data->children.push_back(item);
        }

        return ret;
    }

    operator bool()
    {
        return (bool)data;
    }

    operator NodeArray<Node>()
    {
        return data->children;
    }
    
    inline operator SyntaxKind()
    {
        return data->kind;
    }

    auto operator=(undefined_t) -> Node&
    {
        data->kind = SyntaxKind::Unknown;
        data->children.clear();
        return *this;
    }    

    auto operator=(NodeArray<Node> values) -> Node&
    {
        data->kind = SyntaxKind::ArrayType;
        data->children = values;
        return *this;
    }

    auto operator||(Node rhs) -> Node
    {
        if (operator bool())
        {
            return *this;
        }

        return rhs;
    }

    auto size() -> number
    {
        return data->children.size();
    }

    auto operator[](number i) const -> Node
    {
        return data->children[i];
    }    

    auto operator[](number i) -> Node&
    {
        return data->children[i];
    }    

    auto push_back(Node node) -> void
    {
        data->kind = SyntaxKind::ArrayType;
        data->children.push_back(node);
    }    

    auto begin() -> decltype(data->children.begin())
    {
        return data->children.begin();
    }

    auto end() -> decltype(data->children.end())
    {
        return data->children.end();
    }

    auto operator!()
    {
        return !data || data->kind == SyntaxKind::Unknown;
    }

    auto operator==(undefined_t)
    {
        return !data;
    }

    auto operator!=(undefined_t)
    {
        return !!data;
    }

    auto operator==(std::nullptr_t)
    {
        return data.get() == nullptr;
    }

    auto operator!=(std::nullptr_t)
    {
        return data.get() != nullptr;
    }    
};

struct BaseNode
{
    Node node;
    BaseNode() : node(Node()) {}
    BaseNode(Node node) : node(node) {}

    inline operator Node()
    {
        return node;
    }

    inline operator SyntaxKind()
    {
        return node->kind;
    }

    inline bool operator !()
    {
        return !static_cast<bool>(node);
    }

    inline NodeData* operator->()
    {
        return node.operator->();
    }    

    auto operator||(Node rhs) -> Node
    {
        if (static_cast<bool>(node))
        {
            return *this;
        }

        return rhs;
    }

    template <typename T> 
    auto as() -> T
    {
        return T(node);
    }

    template <typename T> 
    auto asMutable() -> T
    {
        return T(node);
    }          
};

static auto isArray(Node &node) -> boolean
{
    return node.data->kind == SyntaxKind::Array;
}

typedef SyntaxKind PrefixUnaryOperator, PostfixUnaryOperator;

typedef Node PropertyName, PrivateIdentifier, LiteralExpression, EntityName, Expression, IndexSignatureDeclaration,
    TypeElement, UnaryExpression, UpdateExpression, LeftHandSideExpression, MemberExpression, JsxText, JsxChild, JsxTagNameExpression,
    JsxClosingFragment, PrimaryExpression, FunctionExpression, Statement, CaseOrDefaultClause, ArrayBindingElement,
    ObjectBindingPattern, ArrayBindingPattern, FunctionDeclaration, ClassElement, ClassExpression,
    ModuleBlock, SuperExpression, ThisExpression, PseudoBigInt, MissingDeclaration, JsonObjectExpressionStatement, BindingName,
    CallSignatureDeclaration, MethodSignature, ConstructSignatureDeclaration, IndexSignatureDeclaration,
    MemberName, ElementAccessChain, CallChain, NewExpression, ConciseBody,
    Expression, OmittedExpression, NonNullChain, SemicolonClassElement, EmptyStatement, ForInitializer, ContinueStatement, 
    BreakStatement, DebuggerStatement, ModuleName, ModuleBody, ModuleReference, NamedImportBindings, NamedImports,
    NamedExportBindings, NamedExports, DestructuringAssignment, PropertyDescriptorAttributes, CallBinding, Declaration;

typedef Node FalseLiteral, TrueLiteral, NullLiteral, BooleanLiteral, NumericLiteral, BigIntLiteral, StringLiteral, PropertyNameLiteral, RegularExpressionLiteral, 
    ObjectLiteralElementLike, TemplateLiteral, NoSubstitutionTemplateLiteral;

typedef Node ThisTypeNode, UnionTypeNode, IntersectionTypeNode;

typedef Node QuestionDotToken, EndOfFileToken, DotDotDotToken, QuestionToken, PlusToken, MinusToken,
    AsteriskToken, EqualsGreaterThanToken, ColonToken, ExclamationToken, EqualsToken;

typedef Node LiteralToken, BinaryOperatorToken;

typedef Node ReadonlyKeyword, AssertsKeyword, AwaitKeyword;

typedef Node JSDocAllType, JSDocUnknownType, JSDocNonNullableType, JSDocNullableType, JSDocOptionalType, JSDocVariadicType, JSDocNamepathType,
    JSDocAuthorTag, JSDocClassTag, JSDocPublicTag, JSDocPrivateTag, JSDocProtectedTag, JSDocReadonlyTag, JSDocUnknownTag, JSDocDeprecatedTag,
    JSDocParameterTag, JSDocPropertyTag; 

typedef Node JsxSelfClosingElement, JsxOpeningFragment, JsxAttributeLike, JsxTagNamePropertyAccess;

typedef Node UnparsedPrologue, UnparsedSyntheticReference, UnparsedSourceText, UnparsedSource, UnparsedPrepend, UnparsedTextLike, InputFiles;

typedef Node SyntheticExpression, SyntaxList, NotEmittedStatement, EndOfDeclarationMarker, SyntheticReferenceExpression, MergeDeclarationMarker, Bundle;

typedef Node PrologueDirective;

template<typename T>
using AssignmentExpression = Node;

template<typename T>
using Push = Node;

template<typename T>
using VisitResult = Node;

#include "parser_types.h"

struct DiagnosticWithLocation : DiagnosticWithDetachedLocation {
    SourceFile file;
};

namespace ts
{
    auto processCommentPragmas(SourceFile context, string sourceText) -> void;
    auto processPragmasIntoFields(SourceFile context, PragmaDiagnosticReporter reportDiagnostic) -> void;
    auto isExternalModule(SourceFile file) -> boolean;
    auto tagNamesAreEquivalent(JsxTagNameExpression lhs, JsxTagNameExpression rhs) -> boolean;
    auto fixupParentReferences(Node rootNode) -> void;
}

#include "incremental_parser.h"

#endif // PARSER_H