#ifndef TYPESCRIPT_AST_H_
#define TYPESCRIPT_AST_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>
#include <stack>

#include "TypeScriptParserANTLR.h"
#include "EnumsAST.h"

#define MAKE(ty, ctx)  \
    static std::shared_ptr<ty> parse(TypeScriptParserANTLR::ctx* _ctx) { \
        return _ctx ? std::make_shared<ty>(_ctx) : nullptr; \
    }

#define PASS(ty, ctx, fld)  \
    static std::shared_ptr<ty> parse(TypeScriptParserANTLR::ctx* _ctx) { \
        return _ctx ? std::static_pointer_cast<ty>(parse(_ctx->fld())) : nullptr;  \
    } 

#define PASS_COLL(ty, ctx)  \
    static std::vector<std::shared_ptr<ty>> parse(std::vector<TypeScriptParserANTLR::ctx *> _ctx) { \
        std::vector<std::shared_ptr<ty>> items; \
        for (auto *item : _ctx) \
        {   \
            items.push_back(std::static_pointer_cast<ty>(parse(item)));   \
        }   \
    \
        return items;   \
    } 

#define PASS_FIELD_COLL(ty, ctx, fld)  \
    static std::vector<std::shared_ptr<ty>> parse(TypeScriptParserANTLR::ctx* _ctx) { \
        std::vector<std::shared_ptr<ty>> items; \
        for (auto *item : _ctx->fld()) \
        {   \
            items.push_back(std::static_pointer_cast<ty>(parse(item)));   \
        }   \
    \
        return items;   \
    } 

#define PASS_CHOICES(ctx)  \
    static std::shared_ptr<NodeAST> parse(TypeScriptParserANTLR::ctx* _ctx) {    \
        if (_ctx)   \
        {

#define PASS_CHOICES_TYPED(ty, ctx)  \
    static std::shared_ptr<ty> parse(TypeScriptParserANTLR::ctx* _ctx) {    \
        if (_ctx)   \
        {            

#define PASS_CHOICE_TYPED(fld)  \
            if (auto _fld = _ctx->fld()) \
            {   \
                return parse(_fld);    \
            }

#define PASS_CHOICE(fld)  \
            if (auto _fld = _ctx->fld()) \
            {   \
                return std::static_pointer_cast<NodeAST>(parse(_fld));    \
            }

#define PASS_CHOICE_FIRST(fld)  \
            if (auto _fld = _ctx->fld().front()) \
            {   \
                return std::static_pointer_cast<NodeAST>(parse(_fld));    \
            }            

#define MAKE_CHOICE_IF_TYPED(cond, ty)  \
            if (_ctx->cond()) \
            {   \
                return std::make_shared<ty>(_ctx);    \
            }      

#define MAKE_CHOICE_IF(cond, ty)  \
            if (_ctx->cond()) \
            {   \
                return std::static_pointer_cast<NodeAST>(std::make_shared<ty>(_ctx));    \
            }                    

#define MAKE_CHOICE_IF_ANY(cond, ty)  \
            if (_ctx->cond().size() > 0) \
            {   \
                return std::static_pointer_cast<NodeAST>(std::make_shared<ty>(_ctx));    \
            }                    

#define PASS_CHOICE_END()  \
        } \
        \
        return nullptr; \
    } 

namespace typescript
{
    struct TextRange
    {
        int pos;
        int end;
        TextRange(antlr4::tree::ParseTree *tree)
        {
            const antlr4::misc::Interval &loc = tree->getSourceInterval();
            pos = static_cast<int>(loc.a);
            end = static_cast<int>(loc.b);
        }
    };

    class NodeAST;
    class NullLiteralAST;
    class TrueLiteralAST;
    class FalseLiteralAST;
    class NumericLiteralAST;
    class StringLiteralAST;
    class IdentifierAST;
    class TypeReferenceAST;
    class PropertyAccessExpressionAST;
    class ConditionalExpressionAST;
    class CommaListExpressionAST;
    class CallExpressionAST;
    class ParameterDeclarationAST;
    class ParametersDeclarationAST;
    class FunctionDeclarationAST;
    class ModuleBlockAST;
    class ModuleAST;

    template <typename Ty>
    static std::vector<Ty> merge(std::vector<Ty> data, Ty item)
    {
        data.push_back(item);
        return data;        
    }
    
    inline std::string text(antlr4::tree::TerminalNode* node)
    {
        return node ? node->toString() : "";
    }      

    MAKE(IdentifierAST, IdentifierContext)

    PASS(IdentifierAST, BindingIdentifierContext, identifier)
    
    MAKE(IdentifierAST, OptionalChainContext)

    MAKE(TypeReferenceAST, TypeDeclarationContext)

    PASS(TypeReferenceAST, TypeParameterContext, typeDeclaration)

    PASS(NodeAST, IdentifierReferenceContext, identifier)

    MAKE(NullLiteralAST, NullLiteralContext)
    
    PASS_CHOICES(BooleanLiteralContext)
    MAKE_CHOICE_IF(TRUE_KEYWORD, TrueLiteralAST)
    MAKE_CHOICE_IF(FALSE_KEYWORD, TrueLiteralAST)
    PASS_CHOICE_END()

    PASS_CHOICES(NumericLiteralContext)
    MAKE_CHOICE_IF(DecimalLiteral, NumericLiteralAST)
    PASS_CHOICE_END()    

    PASS_CHOICES(LiteralContext)
    PASS_CHOICE(nullLiteral)
    PASS_CHOICE(booleanLiteral)
    PASS_CHOICE(numericLiteral)
    MAKE_CHOICE_IF(StringLiteral, StringLiteralAST)
    PASS_CHOICE_END()  

    PASS_CHOICES(PrimaryExpressionContext)
    PASS_CHOICE(literal)
    PASS_CHOICE(identifierReference)
    PASS_CHOICE_END()    

    PASS_CHOICES(MemberExpressionContext)
    PASS_CHOICE(primaryExpression)
    //MAKE_CHOICE_IF(DOT_TOKEN, MemberExpressionAST)
    PASS_CHOICE_END()

    MAKE(CallExpressionAST, CoverCallExpressionAndAsyncArrowHeadContext)

    PASS_CHOICES(NewExpressionContext)
    PASS_CHOICE(memberExpression)
    //MAKE_CHOICE_IF(NEW_KEYWORD, NewExpressionAST)
    PASS_CHOICE_END()

    PASS_CHOICES_TYPED(CallExpressionAST, CallExpressionContext)
    PASS_CHOICE_TYPED(coverCallExpressionAndAsyncArrowHead)
    MAKE_CHOICE_IF_TYPED(callExpression, CallExpressionAST)
    PASS_CHOICE_END()    

    MAKE(PropertyAccessExpressionAST, OptionalExpressionContext)

    PASS_CHOICES(LeftHandSideExpressionContext)
    PASS_CHOICE(newExpression)
    PASS_CHOICE(callExpression)
    PASS_CHOICE(optionalExpression)
    PASS_CHOICE_END()

    PASS(NodeAST, UpdateExpressionContext, leftHandSideExpression)

    PASS(NodeAST, UnaryExpressionContext, updateExpression)

    PASS(NodeAST, ExponentiationExpressionContext, unaryExpression)

    PASS(NodeAST, MultiplicativeExpressionContext, exponentiationExpression)

    PASS(NodeAST, AdditiveExpressionContext, multiplicativeExpression)

    PASS(NodeAST, ShiftExpressionContext, additiveExpression)

    PASS(NodeAST, RelationalExpressionContext, shiftExpression)

    PASS_CHOICES(EqualityExpressionContext)
    PASS_CHOICE(relationalExpression)
    PASS_CHOICE_END()

    PASS(NodeAST, BitwiseANDExpressionContext, equalityExpression)

    PASS(NodeAST, BitwiseXORExpressionContext, bitwiseANDExpression)

    PASS(NodeAST, BitwiseORExpressionContext, bitwiseXORExpression)

    PASS(NodeAST, LogicalANDExpressionContext, bitwiseORExpression)

    PASS(NodeAST, LogicalORExpressionContext, logicalANDExpression)

    PASS(NodeAST, ShortCircuitExpressionContext, logicalORExpression)

    PASS_CHOICES(ConditionalExpressionContext)
    MAKE_CHOICE_IF(QUESTION_TOKEN, ConditionalExpressionAST)
    PASS_CHOICE(shortCircuitExpression)
    PASS_CHOICE_END()

    PASS(NodeAST, AssignmentExpressionContext, conditionalExpression)

    PASS_COLL(NodeAST, AssignmentExpressionContext);

    PASS_CHOICES(ExpressionContext)
    MAKE_CHOICE_IF_ANY(COMMA_TOKEN, CommaListExpressionAST)
    PASS_CHOICE_FIRST(assignmentExpression)
    PASS_CHOICE_END()

    PASS_FIELD_COLL(NodeAST, ArgumentsContext, expression)

    PASS(NodeAST, InitializerContext, assignmentExpression)

    MAKE(ParameterDeclarationAST, FormalParameterContext)    

    MAKE(ParameterDeclarationAST, FunctionRestParameterContext)    

    PASS_COLL(ParameterDeclarationAST, FormalParameterContext)

    MAKE(ParametersDeclarationAST, FormalParametersContext)    

    MAKE(FunctionDeclarationAST, FunctionDeclarationContext)    

    PASS(NodeAST, HoistableDeclarationContext, functionDeclaration)

    PASS(NodeAST, DeclarationContext, hoistableDeclaration)

    static std::shared_ptr<NodeAST> parse(TypeScriptParserANTLR::StatementContext* statement) {
        return nullptr;
    }  

    PASS_CHOICES(StatementListItemContext)
    PASS_CHOICE(statement)
    PASS_CHOICE(declaration)
    PASS_CHOICE_END()

    PASS(NodeAST, ModuleItemContext, statementListItem)  
   
    PASS_COLL(NodeAST, ModuleItemContext)

    MAKE(ModuleBlockAST, ModuleBodyContext)    

    MAKE(ModuleAST, MainContext)    

    // nodes
    class NodeAST
    {
    public:
        using TypePtr = std::shared_ptr<NodeAST>;

        NodeAST(SyntaxKind kind, TextRange range)
            : kind(kind), range(range) {}
        virtual ~NodeAST() = default;

        SyntaxKind getKind() const { return kind; }

        const TextRange &getLoc() { return range; }

    protected:
        TextRange range;
        SyntaxKind kind;
        NodeFlags flags;
        NodeAST *parent;
    };

    class BlockAST : public NodeAST
    {
        std::vector<NodeAST::TypePtr> items;

    public:
        using TypePtr = std::shared_ptr<BlockAST>;

        // TODO: remove it when finish
        BlockAST(TextRange range, std::vector<NodeAST::TypePtr> items)
            : NodeAST(SyntaxKind::Block, range), items(items) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::Block;
        }               
    };   

    class NullLiteralAST : public NodeAST
    {
    public:
        using TypePtr = std::shared_ptr<NullLiteralAST>;

        // TODO: remove it when finish
        NullLiteralAST(TextRange range)
            : NodeAST(SyntaxKind::NullKeyword, range) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::NullKeyword;
        }               
    };      

    class TrueLiteralAST : public NodeAST
    {
    public:
        using TypePtr = std::shared_ptr<TrueLiteralAST>;

        // TODO: remove it when finish
        TrueLiteralAST(TextRange range)
            : NodeAST(SyntaxKind::TrueKeyword, range) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::TrueKeyword;
        }               
    };     

    class FalseLiteralAST : public NodeAST
    {
    public:
        using TypePtr = std::shared_ptr<FalseLiteralAST>;

        // TODO: remove it when finish
        FalseLiteralAST(TextRange range)
            : NodeAST(SyntaxKind::FalseKeyword, range) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::FalseKeyword;
        }               
    };     

    class NumericLiteralAST : public NodeAST
    {
        long longVal;
        double doubleVal;
    public:
        using TypePtr = std::shared_ptr<NumericLiteralAST>;

        NumericLiteralAST(TypeScriptParserANTLR::NumericLiteralContext* numericLiteralContext) 
            : NodeAST(SyntaxKind::NumericLiteral, TextRange(numericLiteralContext))
        {
            parseNode(numericLiteralContext);
        }

        NumericLiteralAST(TextRange range, long longVal)
            : NodeAST(SyntaxKind::NumericLiteral, range), longVal(longVal) {}

        NumericLiteralAST(TextRange range, double doubleVal)
            : NodeAST(SyntaxKind::NumericLiteral, range), doubleVal(doubleVal) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::NumericLiteral;
        }   

    private:
        void parseNode(TypeScriptParserANTLR::NumericLiteralContext* numericLiteralContext)
        {
            if (numericLiteralContext->DecimalLiteral())
            {
                doubleVal = std::stod(text(numericLiteralContext->DecimalLiteral()));
            }
            else if (numericLiteralContext->DecimalIntegerLiteral())
            {
                longVal = std::stol(text(numericLiteralContext->DecimalIntegerLiteral()));
            }
        }            
    };    

    class BigIntLiteralAST : public NodeAST
    {
        long long longVal;
    public:
        using TypePtr = std::shared_ptr<BigIntLiteralAST>;

        BigIntLiteralAST(TypeScriptParserANTLR::NumericLiteralContext* numericLiteralContext) 
            : NodeAST(SyntaxKind::BigIntLiteral, TextRange(numericLiteralContext))
        {
            parseNode(numericLiteralContext);
        }        

        BigIntLiteralAST(TextRange range, long long longVal)
            : NodeAST(SyntaxKind::BigIntLiteral, range), longVal(longVal) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::BigIntLiteral;
        }    

    private:
        void parseNode(TypeScriptParserANTLR::NumericLiteralContext* numericLiteralContext)
        {
            if (numericLiteralContext->DecimalBigIntegerLiteral())
            {
                longVal = std::stoll(text(numericLiteralContext->DecimalBigIntegerLiteral()));
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }                    
    };     

    class StringLiteralAST : public NodeAST
    {
        std::string val;
    public:
        using TypePtr = std::shared_ptr<StringLiteralAST>;

        StringLiteralAST(TypeScriptParserANTLR::LiteralContext* literalContext) 
            : NodeAST(SyntaxKind::StringLiteral, TextRange(literalContext)),
              val(text(literalContext->StringLiteral())) {}

        StringLiteralAST(TextRange range, std::string val)
            : NodeAST(SyntaxKind::StringLiteral, range), val(val) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::StringLiteral;
        }               
    };

    class IdentifierAST : public NodeAST
    {
        std::string name;
    public:
        using TypePtr = std::shared_ptr<IdentifierAST>;

        IdentifierAST(TypeScriptParserANTLR::IdentifierContext* identifierContext) 
            : IdentifierAST(TextRange(identifierContext), text(identifierContext->IdentifierName())) {}     

        IdentifierAST(TypeScriptParserANTLR::OptionalChainContext* optionalChainContext) 
            : IdentifierAST(TextRange(optionalChainContext), text(optionalChainContext->IdentifierName())) {}     

        IdentifierAST(TextRange range, std::string identifier)
            : NodeAST(SyntaxKind::Identifier, range), name(identifier) {}

        const std::string& getName() const { return name; }

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::Identifier;
        }      
    };

    class TypeReferenceAST : public NodeAST
    {
        std::string typeName;
        SyntaxKind typeKind;
    public:
        using TypePtr = std::shared_ptr<TypeReferenceAST>;

        TypeReferenceAST(TypeScriptParserANTLR::TypeDeclarationContext* typeDeclarationContext) 
            : NodeAST(SyntaxKind::TypeReference, TextRange(typeDeclarationContext)), 
              typeKind(parseKind(typeDeclarationContext)) {}   

        TypeReferenceAST(TextRange range, SyntaxKind typeKind)
            : NodeAST(SyntaxKind::Identifier, range), typeKind(typeKind) {}

        TypeReferenceAST(TextRange range, std::string typeName)
            : NodeAST(SyntaxKind::Identifier, range), typeKind(SyntaxKind::Unknown), typeName(typeName) {}

        SyntaxKind getTypeKind() const { return typeKind; }
        const std::string& getTypeName() const { return typeName; }

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::TypeReference;
        }           

    private:
        SyntaxKind parseKind(TypeScriptParserANTLR::TypeDeclarationContext* typeDeclarationContext)
        {
            if (auto anyKeyword = typeDeclarationContext->ANY_KEYWORD())
            {
                return SyntaxKind::AnyKeyword;
            }

            if (auto anyKeyword = typeDeclarationContext->NUMBER_KEYWORD())
            {
                return SyntaxKind::NumberKeyword;
            }

            if (auto anyKeyword = typeDeclarationContext->BOOLEAN_KEYWORD())
            {
                return SyntaxKind::BooleanKeyword;
            }

            if (auto anyKeyword = typeDeclarationContext->STRING_KEYWORD())
            {
                return SyntaxKind::StringKeyword;
            }            

            if (auto anyKeyword = typeDeclarationContext->BIGINT_KEYWORD())
            {
                return SyntaxKind::BigIntKeyword;
            }         

            llvm_unreachable("SyntaxKind is unknown");        
        }
    };    

    class PropertyAccessExpressionAST : public NodeAST
    {
        NodeAST::TypePtr memberExpression;
        IdentifierAST::TypePtr name;
    public:
        using TypePtr = std::shared_ptr<PropertyAccessExpressionAST>;

        PropertyAccessExpressionAST(TypeScriptParserANTLR::OptionalExpressionContext* optionalExpressionContext) 
            : NodeAST(SyntaxKind::PropertyAccessExpression, TextRange(optionalExpressionContext)), 
              memberExpression(parse(optionalExpressionContext->memberExpression())),
              name(parse(optionalExpressionContext->optionalChain())) {}     

        PropertyAccessExpressionAST(TextRange range, NodeAST::TypePtr memberExpression, IdentifierAST::TypePtr name)
            : NodeAST(SyntaxKind::Parameters, range), memberExpression(memberExpression), name(name) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::PropertyAccessExpression;
        }         
    };

    class CommaListExpressionAST : public NodeAST
    {
        std::vector<NodeAST::TypePtr> expressions;
    public:
        using TypePtr = std::shared_ptr<CommaListExpressionAST>;

        CommaListExpressionAST(TypeScriptParserANTLR::ExpressionContext* expressionContext) 
            : NodeAST(SyntaxKind::CommaListExpression, TextRange(expressionContext))/*,
              expressions(parse(expressionContext->assignmentExpression()))*/ {}     

        CommaListExpressionAST(TextRange range, std::vector<NodeAST::TypePtr> expressions)
            : NodeAST(SyntaxKind::Parameters, range), expressions(expressions) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::CommaListExpression;
        }          
    };

    class ConditionalExpressionAST : public NodeAST
    {
        NodeAST::TypePtr condition;
        NodeAST::TypePtr whenTrue;
        NodeAST::TypePtr whenFalse;
    public:
        using TypePtr = std::shared_ptr<ConditionalExpressionAST>;

        ConditionalExpressionAST(TypeScriptParserANTLR::ConditionalExpressionContext* conditionalExpressionContext) 
            : NodeAST(SyntaxKind::ConditionalExpression, TextRange(conditionalExpressionContext)) {}     

        ConditionalExpressionAST(TextRange range, NodeAST::TypePtr condition, NodeAST::TypePtr whenTrue, NodeAST::TypePtr whenFalse)
            : NodeAST(SyntaxKind::Parameters, range), condition(condition), whenTrue(whenTrue), whenFalse(whenFalse) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::ConditionalExpression;
        }         
    };

    class CallExpressionAST : public NodeAST
    {
        NodeAST::TypePtr expression;
        std::vector<NodeAST::TypePtr> arguments;
    public:
        using TypePtr = std::shared_ptr<CallExpressionAST>;

        CallExpressionAST(TypeScriptParserANTLR::CallExpressionContext* callExpressionContext) 
            : NodeAST(SyntaxKind::CallExpression, TextRange(callExpressionContext)),
              expression(parse(callExpressionContext->callExpression())), 
              arguments(parse(callExpressionContext->arguments())) {}     

        CallExpressionAST(TypeScriptParserANTLR::CoverCallExpressionAndAsyncArrowHeadContext* coverCallExpressionAndAsyncArrowHeadContext) 
            : NodeAST(SyntaxKind::CallExpression, TextRange(coverCallExpressionAndAsyncArrowHeadContext)),
              expression(parse(coverCallExpressionAndAsyncArrowHeadContext->memberExpression())), 
              arguments(parse(coverCallExpressionAndAsyncArrowHeadContext->arguments())) {}     

        CallExpressionAST(TextRange range, NodeAST::TypePtr expression, std::vector<NodeAST::TypePtr> arguments)
            : NodeAST(SyntaxKind::Parameters, range), expression(expression), arguments(arguments) {}

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::CallExpression;
        }          
    };    

    class ParameterDeclarationAST : public NodeAST
    {
        IdentifierAST::TypePtr identifier;
        TypeReferenceAST::TypePtr type;
        NodeAST::TypePtr initializer;
        bool isOptional;
        bool dotdotdot;

    public:
        using TypePtr = std::shared_ptr<ParameterDeclarationAST>;

        ParameterDeclarationAST(TypeScriptParserANTLR::FormalParameterContext* formalParameterContext) 
            : NodeAST(SyntaxKind::Parameter, TextRange(formalParameterContext)),
              identifier(std::make_shared<IdentifierAST>(formalParameterContext->IdentifierName(), formalParameterContext->IdentifierName()->toString())),
              type(parse(formalParameterContext->typeParameter())),
              initializer(parse(formalParameterContext->initializer())),
              isOptional(!!formalParameterContext->QUESTION_TOKEN()) {}   

        ParameterDeclarationAST(TypeScriptParserANTLR::FunctionRestParameterContext* functionRestParameterContext) 
            : ParameterDeclarationAST(functionRestParameterContext->formalParameter()) 
        {
            dotdotdot = true;
        }   

        ParameterDeclarationAST(TextRange range, IdentifierAST::TypePtr identifier, TypeReferenceAST::TypePtr type, NodeAST::TypePtr initialize)
            : NodeAST(SyntaxKind::FunctionDeclaration, range), identifier(identifier), type(type), initializer(initializer) {}

        const IdentifierAST::TypePtr& getIdentifier() const { return identifier; }
        const TypeReferenceAST::TypePtr& getType() const { return type; }
        const NodeAST::TypePtr& getInitializer() const { return initializer; }
        bool getIsOptional() const { return isOptional; }
        bool getDotDotDot() const { return dotdotdot; }
        void setDotDotDot(bool val) { dotdotdot = val; }

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::Parameter;
        }
    };

    class ParametersDeclarationAST : public NodeAST
    {
        std::vector<ParameterDeclarationAST::TypePtr> parameters;

    public:
        using TypePtr = std::shared_ptr<ParametersDeclarationAST>;

        ParametersDeclarationAST(TypeScriptParserANTLR::FormalParametersContext* formalParametersContext) 
            : NodeAST(SyntaxKind::Parameters, TextRange(formalParametersContext)),
              parameters(merge(parse(formalParametersContext->formalParameter()), parse(formalParametersContext->functionRestParameter()))) {}     

        ParametersDeclarationAST(TextRange range, std::vector<ParameterDeclarationAST::TypePtr> parameters)
            : NodeAST(SyntaxKind::Parameters, range), parameters(parameters) {}

        const std::vector<ParameterDeclarationAST::TypePtr>& getParameters() const { return parameters; }

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::Parameters;
        }          
    };

    class FunctionDeclarationAST : public NodeAST
    {
        IdentifierAST::TypePtr identifier;
        ParametersDeclarationAST::TypePtr parameters;
        TypeReferenceAST::TypePtr typeParameter;

    public:
        using TypePtr = std::shared_ptr<FunctionDeclarationAST>;

        FunctionDeclarationAST(TypeScriptParserANTLR::FunctionDeclarationContext* functionDeclarationContext) 
            : NodeAST(SyntaxKind::FunctionDeclaration, TextRange(functionDeclarationContext)), 
              identifier(parse(functionDeclarationContext->bindingIdentifier())), 
              parameters(parse(functionDeclarationContext->formalParameters())), 
              typeParameter(parse(functionDeclarationContext->typeParameter())) {}     
              
        FunctionDeclarationAST(TextRange range, IdentifierAST::TypePtr identifier, ParametersDeclarationAST::TypePtr parameters, TypeReferenceAST::TypePtr typeParameter)
            : NodeAST(SyntaxKind::FunctionDeclaration, range), identifier(identifier), parameters(parameters), typeParameter(typeParameter) {}

        const IdentifierAST::TypePtr& getIdentifier() const { return identifier; }
        const ParametersDeclarationAST::TypePtr& getParameters() const { return parameters; }
        const TypeReferenceAST::TypePtr& getTypeParameter() const { return typeParameter; }

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::FunctionDeclaration;
        }
    };

    class ModuleBlockAST : public NodeAST
    {
        std::vector<NodeAST::TypePtr> items;

    public:
        using TypePtr = std::shared_ptr<ModuleBlockAST>;

        ModuleBlockAST(TypeScriptParserANTLR::ModuleBodyContext* moduleBodyContext) 
            : NodeAST(SyntaxKind::ModuleBlock, TextRange(moduleBodyContext)), 
              items(parse(moduleBodyContext->moduleItem())) {}        

        ModuleBlockAST(TextRange range, std::vector<NodeAST::TypePtr> items)
            : NodeAST(SyntaxKind::ModuleBlock, range), items(items) {}

        const std::vector<NodeAST::TypePtr>& getItems() const { return items; }

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::ModuleBlock;
        }               
    };   

    class ModuleAST : public NodeAST
    {
        ModuleBlockAST::TypePtr block;

    public:
        using TypePtr = std::shared_ptr<ModuleAST>;

        ModuleAST(TypeScriptParserANTLR::MainContext* mainContext) 
            : NodeAST(SyntaxKind::ModuleDeclaration, TextRange(mainContext)), 
              block(parse(mainContext->moduleBody())) {}

        ModuleAST(TextRange range, ModuleBlockAST::TypePtr block)
            : NodeAST(SyntaxKind::ModuleDeclaration, range), block(block) {}

        auto begin() -> decltype(block.get()->getItems().begin()) { return block.get()->getItems().begin(); }
        auto end() -> decltype(block.get()->getItems().end()) { return block.get()->getItems().end(); }

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::ModuleDeclaration;
        }         
    };

} // namespace typescript

#endif // TYPESCRIPT_AST_H_
