#ifndef TYPESCRIPT_AST_H_
#define TYPESCRIPT_AST_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>
#include <stack>

#include "TypeScriptParserANTLR.h"
#include "EnumsAST.h"

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
    class IdentifierAST;
    class TypeReferenceAST;
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

    // parsers
    static std::shared_ptr<IdentifierAST> parse(TypeScriptParserANTLR::IdentifierContext* identifierContext) {
        return identifierContext ? std::make_shared<IdentifierAST>(identifierContext) : nullptr;
    } 

    static std::shared_ptr<IdentifierAST> parse(TypeScriptParserANTLR::BindingIdentifierContext* bindingIdentifierContext) {
        return bindingIdentifierContext ? parse(bindingIdentifierContext->identifier()) : nullptr;
    } 

    static std::shared_ptr<TypeReferenceAST> parse(TypeScriptParserANTLR::TypeDeclarationContext* typeDeclarationContext) {
        return typeDeclarationContext ? std::make_shared<TypeReferenceAST>(typeDeclarationContext) : nullptr;
    } 

    static std::shared_ptr<TypeReferenceAST> parse(TypeScriptParserANTLR::TypeParameterContext* typeParameterContext) {
        return typeParameterContext ? parse(typeParameterContext->typeDeclaration()) : nullptr;
    } 

    static std::shared_ptr<ParameterDeclarationAST> parse(TypeScriptParserANTLR::FormalParameterContext* formalParameterContext) {
        return formalParameterContext ? std::make_shared<ParameterDeclarationAST>(formalParameterContext) : nullptr;
    }  

    static std::shared_ptr<ParameterDeclarationAST> parse(TypeScriptParserANTLR::FunctionRestParameterContext* functionRestParameterContext) {
        return functionRestParameterContext ? std::make_shared<ParameterDeclarationAST>(functionRestParameterContext) : nullptr;
    }      

    static std::vector<std::shared_ptr<ParameterDeclarationAST>> parse(std::vector<TypeScriptParserANTLR::FormalParameterContext *> formalParameterItems) {
        std::vector<std::shared_ptr<ParameterDeclarationAST>> items;
        for (auto *item : formalParameterItems)
        {
            items.push_back(parse(item));
        }

        return items;
    }    

    static std::shared_ptr<ParametersDeclarationAST> parse(TypeScriptParserANTLR::FormalParametersContext* formalParameters) {
        return formalParameters ? std::make_shared<ParametersDeclarationAST>(formalParameters) : nullptr;
    } 

    static std::shared_ptr<NodeAST> parse(TypeScriptParserANTLR::FunctionDeclarationContext* functionDeclaration) {
        return functionDeclaration ? std::static_pointer_cast<NodeAST>(std::make_shared<FunctionDeclarationAST>(functionDeclaration)) : nullptr;
    } 

    static std::shared_ptr<NodeAST> parse(TypeScriptParserANTLR::HoistableDeclarationContext* hoistableDeclaration) {
        if (hoistableDeclaration)
        {
            if (auto functionDeclaration = hoistableDeclaration->functionDeclaration())
            {
                return parse(functionDeclaration);
            }
        }

        return nullptr;
    }  

    static std::shared_ptr<NodeAST> parse(TypeScriptParserANTLR::DeclarationContext* declaration) {
        if (declaration)
        {
            if (auto hoistableDeclaration = declaration->hoistableDeclaration())
            {
                return parse(hoistableDeclaration);
            }
        }

        return nullptr;
    }  

    static std::shared_ptr<NodeAST> parse(TypeScriptParserANTLR::StatementContext* statement) {
        return nullptr;
    }  


    static std::shared_ptr<NodeAST> parse(TypeScriptParserANTLR::StatementListItemContext* statementListItem) {
        if (statementListItem)
        { 
            if (auto statement = statementListItem->statement())
            {
                return parse(statement);
            }

            if (auto declaration = statementListItem->declaration())
            {
                return parse(declaration);
            }
        }

        return nullptr;
    }  

    static std::shared_ptr<NodeAST> parse(TypeScriptParserANTLR::ModuleItemContext* moduleItem) {
        return moduleItem ? parse(moduleItem->statementListItem()) : nullptr;
    }          

    static std::vector<std::shared_ptr<NodeAST>> parse(std::vector<TypeScriptParserANTLR::ModuleItemContext *> moduleItems) {
        std::vector<std::shared_ptr<NodeAST>> items;
        for (auto *item : moduleItems)
        {
            items.push_back(std::static_pointer_cast<NodeAST>(parse(item)));
        }

        return items;
    }      

    static std::shared_ptr<ModuleBlockAST> parse(TypeScriptParserANTLR::ModuleBodyContext* moduleBodyContext) {
        return moduleBodyContext ? std::make_shared<ModuleBlockAST>(moduleBodyContext) : nullptr;
    }

    static std::shared_ptr<ModuleAST> parse(TypeScriptParserANTLR::MainContext* mainContext) {
        return mainContext ? std::make_shared<ModuleAST>(mainContext) : nullptr;
    }

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

    class IdentifierAST : public NodeAST
    {
        std::string name;
    public:
        using TypePtr = std::shared_ptr<IdentifierAST>;

        IdentifierAST(TypeScriptParserANTLR::IdentifierContext* identifierContext) 
            : NodeAST(SyntaxKind::Identifier, TextRange(identifierContext)), 
              name(identifierContext->IdentifierName() ? identifierContext->IdentifierName()->toString() : "") {}     

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

    class ParameterDeclarationAST : public NodeAST
    {
        IdentifierAST::TypePtr identifier;
        TypeReferenceAST::TypePtr type;
        NodeAST::TypePtr initializer;
        bool dotdotdot;

    public:
        using TypePtr = std::shared_ptr<ParameterDeclarationAST>;

        ParameterDeclarationAST(TypeScriptParserANTLR::FormalParameterContext* formalParameterContext) 
            : NodeAST(SyntaxKind::Parameter, TextRange(formalParameterContext)),
              identifier(std::make_shared<IdentifierAST>(formalParameterContext->IdentifierName(), formalParameterContext->IdentifierName()->toString())),
              type(parse(formalParameterContext->typeParameter())) {}   

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
        bool getDotDotDot() { return dotdotdot; }
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
        NodeAST::TypePtr typeParameter;

    public:
        using TypePtr = std::shared_ptr<FunctionDeclarationAST>;

        FunctionDeclarationAST(TypeScriptParserANTLR::FunctionDeclarationContext* functionDeclarationContext) 
            : NodeAST(SyntaxKind::FunctionDeclaration, TextRange(functionDeclarationContext)), 
              identifier(parse(functionDeclarationContext->bindingIdentifier())), 
              parameters(parse(functionDeclarationContext->formalParameters())), 
              typeParameter(parse(functionDeclarationContext->typeParameter())) {}     
              
        FunctionDeclarationAST(TextRange range, IdentifierAST::TypePtr identifier, ParametersDeclarationAST::TypePtr parameters, NodeAST::TypePtr typeParameter)
            : NodeAST(SyntaxKind::FunctionDeclaration, range), identifier(identifier), parameters(parameters), typeParameter(typeParameter) {}

        const IdentifierAST::TypePtr& getIdentifier() const { return identifier; }
        const ParametersDeclarationAST::TypePtr& getParameters() const { return parameters; }

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
