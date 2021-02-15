#ifndef TYPESCRIPT_AST_H_
#define TYPESCRIPT_AST_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>
#include <stack>

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

        // TODO: remove it when finish
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
    };    

    class ParameterDeclarationAST : public NodeAST
    {
        IdentifierAST::TypePtr identifier;
        TypeReferenceAST::TypePtr type;
        NodeAST::TypePtr initializer;
        bool dotdotdot;

    public:
        using TypePtr = std::shared_ptr<ParameterDeclarationAST>;

        // TODO: remove it when finish
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

        // TODO: remove it when finish
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
        TypeReferenceAST::TypePtr typeReference;

    public:
        using TypePtr = std::shared_ptr<FunctionDeclarationAST>;

        // TODO: remove it when finish
        FunctionDeclarationAST(TextRange range, IdentifierAST::TypePtr identifier, ParametersDeclarationAST::TypePtr parameters, TypeReferenceAST::TypePtr typeReference)
            : NodeAST(SyntaxKind::FunctionDeclaration, range), identifier(identifier), parameters(parameters), typeReference(typeReference) {}

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
            : NodeAST(SyntaxKind::ModuleDeclaration, TextRange(mainContext)) {}

        ModuleAST(TextRange range, ModuleBlockAST::TypePtr block)
            : NodeAST(SyntaxKind::ModuleDeclaration, range), block(block) {}

        auto begin() -> decltype(block.get()->getItems().begin()) { return block.get()->getItems().begin(); }
        auto end() -> decltype(block.get()->getItems().end()) { return block.get()->getItems().end(); }

        TypePtr parse(TypeScriptParserANTLR::MainContext* mainContext) {
            if (mainContext)
            {
                return std::make_shared<ModuleAST>(mainContext);
            }

            return nullptr;
        }

        /// LLVM style RTTI
        static bool classof(const NodeAST *N) 
        {
            return N->getKind() == SyntaxKind::ModuleDeclaration;
        }         
    };

} // namespace typescript

#endif // TYPESCRIPT_AST_H_
