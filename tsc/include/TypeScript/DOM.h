#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "TypeScriptLexerANTLR.h"
#include "TypeScriptParserANTLR.h"

#include <string>
#include <vector>
#include <memory>

using namespace llvm;
using namespace antlr4;

class BaseDOM
{
public:
    enum BaseDOMKind
    {
        Base_VariableDeclaration,
        Base_FunctionParam,
        Base_FunctionProto,
    };

    using TypePtr = std::unique_ptr<BaseDOM>;

    BaseDOM(BaseDOMKind kind, tree::ParseTree *parseTree) : kind(kind), parseTree(parseTree)
    {
    }

    virtual ~BaseDOM() = default;

    BaseDOMKind getKind() const { return kind; }

    tree::ParseTree *getParseTree() { return parseTree; }

protected:
    tree::ParseTree *parseTree;

private:
    const BaseDOMKind kind;
};

class VariableDeclarationDOM : public BaseDOM
{
    std::string name;
    mlir::Type type;
    tree::ParseTree *initVal;

public:

    using TypePtr = std::unique_ptr<VariableDeclarationDOM>;

    VariableDeclarationDOM(tree::ParseTree *parseTree, StringRef name, mlir::Type type, tree::ParseTree *initVal = nullptr)
        : BaseDOM(Base_VariableDeclaration, parseTree), name(name), type(std::move(type)),
          initVal(initVal)
    {
    }

    StringRef getName() { return name; }
    tree::ParseTree *getInitVal() { return initVal; }
    const mlir::Type &getType() { return type; }
    bool getReadWriteAccess() { return readWrite; };
    void SetReadWriteAccess() { readWrite = true; };

protected:
    tree::ParseTree *parseTree;
    bool readWrite;
};

class FunctionParamDOM : public VariableDeclarationDOM
{    
public:

    using TypePtr = std::unique_ptr<FunctionParamDOM>;

    FunctionParamDOM(tree::ParseTree *parseTree, StringRef name, mlir::Type type, bool isOptional = false, tree::ParseTree *initVal = nullptr)
        : isOptional(isOptional), VariableDeclarationDOM(parseTree, name, type, initVal)
    {
    }

    bool getIsOptional()
    {
        return isOptional;
    }

    /// LLVM style RTTI
    static bool classof(const BaseDOM *c) { return c->getKind() == Base_FunctionParam; }

private:
    bool isOptional;
};

class FunctionPrototypeDOM
{
    std::string name;
    std::vector<FunctionParamDOM::TypePtr> args;

public:

    using TypePtr = std::unique_ptr<FunctionPrototypeDOM>;

    FunctionPrototypeDOM(tree::ParseTree *parseTree, const std::string &name, std::vector<FunctionParamDOM::TypePtr> args)
        : parseTree(parseTree), name(name), args(std::move(args))
    {
    }

    StringRef getName() const { return name; }
    ArrayRef<FunctionParamDOM::TypePtr> getArgs() { return args; }

protected:
    tree::ParseTree *parseTree;
};

class ModuleDOM
{
    std::vector<FunctionPrototypeDOM::TypePtr> functionProtos;

public:

    using TypePtr = std::unique_ptr<ModuleDOM>;

    ModuleDOM()
    {
    }

    std::vector<FunctionPrototypeDOM::TypePtr>& getFunctionProtos() { return functionProtos; }

    tree::ParseTree *parseTree;
};
