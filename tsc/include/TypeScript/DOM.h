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

protected:
    tree::ParseTree *parseTree;

private:
    const BaseDOMKind kind;
};

class VariableDeclarationDOM : public BaseDOM
{
    std::string name;
    mlir::Type type;
    std::unique_ptr<BaseDOM> initVal;

public:

    using TypePtr = std::unique_ptr<VariableDeclarationDOM>;

    VariableDeclarationDOM(tree::ParseTree *parseTree, StringRef name, mlir::Type type, BaseDOM::TypePtr initVal = nullptr)
        : BaseDOM(Base_VariableDeclaration, parseTree), name(name), type(std::move(type)),
          initVal(std::move(initVal))
    {
    }

    StringRef getName() { return name; }
    BaseDOM *getInitVal() { return initVal.get(); }
    const mlir::Type &getType() { return type; }

protected:
    tree::ParseTree *parseTree;
};

class FunctionParamDOM : public VariableDeclarationDOM
{    
public:

    using TypePtr = std::unique_ptr<FunctionParamDOM>;

    FunctionParamDOM(tree::ParseTree *parseTree, StringRef name, mlir::Type type, bool hasInitValue = false, BaseDOM::TypePtr initVal = nullptr)
        : hasInitValue(hasInitValue), VariableDeclarationDOM(parseTree, name, type, std::move(initVal))
    {
    }

    /// LLVM style RTTI
    static bool classof(const BaseDOM *c) { return c->getKind() == Base_FunctionParam; }

private:
    bool hasInitValue;
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
