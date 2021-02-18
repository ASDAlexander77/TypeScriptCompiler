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
using namespace typescript;

namespace typescript
{
    class NodeAST;
}

class BaseDOM
{
public:
    enum BaseDOMKind
    {
        Base_VariableDeclaration,
        Base_FunctionParam,
        Base_FunctionProto,
    };

    using TypePtr = std::shared_ptr<BaseDOM>;

    BaseDOM(BaseDOMKind kind) : kind(kind)
    {
    }

    virtual ~BaseDOM() = default;

    BaseDOMKind getKind() const { return kind; }

private:
    const BaseDOMKind kind;
};

class VariableDeclarationDOM : public BaseDOM
{
    std::string name;
    mlir::Type type;
    mlir::Location loc;
    std::shared_ptr<NodeAST> initValue;

public:

    using TypePtr = std::shared_ptr<VariableDeclarationDOM>;

    VariableDeclarationDOM(StringRef name, mlir::Type type, mlir::Location loc, std::shared_ptr<NodeAST> initValue = nullptr)
        : BaseDOM(Base_VariableDeclaration), name(name), type(type), loc(loc), initValue(initValue)
    {
    }

    StringRef getName() const { return name; }
    const mlir::Type &getType() const { return type; }
    const mlir::Location &getLoc() const { return loc; }
    const std::shared_ptr<NodeAST> &getInitValue() const { return initValue; }
    bool hasInitValue() const { return !!initValue; }
    bool getReadWriteAccess() const { return readWrite; };
    void SetReadWriteAccess() { readWrite = true; };

protected:
    bool readWrite;
};

class FunctionParamDOM : public VariableDeclarationDOM
{    
public:

    using TypePtr = std::shared_ptr<FunctionParamDOM>;

    FunctionParamDOM(StringRef name, mlir::Type type, mlir::Location loc, bool isOptional = false, std::shared_ptr<NodeAST> initValue = nullptr)
        : isOptional(isOptional), VariableDeclarationDOM(name, type, loc, initValue)
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

    using TypePtr = std::shared_ptr<FunctionPrototypeDOM>;

    FunctionPrototypeDOM(const std::string &name, std::vector<FunctionParamDOM::TypePtr> args)
        : name(name), args(args)
    {
    }

    StringRef getName() const { return name; }
    // ArrayRef should not be "&" or "*"
    ArrayRef<FunctionParamDOM::TypePtr> getArgs() const { return args; }
};
