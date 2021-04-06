#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <string>
#include <vector>
#include <memory>

using namespace llvm;
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
    Expression initValue;

public:

    using TypePtr = std::shared_ptr<VariableDeclarationDOM>;

    VariableDeclarationDOM(StringRef name, mlir::Type type, mlir::Location loc, Expression initValue = Expression())
        : BaseDOM(Base_VariableDeclaration), name(name), type(type), loc(loc), initValue(initValue), readWrite(false)
    {
    }

    StringRef getName() const { return name; }
    const mlir::Type &getType() const { return type; }
    const mlir::Location &getLoc() const { return loc; }
    const Expression &getInitValue() const { return initValue; }
    bool hasInitValue() const { return !!initValue; }
    bool getReadWriteAccess() const { return readWrite; };
    void setReadWriteAccess(bool value = true) { readWrite = value; };
    bool getIsGlobal() const { return isGlobal; };
    void setIsGlobal(bool value = true) { isGlobal = value; };

protected:
    bool readWrite;
    bool isGlobal;
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
    mlir::Type returnType;

public:

    using TypePtr = std::shared_ptr<FunctionPrototypeDOM>;

    FunctionPrototypeDOM(const std::string &name, std::vector<FunctionParamDOM::TypePtr> args)
        : name(name), args(args)
    {
    }

    StringRef getName() const { return name; }
    // ArrayRef should not be "&" or "*"
    ArrayRef<FunctionParamDOM::TypePtr> getArgs() const { return args; }
    const mlir::Type &getReturnType() const { return returnType; }
    void setReturnType(mlir::Type returnType_) 
    {
        returnType = returnType_;
    }
};
