#ifndef MLIR_TYPESCRIPT_DOM_H_
#define MLIR_TYPESCRIPT_DOM_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <string>
#include <vector>

#include "parser.h"

using namespace llvm;
namespace mlir_ts = mlir::typescript;

namespace ts
{
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

    BaseDOMKind getKind() const
    {
        return kind;
    }

  private:
    const BaseDOMKind kind;
};

class VariableDeclarationDOM : public BaseDOM
{
    std::string name;
    mlir::Type type;
    mlir::Location loc;
    Expression initValue;
    mlir_ts::FuncOp functionScope;

  public:
    using TypePtr = std::shared_ptr<VariableDeclarationDOM>;

    VariableDeclarationDOM(StringRef name, mlir::Type type, mlir::Location loc, Expression initValue = undefined)
        : BaseDOM(Base_VariableDeclaration), name(name), type(type), loc(loc), initValue(initValue), readWrite(false)
    {
    }

    StringRef getName() const
    {
        return name;
    }
    const mlir::Type &getType() const
    {
        return type;
    }
    const mlir::Location &getLoc() const
    {
        return loc;
    }
    const Expression &getInitValue() const
    {
        return initValue;
    }
    bool hasInitValue()
    {
        return !!initValue;
    }
    bool getReadWriteAccess() const
    {
        return readWrite;
    };
    void setReadWriteAccess(bool value = true)
    {
        readWrite = value;
    };
    mlir_ts::FuncOp getFuncOp() const
    {
        return functionScope;
    };
    void setFuncOp(mlir_ts::FuncOp value)
    {
        functionScope = value;
    };

  protected:
    bool readWrite;
};

class FunctionParamDOM : public VariableDeclarationDOM
{
  public:
    using TypePtr = std::shared_ptr<FunctionParamDOM>;

    FunctionParamDOM(StringRef name, mlir::Type type, mlir::Location loc, bool isOptional = false, Expression initValue = undefined)
        : VariableDeclarationDOM(name, type, loc, initValue), isOptional(isOptional)
    {
    }

    bool getIsOptional()
    {
        return isOptional;
    }

    /// LLVM style RTTI
    static bool classof(const BaseDOM *c)
    {
        return c->getKind() == Base_FunctionParam;
    }

  private:
    bool isOptional;
};

class FunctionPrototypeDOM
{
    std::string name;
    std::string nameWithoutNamespace;
    std::vector<FunctionParamDOM::TypePtr> args;
    mlir::Type returnType;
    bool discovered;
    bool hasCapturedVars;
    bool noBody;

  public:
    using TypePtr = std::shared_ptr<FunctionPrototypeDOM>;

    FunctionPrototypeDOM(StringRef name, std::vector<FunctionParamDOM::TypePtr> args)
        : name(name), args(args), returnType(), discovered(false), hasCapturedVars(false)
    {
    }

    StringRef getName() const
    {
        return name;
    }

    StringRef getNameWithoutNamespace() const
    {
        return nameWithoutNamespace;
    }
    void setNameWithoutNamespace(StringRef nameWithoutNamespace_)
    {
        nameWithoutNamespace = nameWithoutNamespace_;
    }

    // ArrayRef should not be "&" or "*"
    ArrayRef<FunctionParamDOM::TypePtr> getArgs() const
    {
        return args;
    }

    const mlir::Type &getReturnType() const
    {
        return returnType;
    }
    void setReturnType(mlir::Type returnType_)
    {
        returnType = returnType_;
    }

    bool getDiscovered() const
    {
        return discovered;
    }
    void setDiscovered(bool discovered_)
    {
        discovered = discovered_;
    }

    bool getHasCapturedVars() const
    {
        return hasCapturedVars;
    }
    void setHasCapturedVars(bool hasCapturedVars_)
    {
        hasCapturedVars = hasCapturedVars_;
    }

    bool getNoBody() const
    {
        return noBody;
    }
    void setNoBody(bool noBody_)
    {
        noBody = noBody_;
    }
};

} // namespace ts

#endif // MLIR_TYPESCRIPT_DOM_H_