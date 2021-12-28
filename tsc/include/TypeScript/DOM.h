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
        Base_TypeParameter,
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
    bool captured;

  public:
    using TypePtr = std::shared_ptr<VariableDeclarationDOM>;

    VariableDeclarationDOM(StringRef name, mlir::Type type, mlir::Location loc, Expression initValue = undefined)
        : BaseDOM(Base_VariableDeclaration), name(name), type(type), loc(loc), initValue(initValue), captured(false), readWrite(false)
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
    void setType(mlir::Type type_)
    {
        type = type_;
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
    bool getCaptured() const
    {
        return captured;
    };
    void setCaptured(bool value = true)
    {
        captured = value;
    };

  protected:
    bool readWrite;
};

class FunctionParamDOM : public VariableDeclarationDOM
{
  public:
    using TypePtr = std::shared_ptr<FunctionParamDOM>;

    FunctionParamDOM(StringRef name, mlir::Type type, mlir::Location loc, bool isOptional = false, bool isMultiArgs = false,
                     Expression initValue = undefined, Node bindingPattern = undefined)
        : VariableDeclarationDOM(name, type, loc, initValue), isOptional(isOptional), isMultiArgs(isMultiArgs),
          bindingPattern(bindingPattern)
    {
    }

    bool getIsOptional()
    {
        return isOptional;
    }

    bool getIsMultiArgs()
    {
        return isMultiArgs;
    }

    Node getBindingPattern()
    {
        return bindingPattern;
    }

    /// LLVM style RTTI
    static bool classof(const BaseDOM *c)
    {
        return c->getKind() == Base_FunctionParam;
    }

  private:
    bool isOptional;
    bool isMultiArgs;
    Node bindingPattern;
};

class FunctionPrototypeDOM
{
    std::string name;
    std::string nameWithoutNamespace;
    std::vector<FunctionParamDOM::TypePtr> args;
    mlir::Type returnType;
    bool discovered;
    bool hasCapturedVars;
    bool hasExtraFields;
    bool noBody;

  public:
    using TypePtr = std::shared_ptr<FunctionPrototypeDOM>;

    FunctionPrototypeDOM(StringRef name, std::vector<FunctionParamDOM::TypePtr> args)
        : name(name.str()), args(args), returnType(), discovered(false), hasCapturedVars(false), hasExtraFields(false), noBody(false)
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
        nameWithoutNamespace = nameWithoutNamespace_.str();
    }

    // ArrayRef should not be "&" or "*"
    ArrayRef<FunctionParamDOM::TypePtr> getArgs() const
    {
        return args;
    }

    bool isMultiArgs()
    {
        if (args.size() == 0)
        {
            return false;
        }

        return args.back()->getIsMultiArgs();
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

    bool getHasExtraFields() const
    {
        return hasExtraFields;
    }
    void setHasExtraFields(bool hasExtraFields_)
    {
        hasExtraFields = hasExtraFields_;
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

class TypeParameterDOM : public BaseDOM
{
    std::string name;
    mlir::Type constraint;
    mlir::Type default;

  public:
    using TypePtr = std::shared_ptr<TypeParameterDOM>;

    TypeParameterDOM(std::string name) : BaseDOM(Base_TypeParameter), name(name)
    {
    }

    StringRef getName() const
    {
        return name;
    }

    void setConstraint(mlir::Type constraint_)
    {
        constraint = constraint_;
    }

    mlir::Type getConstraint()
    {
        return constraint;
    }

    void setDefault(mlir::Type default_)
    {
        default = default_;
    }

    mlir::Type getDefault()
    {
        return default;
    }
};

} // namespace ts

#endif // MLIR_TYPESCRIPT_DOM_H_