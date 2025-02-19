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
    bool ignoreCapturing;
    bool _using;

  public:
    using TypePtr = std::shared_ptr<VariableDeclarationDOM>;

    VariableDeclarationDOM(StringRef name, mlir::Type type, mlir::Location loc, Expression initValue = undefined)
        : BaseDOM(Base_VariableDeclaration), name(name), type(type), loc(loc), initValue(initValue), captured(false), ignoreCapturing(false), _using(false), readWrite(false)
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
    bool getCaptured() const
    {
        return captured;
    };
    void setCaptured(bool value = true)
    {
        captured = value;
    };
    bool getIgnoreCapturing()
    {
        return ignoreCapturing;
    };
    void setIgnoreCapturing(bool value = true)
    {
        ignoreCapturing = value;
    };
    bool getUsing() const
    {
        return _using;
    };
    void setUsing(bool value = true)
    {
        _using = value;
    };

  protected:
    bool readWrite;
};

class FunctionParamDOM : public VariableDeclarationDOM
{
  public:
    using TypePtr = std::shared_ptr<FunctionParamDOM>;

    FunctionParamDOM(StringRef name, mlir::Type type, mlir::Location loc, bool isOptional = false, bool isMultiArgsParam = false,
                     Expression initValue = undefined, Node bindingPattern = undefined)
        : VariableDeclarationDOM(name, type, loc, initValue), processed(false), isOptional(isOptional), isMultiArgsParam(isMultiArgsParam),
          bindingPattern(bindingPattern)
    {
    }

    bool getIsOptional()
    {
        return isOptional;
    }

    bool getIsMultiArgsParam()
    {
        return isMultiArgsParam;
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

    bool processed;

  private:
    bool isOptional;
    bool isMultiArgsParam;
    Node bindingPattern;
};

class FunctionPrototypeDOM
{
    std::string name;
    std::string nameWithoutNamespace;
    std::vector<FunctionParamDOM::TypePtr> params;
    mlir_ts::FunctionType funcType;
    mlir::Type returnType;
    bool discovered;
    bool hasCapturedVars;
    bool hasExtraFields;
    bool isGenericFunction;
    bool noBody;

  public:
    using TypePtr = std::shared_ptr<FunctionPrototypeDOM>;

    FunctionPrototypeDOM(StringRef name, std::vector<FunctionParamDOM::TypePtr> params)
        : name(name.str()), params(params), funcType(), returnType(), discovered(false), hasCapturedVars(false), hasExtraFields(false), isGenericFunction(false), noBody(false)
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
    ArrayRef<FunctionParamDOM::TypePtr> getParams() const
    {
        return params;
    }

    bool isMultiArgs()
    {
        return params.size() > 0 && params.back()->getIsMultiArgsParam();
    }

    const mlir_ts::FunctionType &getFuncType() const
    {
        return funcType;
    }
    void setFuncType(mlir_ts::FunctionType funcType_)
    {
        funcType = funcType_;
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

    bool getIsGeneric() const
    {
        return isGenericFunction;
    }
    void setIsGeneric(bool isGenericFunction_)
    {
        isGenericFunction = isGenericFunction_;
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
    TypeNode constraint;
    TypeNode _default;
    bool _hasConstraint;
    bool _hasDefault;

  public:
    using TypePtr = std::shared_ptr<TypeParameterDOM>;

    TypeParameterDOM(std::string name) : BaseDOM(Base_TypeParameter), name(name), _hasConstraint(false), _hasDefault(false)
    {
    }

    StringRef getName() const
    {
        return name;
    }

    void setConstraint(TypeNode constraint_)
    {
        _hasConstraint = constraint_;
        constraint = constraint_;
    }

    TypeNode getConstraint()
    {
        return constraint;
    }

    bool hasConstraint()
    {
        return _hasConstraint;
    }    

    void setDefault(TypeNode default_)
    {
        _hasDefault = default_;
        _default = default_;
    }

    TypeNode getDefault()
    {
        return _default;
    }

    bool hasDefault()
    {
        return _hasDefault;
    }    
};

} // namespace ts

#endif // MLIR_TYPESCRIPT_DOM_H_