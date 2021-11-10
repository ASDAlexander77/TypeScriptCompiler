#ifndef MLIR_TYPESCRIPT_MLIRGENCONTEXT_H_
#define MLIR_TYPESCRIPT_MLIRGENCONTEXT_H_

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "TypeScript/DOM.h"

#include "parser_types.h"

#include <numeric>

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace
{

struct PassResult
{
    PassResult() : functionReturnTypeShouldBeProvided(false)
    {
    }

    mlir::Type functionReturnType;
    bool functionReturnTypeShouldBeProvided;
    llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> outerVariables;
    SmallVector<mlir_ts::FieldInfo> extraFieldsInThisContext;
};

struct GenContext
{
    GenContext() = default;

    // TODO: you are using "theModule.getBody()->clear();", do you need this hack anymore?
    void clean()
    {
        if (cleanUps)
        {
            for (auto op : *cleanUps)
            {
                op->dropAllDefinedValueUses();
                op->dropAllUses();
                op->dropAllReferences();
                op->erase();
            }

            delete cleanUps;
            cleanUps = nullptr;
        }

        if (passResult)
        {
            delete passResult;
            passResult = nullptr;
        }

        cleanState();
    }

    void cleanState()
    {
        if (state)
        {
            delete state;
            state = nullptr;
        }
    }

    void cleanUnresolved()
    {
        if (unresolved)
        {
            delete unresolved;
            unresolved = nullptr;
        }
    }

    bool allowPartialResolve;
    bool dummyRun;
    bool allowConstEval;
    bool allocateVarsInContextThis;
    bool allocateVarsOutsideOfOperation;
    bool skipProcessed;
    bool rediscover;
    mlir::Operation *currentOperation;
    mlir_ts::FuncOp funcOp;
    llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> *capturedVars;
    mlir::Type thisType;
    mlir::FunctionType destFuncType;
    mlir::Type argTypeDestFuncType;
    PassResult *passResult;
    mlir::SmallVector<mlir::Block *> *cleanUps;
    NodeArray<Statement> generatedStatements;
    mlir::SmallVector<std::pair<mlir::Location, std::string>> *unresolved;
    int *state;
};

enum class VariableClass
{
    Const,
    Let,
    Var,
    ConstRef
};

struct StaticFieldInfo
{
    mlir::Attribute id;
    mlir::StringRef globalVariableName;
};

struct MethodInfo
{
    std::string name;
    mlir_ts::FuncOp funcOp;
    bool isStatic;
    bool isVirtual;
    int virtualIndex;
};

struct VirtualMethodOrInterfaceVTableInfo
{
    VirtualMethodOrInterfaceVTableInfo() = default;

    MethodInfo methodInfo;
    bool isInterfaceVTable;
};

struct AccessorInfo
{
    std::string name;
    mlir_ts::FuncOp get;
    mlir_ts::FuncOp set;
    bool isStatic;
    bool isVirtual;
};

struct InterfaceFieldInfo
{
    mlir::Attribute id;
    mlir::Type type;
    int interfacePosIndex;
};

struct InterfaceMethodInfo
{
    std::string name;
    mlir::FunctionType funcType;
    int interfacePosIndex;
};

struct VirtualMethodOrFieldInfo
{
    VirtualMethodOrFieldInfo(MethodInfo methodInfo) : methodInfo(methodInfo), isField(false)
    {
    }

    VirtualMethodOrFieldInfo(mlir_ts::FieldInfo fieldInfo) : fieldInfo(fieldInfo), isField(true)
    {
    }

    MethodInfo methodInfo;
    mlir_ts::FieldInfo fieldInfo;
    bool isField;
};

struct InterfaceInfo
{
  public:
    using TypePtr = std::shared_ptr<InterfaceInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir_ts::InterfaceType interfaceType;

    llvm::SmallVector<InterfaceInfo::TypePtr> implements;

    llvm::SmallVector<InterfaceFieldInfo> fields;

    llvm::SmallVector<InterfaceMethodInfo> methods;

    InterfaceInfo()
    {
    }

    mlir::LogicalResult getVirtualTable(llvm::SmallVector<VirtualMethodOrFieldInfo> &vtable,
                                        std::function<mlir_ts::FieldInfo(mlir::Attribute, mlir::Type)> resolveField,
                                        std::function<MethodInfo &(std::string, mlir::FunctionType)> resolveMethod)
    {
        // do vtable for current
        for (auto &method : methods)
        {
            auto &classMethodInfo = resolveMethod(method.name, method.funcType);
            if (classMethodInfo.name.empty())
            {
                return mlir::failure();
            }

            vtable.push_back({classMethodInfo});
        }

        for (auto &field : fields)
        {
            auto fieldInfo = resolveField(field.id, field.type);
            if (!fieldInfo.id)
            {
                return mlir::failure();
            }

            vtable.push_back({fieldInfo});
        }

        return mlir::success();
    }

    int getMethodIndex(mlir::StringRef name)
    {
        auto dist = std::distance(methods.begin(), std::find_if(methods.begin(), methods.end(),
                                                                [&](InterfaceMethodInfo methodInfo) { return name == methodInfo.name; }));
        return (signed)dist >= (signed)methods.size() ? -1 : dist;
    }

    int getFieldIndex(mlir::Attribute id)
    {
        auto dist = std::distance(
            fields.begin(), std::find_if(fields.begin(), fields.end(), [&](InterfaceFieldInfo fieldInfo) { return id == fieldInfo.id; }));
        return (signed)dist >= (signed)fields.size() ? -1 : dist;
    }

    int getNextVTableMemberIndex()
    {
        return fields.size() + methods.size();
    }
};

struct ImplementInfo
{
    InterfaceInfo::TypePtr interface;
    int virtualIndex;
    bool processed;
};

struct ClassInfo
{
  public:
    using TypePtr = std::shared_ptr<ClassInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir_ts::ClassType classType;

    llvm::SmallVector<ClassInfo::TypePtr> baseClasses;

    llvm::SmallVector<ImplementInfo> implements;

    llvm::SmallVector<StaticFieldInfo> staticFields;

    llvm::SmallVector<MethodInfo> methods;

    llvm::SmallVector<AccessorInfo> accessors;

    bool hasConstructor;
    bool hasInitializers;
    bool hasStaticConstructor;
    bool hasStaticInitializers;
    bool hasVirtualTable;
    bool isAbstract;
    bool hasRTTI;

    ClassInfo()
        : hasConstructor(false), hasInitializers(false), hasStaticConstructor(false), hasStaticInitializers(false), hasVirtualTable(false),
          isAbstract(false), hasRTTI(false)
    {
    }

    auto getHasConstructor() -> bool
    {
        if (hasConstructor)
        {
            return true;
        }

        for (auto &base : baseClasses)
        {
            if (base->hasConstructor)
            {
                return true;
            }
        }

        return false;
    }

    auto getHasVirtualTable() -> bool
    {
        if (hasVirtualTable)
        {
            return true;
        }

        for (auto &base : baseClasses)
        {
            if (base->hasVirtualTable)
            {
                return true;
            }
        }

        return false;
    }

    auto getHasVirtualTableVariable() -> bool
    {
        for (auto &base : baseClasses)
        {
            if (base->hasVirtualTable)
            {
                return false;
            }
        }

        if (hasVirtualTable)
        {
            return true;
        }

        return false;
    }

    void getVirtualTable(llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> &vtable)
    {
        for (auto &base : baseClasses)
        {
            base->getVirtualTable(vtable);
        }

        // do vtable for current class
        for (auto &implement : implements)
        {
            auto index = std::distance(vtable.begin(), std::find_if(vtable.begin(), vtable.end(), [&](auto vTableRecord) {
                                           return implement.interface->fullName == vTableRecord.methodInfo.name;
                                       }));
            if ((size_t)index < vtable.size())
            {
                // found interface
                continue;
            }

            MethodInfo methodInfo;
            methodInfo.name = implement.interface->fullName.str();
            implement.virtualIndex = vtable.size();
            vtable.push_back({methodInfo, true});
        }

        // methods
        for (auto &method : methods)
        {
            auto index = std::distance(vtable.begin(), std::find_if(vtable.begin(), vtable.end(), [&](auto vTableMethod) {
                                           return method.name == vTableMethod.methodInfo.name;
                                       }));
            if ((size_t)index < vtable.size())
            {
                // found method
                vtable[index].methodInfo.funcOp = method.funcOp;
                method.virtualIndex = index;
                method.isVirtual = true;
                continue;
            }

            if (method.isVirtual)
            {
                method.virtualIndex = vtable.size();
                vtable.push_back({method, false});
            }
        }
    }

    auto getBasesWithRoot(SmallVector<StringRef> &classNames) -> bool
    {
        classNames.push_back(fullName);

        for (auto &base : baseClasses)
        {
            base->getBasesWithRoot(classNames);
        }

        return true;
    }

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;

    int getStaticFieldIndex(mlir::Attribute id)
    {
        auto dist = std::distance(staticFields.begin(), std::find_if(staticFields.begin(), staticFields.end(),
                                                                     [&](StaticFieldInfo fldInf) { return id == fldInf.id; }));
        return (signed)dist >= (signed)staticFields.size() ? -1 : dist;
    }

    int getMethodIndex(mlir::StringRef name)
    {
        auto dist = std::distance(methods.begin(), std::find_if(methods.begin(), methods.end(), [&](MethodInfo methodInfo) {
                                      LLVM_DEBUG(dbgs() << "\nmatching method: " << name << " to " << methodInfo.name << "\n\n";);
                                      return name == methodInfo.name;
                                  }));
        return (signed)dist >= (signed)methods.size() ? -1 : dist;
    }

    mlir_ts::FieldInfo findField(mlir::Attribute id, bool &foundField)
    {
        foundField = false;
        auto storageClass = classType.getStorageType().cast<mlir_ts::ClassStorageType>();
        auto index = storageClass.getIndex(id);
        if (index >= 0)
        {
            foundField = true;
            return storageClass.getFieldInfo(index);
        }

        for (auto &baseClass : baseClasses)
        {
            auto field = baseClass->findField(id, foundField);
            if (foundField)
            {
                return field;
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! can't resolve field: " << id << " in class type: " << storageClass << "\n";);

        return mlir_ts::FieldInfo();
    }

    MethodInfo *findMethod(mlir::StringRef name, bool &foundMethod)
    {
        foundMethod = false;
        auto index = getMethodIndex(name);
        if (index >= 0)
        {
            foundMethod = true;
            return &methods[index];
        }

        for (auto &baseClass : baseClasses)
        {
            auto *method = baseClass->findMethod(name, foundMethod);
            if (foundMethod)
            {
                return method;
            }
        }

        return nullptr;
    }

    int getAccessorIndex(mlir::StringRef name)
    {
        auto dist = std::distance(accessors.begin(), std::find_if(accessors.begin(), accessors.end(),
                                                                  [&](AccessorInfo accessorInfo) { return name == accessorInfo.name; }));
        return (signed)dist >= (signed)accessors.size() ? -1 : dist;
    }

    int getImplementIndex(mlir::StringRef name)
    {
        auto dist = std::distance(implements.begin(), std::find_if(implements.begin(), implements.end(), [&](ImplementInfo implementInfo) {
                                      return name == implementInfo.interface->fullName;
                                  }));
        return (signed)dist >= (signed)implements.size() ? -1 : dist;
    }
};

struct NamespaceInfo
{
  public:
    using TypePtr = std::shared_ptr<NamespaceInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir_ts::NamespaceType namespaceType;

    llvm::StringMap<mlir_ts::FuncOp> functionMap;

    llvm::StringMap<VariableDeclarationDOM::TypePtr> globalsMap;

    llvm::StringMap<llvm::StringMap<ts::VariableDeclarationDOM::TypePtr>> captureVarsMap;

    llvm::StringMap<llvm::SmallVector<mlir::typescript::FieldInfo>> localVarsInThisContextMap;

    llvm::StringMap<mlir::Type> typeAliasMap;

    llvm::StringMap<mlir::StringRef> importEqualsMap;

    llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> enumsMap;

    llvm::StringMap<ClassInfo::TypePtr> classesMap;

    llvm::StringMap<InterfaceInfo::TypePtr> interfacesMap;

    llvm::StringMap<NamespaceInfo::TypePtr> namespacesMap;
};

} // namespace

#endif // MLIR_TYPESCRIPT_MLIRGENCONTEXT_H_
