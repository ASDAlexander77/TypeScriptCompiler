#ifndef MLIR_TYPESCRIPT_MLIRGENSTORE_H_
#define MLIR_TYPESCRIPT_MLIRGENSTORE_H_

#include "TypeScript/DOM.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir"

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

namespace
{

struct NamespaceInfo;
using NamespaceInfo_TypePtr = std::shared_ptr<NamespaceInfo>;

struct GenericFunctionInfo
{
  public:
    using TypePtr = std::shared_ptr<GenericFunctionInfo>;

    mlir::StringRef name;

    llvm::SmallVector<TypeParameterDOM::TypePtr> typeParams;

    FunctionLikeDeclarationBase functionDeclaration;

    FunctionPrototypeDOM::TypePtr funcOp;

    mlir_ts::FunctionType funcType;

    NamespaceInfo_TypePtr elementNamespace;

    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;

    bool processing;
    bool processed;

    GenericFunctionInfo() = default;
};

enum class VariableType
{
    Const,
    Let,
    Var,
    ConstRef,
    External
};

enum class VariableScope
{
    Local,
    Global
};

enum class Select: int
{
    NotSet = -1, 
    Any = 0, 	        //The linker may choose any COMDAT.
    ExactMatch = 1,     //The data referenced by the COMDAT must be the same.
    Largest = 2, 	    //The linker will choose the largest COMDAT.
    NoDeduplicate = 3,  //No deduplication is performed.
    SameSize = 4, 	    //The data referenced by the COMDAT must be the same size.
};

struct VariableClass
{
    VariableClass() : type{VariableType::Const}, isExport{false}, isImport{false}, isPublic{false}, isUsing{false}, isAppendingLinkage{false}, comdat{Select::NotSet}
    {
    }

    VariableClass(VariableType type_) : type{type_}, isExport{false}, isImport{false}, isPublic{false}, isUsing{false}, isAppendingLinkage{false}, comdat{Select::NotSet}
    {
    }

    VariableType type;
    bool isExport;
    bool isImport;
    bool isPublic;
    bool isUsing;
    bool isAppendingLinkage;
    Select comdat;

    inline VariableClass& operator=(VariableType type_) { type = type_; return *this; }

    inline bool operator==(VariableType type_) const { return type == type_; }
};

struct StaticFieldInfo
{
    mlir::Attribute id;
    mlir::Type type;
    mlir::StringRef globalVariableName;
    int virtualIndex;
};

struct MethodInfo
{
    // TODO: convert to attribute as fields 
    std::string name;
    mlir_ts::FunctionType funcType;
    // TODO: remove using it, we do not need it, we need actual name of function not function itself
    mlir_ts::FuncOp funcOp;
    bool isStatic;
    bool isVirtual;
    bool isAbstract;
    int virtualIndex;
    int orderWeight;
};

struct GenericMethodInfo
{
  public:
    std::string name;
    // TODO: review usage of funcType (it is inside FunctionPrototypeDOM already)
    mlir_ts::FunctionType funcType;
    FunctionPrototypeDOM::TypePtr funcOp;
    bool isStatic;
};

struct VirtualMethodOrInterfaceVTableInfo
{
    VirtualMethodOrInterfaceVTableInfo(MethodInfo methodInfo_, bool isInterfaceVTable_) : methodInfo(methodInfo_), isStaticField(false), isInterfaceVTable(isInterfaceVTable_)
    {
    }

    VirtualMethodOrInterfaceVTableInfo(StaticFieldInfo staticFieldInfo_, bool isInterfaceVTable_) : staticFieldInfo(staticFieldInfo_), isStaticField(true), isInterfaceVTable(isInterfaceVTable_)
    {
    }

    MethodInfo methodInfo;
    StaticFieldInfo staticFieldInfo;
    bool isStaticField;
    bool isInterfaceVTable;
};

struct AccessorInfo
{
    std::string name;
    mlir_ts::FuncOp get;
    mlir_ts::FuncOp set;
    bool isStatic;
    bool isVirtual;
    bool isAbstract;
};

struct InterfaceFieldInfo
{
    mlir::Attribute id;
    mlir::Type type;
    bool isConditional;
    int interfacePosIndex;
};

struct InterfaceMethodInfo
{
    std::string name;
    mlir_ts::FunctionType funcType;
    bool isConditional;
    int interfacePosIndex;
};

struct VirtualMethodOrFieldInfo
{
    VirtualMethodOrFieldInfo(MethodInfo methodInfo) : methodInfo(methodInfo), isField(false), isMissing(false)
    {
    }

    VirtualMethodOrFieldInfo(mlir_ts::FieldInfo fieldInfo) : fieldInfo(fieldInfo), isField(true), isMissing(false)
    {
    }

    VirtualMethodOrFieldInfo(MethodInfo methodInfo, bool isMissing)
        : methodInfo(methodInfo), fieldInfo(), isField(false), isMissing(isMissing)
    {
    }

    VirtualMethodOrFieldInfo(mlir_ts::FieldInfo fieldInfo, bool isMissing)
        : methodInfo(), fieldInfo(fieldInfo), isField(true), isMissing(isMissing)
    {
    }

    MethodInfo methodInfo;
    mlir_ts::FieldInfo fieldInfo;
    bool isField;
    bool isMissing;
};

struct InterfaceInfo
{
  public:
    using TypePtr = std::shared_ptr<InterfaceInfo>;
    using InterfaceInfoWithOffset = std::pair<int, InterfaceInfo::TypePtr>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir_ts::InterfaceType interfaceType;

    mlir_ts::InterfaceType originInterfaceType;
    
    llvm::SmallVector<InterfaceInfoWithOffset> extends;

    llvm::SmallVector<InterfaceFieldInfo> fields;

    llvm::SmallVector<InterfaceMethodInfo> methods;

    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;

    bool hasNew;

    InterfaceInfo() : hasNew(false)
    {
    }

    mlir::LogicalResult getTupleTypeFields(llvm::SmallVector<mlir_ts::FieldInfo> &tupleFields, mlir::MLIRContext *context)
    {
        for (auto &extent : extends)
        {
            if (mlir::failed(std::get<1>(extent)->getTupleTypeFields(tupleFields, context)))
            {
                return mlir::failure();
            }
        }

        for (auto &method : methods)
        {
            tupleFields.push_back({MLIRHelper::TupleFieldName(method.name, context), method.funcType, false});
        }

        for (auto &field : fields)
        {
            tupleFields.push_back({field.id, field.type, false});
        }

        return mlir::success();
    }

    mlir::LogicalResult getVirtualTable(
        llvm::SmallVector<VirtualMethodOrFieldInfo> &vtable,
        std::function<mlir_ts::FieldInfo(mlir::Attribute, mlir::Type, bool)> resolveField,
        std::function<MethodInfo &(std::string, mlir_ts::FunctionType, bool, int)> resolveMethod)
    {
        for (auto &extent : extends)
        {
            if (mlir::failed(std::get<1>(extent)->getVirtualTable(vtable, resolveField, resolveMethod)))
            {
                return mlir::failure();
            }
        }

        // do vtable for current
        for (auto &method : methods)
        {
            auto &classMethodInfo = resolveMethod(method.name, method.funcType, method.isConditional, method.interfacePosIndex);
            if (classMethodInfo.name.empty())
            {
                if (method.isConditional)
                {
                    MethodInfo missingMethod;
                    missingMethod.name = method.name;
                    missingMethod.funcType = method.funcType;
                    vtable.push_back({missingMethod, true});
                }
                else
                {
                    return mlir::failure();
                }
            }
            else
            {
                vtable.push_back({classMethodInfo});
            }
        }

        for (auto &field : fields)
        {
            auto fieldInfo = resolveField(field.id, field.type, field.isConditional);
            if (!fieldInfo.id)
            {
                if (field.isConditional)
                {
                    mlir_ts::FieldInfo missingField{field.id, field.type};
                    vtable.push_back({missingField, true});
                }
                else
                {
                    return mlir::failure();
                }
            }
            else
            {
                vtable.push_back({fieldInfo});
            }
        }

        return mlir::success();
    }

    int getMethodIndex(mlir::StringRef name)
    {
        auto dist = std::distance(
            methods.begin(), std::find_if(methods.begin(), methods.end(),
                                          [&](InterfaceMethodInfo methodInfo) { return name == methodInfo.name; }));
        return (signed)dist >= (signed)methods.size() ? -1 : dist;
    }

    int getFieldIndex(mlir::Attribute id)
    {
        auto dist = std::distance(fields.begin(),
                                  std::find_if(fields.begin(), fields.end(),
                                               [&](InterfaceFieldInfo fieldInfo) { return id == fieldInfo.id; }));
        return (signed)dist >= (signed)fields.size() ? -1 : dist;
    }

    InterfaceFieldInfo *findField(mlir::Attribute id, int &totalOffset)
    {
        auto index = getFieldIndex(id);
        if (index >= 0)
        {
            return &this->fields[index];
        }

        for (auto &extent : extends)
        {
            auto totalOffsetLocal = 0;
            auto field = std::get<1>(extent)->findField(id, totalOffsetLocal);
            if (field)
            {
                totalOffset = std::get<0>(extent) + totalOffsetLocal;
                return field;
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! can't resolve field: " << id << " in interface type: " << interfaceType
                                << "\n";);

        return nullptr;
    }

    InterfaceMethodInfo *findMethod(mlir::StringRef name, int &totalOffset)
    {
        auto index = getMethodIndex(name);
        if (index >= 0)
        {
            return &methods[index];
        }

        for (auto &extent : extends)
        {
            auto totalOffsetLocal = 0;
            auto *method = std::get<1>(extent)->findMethod(name, totalOffsetLocal);
            if (method)
            {
                totalOffset = std::get<0>(extent) + totalOffsetLocal;
                return method;
            }
        }

        return nullptr;
    }

    int getNextVTableMemberIndex()
    {
        return getVTableSize();
    }

    int getVTableSize()
    {
        auto offset = 0;
        for (auto &extent : extends)
        {
            offset += std::get<1>(extent)->getVTableSize();
        }

        return offset + fields.size() + methods.size();
    }

    void recalcOffsets()
    {
        auto offset = 0;
        for (auto &extent : extends)
        {
            std::get<0>(extent) = offset;
            offset += std::get<1>(extent)->getVTableSize();
        }
    }
};

struct GenericInterfaceInfo
{
  public:
    using TypePtr = std::shared_ptr<GenericInterfaceInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    llvm::SmallVector<TypeParameterDOM::TypePtr> typeParams;

    mlir_ts::InterfaceType interfaceType;

    InterfaceDeclaration interfaceDeclaration;

    NamespaceInfo_TypePtr elementNamespace;

    GenericInterfaceInfo()
    {
    }
};

struct ImplementInfo
{
    InterfaceInfo::TypePtr interface;
    int virtualIndex;
    bool processed;
};

enum class ProcessingStages : int {
    NotSet = 0,
    ErrorInStorageClass = 1,
    Processing = 2,
    ProcessingStorageClass = 3,
    ProcessedStorageClass = 4,
    ProcessingBody = 5,
    ProcessedBody = 6,
    Processed = 7,
};

struct ClassInfo
{
  public:
    using TypePtr = std::shared_ptr<ClassInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir_ts::ClassType classType;

    mlir_ts::ClassType originClassType;

    llvm::SmallVector<ClassInfo::TypePtr> baseClasses;

    llvm::SmallVector<ImplementInfo> implements;

    llvm::SmallVector<StaticFieldInfo> staticFields;

    llvm::SmallVector<MethodInfo> methods;

    llvm::SmallVector<GenericMethodInfo> staticGenericMethods;

    llvm::SmallVector<AccessorInfo> accessors;

    NodeArray<ClassElement> extraMembers;
    NodeArray<ClassElement> extraMembersPost;

    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;

    bool isDeclaration;
    bool hasNew;
    bool hasConstructor;
    bool hasInitializers;
    bool hasStaticConstructor;
    bool hasStaticInitializers;
    bool hasVirtualTable;
    bool isStatic;
    bool isAbstract;
    bool isExport;
    bool isImport;
    bool isPublic;
    bool isDynamicImport;
    bool hasRTTI;
    ProcessingStages processingAtEvaluation;
    ProcessingStages processing;

    ClassInfo()
        : isDeclaration(false), hasNew(false), hasConstructor(false), hasInitializers(false), hasStaticConstructor(false),
          hasStaticInitializers(false), hasVirtualTable(false), isAbstract(false), isExport(false), isImport(false), 
          isPublic(false), isDynamicImport(false), hasRTTI(false),
          processingAtEvaluation(ProcessingStages::NotSet), processing(ProcessingStages::NotSet)
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
        if (isStatic)
        {
            return false;
        }

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
        // in static class I don't want to have virtual table
        if (isStatic)
        {
            return;
        }

        auto processMethod = [] (auto &method, auto &vtable) {
            auto index =
                std::distance(vtable.begin(), std::find_if(vtable.begin(), vtable.end(), [&](auto vTableMethod) {
                                  return method.name == vTableMethod.methodInfo.name;
                              }));
            if ((size_t)index < vtable.size())
            {
                // found method
                vtable[index].methodInfo.funcOp = method.funcOp;
                method.virtualIndex = index;
                method.isVirtual = true;
                vtable[index].methodInfo.isAbstract = method.isAbstract;
            }
            else if (method.isVirtual)
            {
                method.virtualIndex = vtable.size();
                vtable.push_back({method, false});
            }
        };

        for (auto &base : baseClasses)
        {
            base->getVirtualTable(vtable);
        }
        
        // TODO: we need to process .Rtti first
        // TODO: then we need to process .instanceOf next

        // do vtable for current class
        for (auto &implement : implements)
        {
            auto index =
                std::distance(vtable.begin(), std::find_if(vtable.begin(), vtable.end(), [&](auto vTableRecord) {
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
        std::sort(methods.begin(), methods.end(), [&] (auto &method1, auto &method2) {
            return method1.orderWeight < method2.orderWeight;
        });

        for (auto &method : methods)
        {
#ifndef ADD_STATIC_MEMBERS_TO_VTABLE            
            if (method.isStatic)
            {
                continue;
            }
#endif            

            processMethod(method, vtable);
        }

#ifdef ADD_STATIC_MEMBERS_TO_VTABLE
        // static fields
        for (auto &staticField : staticFields)
        {
            staticField.virtualIndex = vtable.size();
            vtable.push_back({staticField, false});
        }        
#endif        
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
        auto dist =
            std::distance(staticFields.begin(), std::find_if(staticFields.begin(), staticFields.end(),
                                                             [&](auto fldInf) { return id == fldInf.id; }));
        return (signed)dist >= (signed)staticFields.size() ? -1 : dist;
    }

    int getMethodIndex(mlir::StringRef name)
    {
        auto dist = std::distance(
            methods.begin(), std::find_if(methods.begin(), methods.end(), [&](auto methodInfo) {
                return name == methodInfo.name;
            }));
        return (signed)dist >= (signed)methods.size() ? -1 : dist;
    }

    int getGenericMethodIndex(mlir::StringRef name)
    {
        auto dist = std::distance(
            staticGenericMethods.begin(), std::find_if(staticGenericMethods.begin(), staticGenericMethods.end(), [&](auto staticGenericMethodInfo) {
                return name == staticGenericMethodInfo.name;
            }));
        return (signed)dist >= (signed)staticGenericMethods.size() ? -1 : dist;
    }    

    unsigned fieldsCount()
    {
        auto storageClass = classType.getStorageType().cast<mlir_ts::ClassStorageType>();
        return storageClass.size();
    }

    mlir_ts::FieldInfo fieldInfoByIndex(int index)
    {
        if (index >= 0)
        {
            auto storageClass = classType.getStorageType().cast<mlir_ts::ClassStorageType>();
            return storageClass.getFieldInfo(index);
        }

        return mlir_ts::FieldInfo();
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

    MethodInfo *findMethod(mlir::StringRef name)
    {
        auto index = getMethodIndex(name);
        if (index >= 0)
        {
            return &methods[index];
        }

        for (auto &baseClass : baseClasses)
        {
            auto *method = baseClass->findMethod(name);
            if (method)
            {
                return method;
            }
        }

        return nullptr;
    }

    int getAccessorIndex(mlir::StringRef name)
    {
        auto dist = std::distance(accessors.begin(),
                                  std::find_if(accessors.begin(), accessors.end(),
                                               [&](AccessorInfo accessorInfo) { return name == accessorInfo.name; }));
        return (signed)dist >= (signed)accessors.size() ? -1 : dist;
    }

    int getImplementIndex(mlir::StringRef name)
    {
        auto dist = std::distance(implements.begin(),
                                  std::find_if(implements.begin(), implements.end(), [&](ImplementInfo implementInfo) {
                                      return name == implementInfo.interface->fullName;
                                  }));
        return (signed)dist >= (signed)implements.size() ? -1 : dist;
    }
};

struct GenericClassInfo
{
  public:
    using TypePtr = std::shared_ptr<GenericClassInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    llvm::SmallVector<TypeParameterDOM::TypePtr> typeParams;

    mlir_ts::ClassType classType;

    ClassLikeDeclaration classDeclaration;

    NamespaceInfo_TypePtr elementNamespace;

    GenericClassInfo()
    {
    }
};

struct NamespaceInfo
{
  public:
    using TypePtr = std::shared_ptr<NamespaceInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir_ts::NamespaceType namespaceType;

    llvm::StringMap<mlir_ts::FunctionType> functionTypeMap;

    llvm::StringMap<mlir_ts::FuncOp> functionMap;

    llvm::StringMap<GenericFunctionInfo::TypePtr> genericFunctionMap;

    llvm::StringMap<VariableDeclarationDOM::TypePtr> globalsMap;

    llvm::StringMap<llvm::StringMap<ts::VariableDeclarationDOM::TypePtr>> captureVarsMap;

    llvm::StringMap<llvm::SmallVector<mlir::typescript::FieldInfo>> localVarsInThisContextMap;

    llvm::StringMap<mlir::Type> typeAliasMap;

    llvm::StringMap<std::pair<llvm::SmallVector<TypeParameterDOM::TypePtr>, TypeNode>> genericTypeAliasMap;

    llvm::StringMap<mlir::StringRef> importEqualsMap;

    llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> enumsMap;

    llvm::StringMap<ClassInfo::TypePtr> classesMap;

    llvm::StringMap<GenericClassInfo::TypePtr> genericClassesMap;

    llvm::StringMap<InterfaceInfo::TypePtr> interfacesMap;

    llvm::StringMap<GenericInterfaceInfo::TypePtr> genericInterfacesMap;

    llvm::StringMap<NamespaceInfo::TypePtr> namespacesMap;

    bool isFunctionNamespace;

    NamespaceInfo::TypePtr parentNamespace;
};

} // namespace

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_MLIRGENSTORE_H_
