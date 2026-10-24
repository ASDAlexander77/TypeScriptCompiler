#ifndef MLIR_TYPESCRIPT_MLIRGENSTORE_H_
#define MLIR_TYPESCRIPT_MLIRGENSTORE_H_

#include "TypeScript/DOM.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir"

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

struct NamespaceInfo;
using NamespaceInfo_TypePtr = std::shared_ptr<NamespaceInfo>;

struct GenericFunctionInfo
{
  public:
    using TypePtr = std::shared_ptr<GenericFunctionInfo>;

    mlir::StringRef name;

    NamespaceInfo_TypePtr elementNamespace;

    llvm::SmallVector<TypeParameterDOM::TypePtr> typeParams;

    FunctionLikeDeclarationBase functionDeclaration;

    FunctionPrototypeDOM::TypePtr funcOp;

    mlir_ts::FunctionType funcType;

    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;

    mlir::Type thisType;
    mlir_ts::ClassType thisClassType;

    SourceFile sourceFile;
    StringRef fileName;

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
    VariableClass() : type{VariableType::Const}, isExport{false}, isImport{false}, isDynamicImport{false}, isPublic{false}, isUsing{false}, isAppendingLinkage{false}, comdat{Select::NotSet}, isUsed{false}, atomic{false}, ordering{0}, syncscope{StringRef()}, isVolatile{false}, nonTemporal{false}, invariant{false}, isBoxed{false}
    {
    }

    VariableClass(VariableType type_) : type{type_}, isExport{false}, isImport{false}, isDynamicImport{false}, isPublic{false}, isUsing{false}, isAppendingLinkage{false}, comdat{Select::NotSet}, isUsed{false}, atomic{false}, ordering{0}, syncscope{StringRef()}, isVolatile{false}, nonTemporal{false}, invariant{false}, isBoxed{false}
    {
    }

    VariableType type;
    bool isExport;
    bool isImport;
    bool isDynamicImport;
    bool isPublic;
    bool isUsing;
    bool isAppendingLinkage;
    Select comdat;
    bool isUsed;
    bool atomic; // atomic
    int ordering; // atomic ordering
    StringRef syncscope; // atomic syncscope
    bool isVolatile;
    bool nonTemporal;
    bool invariant;
    // @dllimport declaration-only marker: the exported symbol's storage is a
    // single boxed pointer to the real data, not the data inline - see
    // DeclarationPrinter.cpp's printVariableDeclaration and
    // MLIRGenVariables.cpp's isDynamicImport load branch.
    bool isBoxed;
    // @dllname: exported symbol name, empty when the variable keeps its own name
    StringRef dllName;

    inline VariableClass& operator=(VariableType type_) { type = type_; return *this; }

    inline bool operator==(VariableType type_) const { return type == type_; }
};

struct StaticFieldInfo
{
    mlir::Attribute id;
    mlir::Type type;
    mlir::StringRef globalVariableName;
    int virtualIndex;
    mlir_ts::AccessLevel accessLevel;
};

// what we know about a registered function; deliberately not the FuncOp itself -
// discovery-pass ops are erased with the discovery module, so cached op handles dangle.
// Resolve a live op through theModule.lookupSymbol when one is actually needed.
struct FunctionEntry
{
    std::string name;
    mlir_ts::FunctionType funcType;
    // @dllname: exported symbol name, empty when the function keeps name
    std::string dllName;

    explicit operator bool() const
    {
        return static_cast<bool>(funcType);
    }
};

struct MethodInfo
{
    // TODO: convert to attribute as fields
    std::string name;
    mlir_ts::FunctionType funcType;
    // symbol name of the function implementing the method (see FunctionEntry note)
    std::string funcName;
    bool isStatic;
    bool isVirtual;
    bool isAbstract;
    int virtualIndex;
    int orderWeight;
    mlir_ts::AccessLevel accessLevel;
    // @dllname: exported symbol name, empty when the method keeps funcName
    std::string dllName;
    // A @cpp method returning a C++ record: the type it is declared with (funcType is the
    // one its slot is called with, the hidden result pointer added, see
    // mlirGenClassCppVTableHiddenReturn); null for any other method.
    mlir_ts::FunctionType declaredFuncType;
    // @cppnontrivial: the C++ record a @cpp method returns has a destructor or a copy
    // constructor, which decides how the Itanium ABI returns it
    bool cppNonTrivialReturn = false;
    // @slot(N): the vtable slot of a @cpp method, -1 without @slot (a non-virtual method, or
    // one overriding a base class's slot by name)
    int cppSlot = -1;
    // a slot of a @cpp class's vtable that no method names (ClassInfo::getVirtualTable)
    bool isCppSlotGap = false;
    // `@keeps`, `@borrows`, `@lends` (OWN_DECLARED_KEEPS_ATTR_NAME): argument indices, `this` 0, for
    // the declaration an importer re-parses; `@lends` names one, or several for a borrowing record
    // (own spec 32), as a mask of argument indices (a vector would push the vtable's SmallVector
    // past its inline-size limit)
    std::vector<int32_t> ownKeeps;
    std::vector<int32_t> ownBorrows;
    uint64_t ownLendsMask = 0;
    // `@record` (OWN_DECLARED_RECORD_ATTR_NAME): what `@lends` names, the result is a record of
    bool ownRecord = false;
};

struct GenericMethodInfo
{
    std::string name;
    // TODO: review usage of funcType (it is inside FunctionPrototypeDOM already)
    mlir_ts::FunctionType funcType;
    FunctionPrototypeDOM::TypePtr funcProto;
    bool isStatic;
    mlir_ts::AccessLevel accessLevel;
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
    FunctionEntry get;
    FunctionEntry set;
    bool isStatic;
    bool isVirtual;
    bool isAbstract;
    mlir_ts::AccessLevel getAccessLevel;
    mlir_ts::AccessLevel setAccessLevel;
};

struct IndexInfo
{
    mlir_ts::FunctionType indexSignature;
    FunctionEntry get;
    FunctionEntry set;
    mlir_ts::AccessLevel getAccessLevel;
    mlir_ts::AccessLevel setAccessLevel;
};

struct InterfaceFieldInfo
{
    mlir::Attribute id;
    mlir::Type type;
    bool isConditional;
    int interfacePosIndex;
    int virtualIndex;
};

struct InterfaceMethodInfo
{
    std::string name;
    mlir_ts::FunctionType funcType;
    bool isConditional;
    int interfacePosIndex;
    int virtualIndex;
};

struct InterfaceAccessorInfo
{
    mlir::Type type;
    std::string name;
    std::string getMethod;
    std::string setMethod;
};

struct InterfaceIndexInfo
{
    mlir_ts::FunctionType indexSignature;
    std::string getMethod;
    std::string setMethod;
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

    NamespaceInfo_TypePtr elementNamespace;

    mlir_ts::InterfaceType interfaceType;

    mlir_ts::InterfaceType originInterfaceType;
    
    llvm::SmallVector<InterfaceInfoWithOffset> extends;

    llvm::SmallVector<InterfaceFieldInfo> fields;

    llvm::SmallVector<InterfaceMethodInfo> methods;

    llvm::SmallVector<InterfaceAccessorInfo> accessors;

    llvm::SmallVector<InterfaceIndexInfo> indexes;

    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;

    bool hasNew;

    // `@value`: its values are copied, not shared (see ClassInfo::isValue)
    bool isValue;

    InterfaceInfo() : hasNew(false), isValue(false)
    {
    }

    // own methods plus every inherited method from `extends`, in the same
    // extends-first-then-own order as getTupleTypeFields/getVirtualTable -
    // paired with the combined vtable slot each one actually occupies IN THIS
    // (possibly multi-level/diamond `extends`) interface, since
    // method.virtualIndex is only correct standalone (see findField's doc
    // comment above); a method inherited through `extends`
    // needs vtableOffset (the declaring interface's slot-block position) added to
    // it. Needed by mlirGenCreateInterfaceVTableForObject's per-object patch loop,
    // which indexes directly into ONE combined vtable for the cast's root
    // interface - using method.virtualIndex alone there mis-patches every
    // inherited method beyond the first `extends` target (harmless no-op for a
    // single `extends` chain, where each level's own slots start where the
    // previous level's ended and vtableOffset is cumulative from index 0 - but
    // wrong for a diamond, where a second/third `extends` target's methods all
    // stack at vtableOffset 0 relative to themselves).
    void getAllMethodsWithVTableOffset(llvm::SmallVector<std::pair<InterfaceMethodInfo *, int>> &allMethods, int baseOffset = 0)
    {
        auto offset = baseOffset;
        for (auto &extent : extends)
        {
            std::get<1>(extent)->getAllMethodsWithVTableOffset(allMethods, offset);
            offset += std::get<1>(extent)->getVTableSize();
        }

        // own methods' virtualIndex (assignCanonicalVirtualIndexes()) is already
        // absolute WITHIN THIS interface's own combined vtable (it starts counting
        // after this interface's own extends block) - so at THIS level baseOffset
        // is 0 and no further adjustment is needed. It's only when THIS interface
        // is itself reached as an `extends` target (the recursive call above,
        // baseOffset != 0) that its methods - own AND inherited alike, all already
        // pushed into `allMethods` by this point - need the caller's baseOffset,
        // which the recursive call already applied for inherited ones; own ones
        // get it here via baseOffset.
        for (auto &method : methods)
        {
            allMethods.push_back({&method, baseOffset});
        }
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
            tupleFields.push_back(
            {
                MLIRHelper::TupleFieldName(method.name, context), 
                method.funcType, 
                method.isConditional, 
                mlir_ts::AccessLevel::Public
            });
        }

        for (auto &field : fields)
        {
            tupleFields.push_back({field.id, field.type, field.isConditional, mlir_ts::AccessLevel::Public});
        }

        return mlir::success();
    }

    // builds the vtable CONTENTS (constant function-pointer symbols, GEP-on-null field
    // offsets, or an explicit -1 sentinel for an optional member the specific cast target
    // doesn't provide) for ONE implementer of this interface. Does NOT assign virtualIndex -
    // slot numbering is a pure function of the interface's own declaration
    // (assignCanonicalVirtualIndexes() is the sole writer) and must stay identical across
    // every implementer, including ones this method is never called for (e.g. a module that
    // only reads an already-typed interface value it imported never calls this at all - see
    // docs/interface-vtable-simplification-design.md §4). Earlier versions of this method
    // wrote virtualIndex here as a side effect of resolving ONE implementer, which corrupted
    // the shared InterfaceInfo for every OTHER implementer's already-compiled or
    // yet-to-compile field/method access sites whenever presence of an optional member
    // differed between casts (§5) - do not reintroduce that.
    mlir::LogicalResult getVirtualTable(
        llvm::SmallVector<VirtualMethodOrFieldInfo> &vtable,
        std::function<std::pair<mlir_ts::FieldInfo, mlir::LogicalResult>(mlir::Attribute, mlir::Type, bool)> resolveField,
        std::function<std::pair<MethodInfo &, mlir::LogicalResult>(std::string, mlir_ts::FunctionType, bool, int)> resolveMethod,
        bool methodsAsFields = false)
    {
        for (auto &extent : extends)
        {
            if (mlir::failed(std::get<1>(extent)->getVirtualTable(vtable, resolveField, resolveMethod, methodsAsFields)))
            {
                return mlir::failure();
            }
        }

        // do vtable for current
        for (auto &method : methods)
        {
            if (methodsAsFields)
            {
                auto methodNameAttr = mlir::StringAttr::get(method.funcType.getContext(), method.name);
                auto [fieldInfo, result] = resolveField(methodNameAttr, method.funcType, method.isConditional);
                if (mlir::failed(result))
                {
                    return mlir::failure();
                }

                if (!fieldInfo.id)
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
                    vtable.push_back({fieldInfo});
                }
            }
            else
            {
                auto [classMethodInfo, result] = resolveMethod(method.name, method.funcType, method.isConditional, method.interfacePosIndex);
                if (mlir::failed(result))
                {
                    return mlir::failure();
                }

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
        }

        for (auto &field : fields)
        {
            auto [fieldInfo, result] = resolveField(field.id, field.type, field.isConditional);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            if (!fieldInfo.id)
            {
                if (field.isConditional)
                {
                    mlir_ts::FieldInfo missingField{field.id, field.type, false, mlir_ts::AccessLevel::Public};
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

    int getAccessorIndex(mlir::StringRef name)
    {
        auto dist = std::distance(
            accessors.begin(), std::find_if(accessors.begin(), accessors.end(),
                                          [&](InterfaceAccessorInfo accessorInfo) { return name == accessorInfo.name; }));
        return (signed)dist >= (signed)accessors.size() ? -1 : dist;
    }

    // vtableOffset accumulates the position, within the vtable of the interface this call
    // started on (the "root" - what the access site's InterfaceType actually is), where the
    // DECLARING interface's own slots begin. A field/method's own virtualIndex
    // (assignCanonicalVirtualIndexes()) is only correct standalone - t2 extends F1, F2
    // means F2's fields sit at vtableOffset = F1's slot count within t2's combined vtable,
    // not at F2's own standalone index 0. The access site must add the two together
    // (see InterfaceFieldAccess/InterfaceMethodAccess). Earlier code baked this offset
    // directly into the shared InterfaceFieldInfo/InterfaceMethodInfo via a getVirtualTable()
    // side effect keyed to whichever cast ran most recently - correct only by accident for
    // whichever root interface was cast last; see docs/interface-vtable-simplification-design.md.
    InterfaceFieldInfo *findField(mlir::Attribute id, int &vtableOffset)
    {
        vtableOffset = 0;

        auto index = getFieldIndex(id);
        if (index >= 0)
        {
            return &this->fields[index];
        }

        for (auto &extent : extends)
        {
            if (auto *field = std::get<1>(extent)->findField(id, vtableOffset))
            {
                vtableOffset += std::get<0>(extent);
                return field;
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! can't resolve field: " << id << " in interface type: " << interfaceType
                                << "\n";);

        return nullptr;
    }

    InterfaceFieldInfo *findField(mlir::Attribute id)
    {
        int vtableOffset;
        return findField(id, vtableOffset);
    }

    InterfaceMethodInfo *findMethod(mlir::StringRef name, int &vtableOffset)
    {
        vtableOffset = 0;

        auto index = getMethodIndex(name);
        if (index >= 0)
        {
            return &methods[index];
        }

        for (auto &extent : extends)
        {
            if (auto *method = std::get<1>(extent)->findMethod(name, vtableOffset))
            {
                vtableOffset += std::get<0>(extent);
                return method;
            }
        }

        return nullptr;
    }

    InterfaceMethodInfo *findMethod(mlir::StringRef name)
    {
        int vtableOffset;
        return findMethod(name, vtableOffset);
    }

    InterfaceAccessorInfo *findAccessor(mlir::StringRef name)
    {
        auto index = getAccessorIndex(name);
        if (index >= 0)
        {
            return &accessors[index];
        }

        for (auto &extent : extends)
        {
            if (auto *accessor = std::get<1>(extent)->findAccessor(name))
            {
                return accessor;
            }
        }

        return nullptr;
    }

    InterfaceIndexInfo *findIndexer()
    {
        if (indexes.size() > 0)
        {
            return &indexes[0];
        }

        for (auto &extent : extends)
        {
            if (auto *indexer = std::get<1>(extent)->findIndexer())
            {
                return indexer;
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

        // as I remember methods are first in interfaces
        return offset + methods.size() + fields.size();
    }

    // vtable slot numbers must be a pure function of the interface's own declaration
    // (extends, then own methods in order, then own fields in order) - NOT of whichever
    // object happens to be cast to it first. getVirtualTable() re-derives the same
    // methods-then-fields order per-cast (needed to mark per-object optional members
    // missing), but a module that only reads an already-typed interface value - without
    // ever casting an object to it itself (e.g. importing an already-boxed global from
    // another compilation unit) - never runs getVirtualTable() at all. Without this
    // eagerly-computed, cast-independent pass, such a module falls back on the
    // interleaved declaration-order index assigned at member-registration time
    // (mlirGenInterfaceAddFieldMember / addInterfaceMethod), which disagrees with the
    // methods-first layout whenever a field is declared before a method in source order -
    // reading through the wrong vtable slot (e.g. a method's function pointer
    // reinterpreted as a field offset) and crashing.
    void assignCanonicalVirtualIndexes()
    {
        auto offset = 0;
        for (auto &extent : extends)
        {
            offset += std::get<1>(extent)->getVTableSize();
        }

        for (auto &method : methods)
        {
            method.virtualIndex = offset++;
        }

        for (auto &field : fields)
        {
            field.virtualIndex = offset++;
        }
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

    NamespaceInfo_TypePtr elementNamespace;

    llvm::SmallVector<TypeParameterDOM::TypePtr> typeParams;

    mlir_ts::InterfaceType interfaceType;

    InterfaceDeclaration interfaceDeclaration;

    SourceFile sourceFile;
    StringRef fileName;

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
    ErrorInMembers = 2,
    ErrorInBaseInterfaces = 3,
    ErrorInHeritageClauseImplements = 4,
    ErrorInVTable = 5,
    Processing = 6,
    ProcessingStorageClass = 7,
    ProcessedStorageClass = 8,
    ProcessingBody = 9,
    ProcessedBody = 10,
    Processed = 11,
};

struct EnumInfo
{
  public:
    using TypePtr = std::shared_ptr<EnumInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    NamespaceInfo_TypePtr elementNamespace;

    mlir_ts::EnumType enumType;

    EnumInfo() = default;
};

struct ClassInfo
{
  public:
    using TypePtr = std::shared_ptr<ClassInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    NamespaceInfo_TypePtr elementNamespace;

    mlir_ts::ClassType classType;

    mlir_ts::ClassType originClassType;

    llvm::SmallVector<ClassInfo::TypePtr> baseClasses;

    llvm::SmallVector<ImplementInfo> implements;

    llvm::SmallVector<StaticFieldInfo> staticFields;

    llvm::SmallVector<MethodInfo> methods;

    llvm::SmallVector<GenericMethodInfo> staticGenericMethods;

    llvm::SmallVector<AccessorInfo> accessors;

    llvm::SmallVector<IndexInfo> indexes;

    NodeArray<ClassElement> extraMembers;
    NodeArray<ClassElement> extraMembersPost;

    // fields whose initializer has no type yet in a discovery pass (`fn = this.m`: the methods are
    // declared after the fields), typed again once the members are (mlirGenClassRetypeFields)
    llvm::SmallVector<PropertyDeclaration> fieldsTypedAfterMembers;

    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;

    bool isDeclaration;
    bool hasNew;
    bool hasConstructor;
    mlir_ts::AccessLevel constructorAccessLevel;
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
    // `@cpp`: a binding to C++ objects (MSVC/Itanium single inheritance) made elsewhere. The
    // instance is the C++ object, its first word the C++ vtable, and the class's methods are that
    // vtable's slots from 0 in declaration order, a base class's first: no `.instanceOf`, `.new`,
    // interface or static-field slots, no RTTI, and no vtable or constructor of its own.
    bool isCppVTable;
    // `@final`: nothing extends it, so a method of a value of this type is called directly. The
    // vtable stays: interfaces, `instanceof` and calls through a base type use it.
    bool isFinal;
    // `@value`: its values are copied, not shared - no identity, no vtable, no base class, no
    // interface, no type parameters. Set for a `declare` declaration too.
    bool isValue;
    // a @cpp class with methods that have bodies: C++ methods implemented here, so C++ can
    // call them through its objects. It has a vtable (the C++ one's layout: the slots and nothing
    // else, a slot left to a declared C++ method null), may have a constructor and field
    // initializers, and `.new`; still no RTTI. tslang allocates its objects.
    bool isCppVTableImplementation = false;
    // two @cpp methods in one slot, or an override's @slot not its base's: found by
    // getVirtualTable, which can't report it, and reported by its caller
    std::string cppSlotConflict;
    ProcessingStages processingAtEvaluation;
    ProcessingStages processing;
    // the module `processing` was reached in: a discovery module is thrown away with what it holds
    mlir::Operation *processingModule = nullptr;
    // and the declaration it was reached from
    ClassLikeDeclaration processingDeclaration;

    ClassInfo()
        : isDeclaration(false), hasNew(false), hasConstructor(false), constructorAccessLevel(mlir_ts::AccessLevel::Public), hasInitializers(false), hasStaticConstructor(false),
          hasStaticInitializers(false), hasVirtualTable(false), isAbstract(false), isExport(false), isImport(false), 
          isPublic(false), isDynamicImport(false), hasRTTI(false), isCppVTable(false), isFinal(false), isValue(false),
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
            if (base->getHasConstructor())
            {
                return true;
            }
        }

        return false;
    }

    auto getConstructorAccessLevel() -> mlir_ts::AccessLevel
    {
        if (hasConstructor)
        {
            return constructorAccessLevel;
        }

        for (auto &base : baseClasses)
        {
            if (base->hasConstructor)
            {
                return base->getConstructorAccessLevel();
            }
        }

        return mlir_ts::AccessLevel::Public;
    }    

    auto getHasVirtualTable() -> bool
    {
        if (hasVirtualTable)
        {
            return true;
        }

        for (auto &base : baseClasses)
        {
            if (base->getHasVirtualTable())
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
            if (base->getHasVirtualTable())
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
        // name -> slot indices built alongside the vtable: override resolution is O(1)
        // per method instead of a linear scan of the accumulated table, and keeping
        // method and interface slots in separate maps prevents a method whose name
        // matches an interface full name from clobbering the interface slot (the old
        // find_if searched both kinds through methodInfo.name).
        llvm::StringMap<size_t> methodSlots;
        llvm::StringMap<size_t> interfaceSlots;
        getVirtualTable(vtable, methodSlots, interfaceSlots);
    }

    void getVirtualTable(llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> &vtable,
                         llvm::StringMap<size_t> &methodSlots, llvm::StringMap<size_t> &interfaceSlots)
    {
        // in static class I don't want to have virtual table
        if (isStatic)
        {
            return;
        }

        for (auto &base : baseClasses)
        {
            base->getVirtualTable(vtable, methodSlots, interfaceSlots);
        }

        // `.instanceOf` is slot 0 of every class vtable: an object reached with no static type (an
        // `any` being unboxed, `x instanceof C` on an opaque value) is asked through that slot
        // (mlirGenInstanceOfOpaque). A root class puts it there ahead of its interfaces, and a
        // derived class overrides it in place. Before this, a root class that implemented an
        // interface kept the interface's vtable in slot 0, and `<B>anyValue` called it.
        if (baseClasses.empty() && !isCppVTable)
        {
            auto instanceOf = std::find_if(methods.begin(), methods.end(),
                                           [](auto &method) { return method.name == INSTANCEOF_NAME; });
            if (instanceOf != methods.end() && instanceOf->isVirtual && !methodSlots.contains(instanceOf->name))
            {
                methodSlots[instanceOf->name] = vtable.size();
                vtable.push_back({*instanceOf, false});
            }
        }

        // do vtable for current class (a @cpp class implements nothing, see isCppVTable)
        for (auto &implement : implements)
        {
            if (interfaceSlots.contains(implement.interface->fullName))
            {
                // found interface
                continue;
            }

            MethodInfo methodInfo;
            methodInfo.name = implement.interface->fullName.str();
            implement.virtualIndex = vtable.size();
            interfaceSlots[implement.interface->fullName] = vtable.size();
            vtable.push_back({methodInfo, true});
        }

        // methods
        // stable_sort: slot order is ABI for separately compiled modules, so members
        // with equal orderWeight (e.g. compiler-generated ones) must keep their
        // declaration order instead of getting an implementation-defined one
        std::stable_sort(methods.begin(), methods.end(), [] (auto &method1, auto &method2) {
            return method1.orderWeight < method2.orderWeight;
        });

        for (auto &method : methods)
        {
#ifndef ADD_STATIC_MEMBERS_TO_VTABLE
            if (method.isStatic)
            {
                continue;
            }
#else
            if (method.isStatic && isCppVTable)
            {
                continue;
            }
#endif

            auto it = methodSlots.find(method.name);
            if (it != methodSlots.end())
            {
                // found method - override the inherited slot (name and type together:
                // the overriding function's type must win over the inherited one)
                auto index = it->second;
                if (isCppVTable && method.cppSlot >= 0 && (size_t)method.cppSlot != index && cppSlotConflict.empty())
                {
                    cppSlotConflict = "'" + mlir::StringRef(method.name).split(CPP_OVERLOAD_SEPARATOR).first.str() + "' overrides slot "
                        + std::to_string(index) + " of a base class, not slot "
                        + std::to_string(method.cppSlot);
                }

                vtable[index].methodInfo.funcName = method.funcName;
                vtable[index].methodInfo.funcType = method.funcType;
                method.virtualIndex = index;
                method.isVirtual = true;
                vtable[index].methodInfo.isAbstract = method.isAbstract;
            }
            else if (method.isVirtual && isCppVTable && method.cppSlot >= 0)
            {
                // @slot(N): slot N, whatever the declaration order; the slots no method names
                // are gaps, which a class implementing C++ methods leaves null
                auto index = (size_t)method.cppSlot;

                // a C++ overload of a base class's method takes other parameters: with the same ones
                // it is that method, overridden in the wrong slot
                auto declaredName = mlir::StringRef(method.name).split(CPP_OVERLOAD_SEPARATOR).first;
                auto params = method.funcType.getInputs().drop_front(method.funcType.getNumInputs() > 0 ? 1 : 0);
                for (auto [slot, entry] : llvm::enumerate(vtable))
                {
                    auto &inherited = entry.methodInfo;
                    if (!entry.isInterfaceVTable && !entry.isStaticField && !inherited.isCppSlotGap && slot != index
                        && inherited.funcType && inherited.funcType.getNumInputs() > 0
                        && mlir::StringRef(inherited.name).split(CPP_OVERLOAD_SEPARATOR).first == declaredName
                        && inherited.funcType.getInputs().drop_front() == params && cppSlotConflict.empty())
                    {
                        cppSlotConflict = "'" + declaredName.str() + "' overrides slot " + std::to_string(slot)
                            + " of a base class, not slot " + std::to_string(index);
                    }
                }
                while (vtable.size() <= index)
                {
                    MethodInfo gap;
                    gap.name = ".slot" + std::to_string(vtable.size());
                    gap.funcType = mlir_ts::FunctionType::get(classType.getContext(), {}, {}, false);
                    gap.isStatic = false;
                    gap.isVirtual = true;
                    gap.isAbstract = false;
                    gap.virtualIndex = vtable.size();
                    gap.orderWeight = 0;
                    gap.accessLevel = mlir_ts::AccessLevel::Public;
                    gap.isCppSlotGap = true;
                    vtable.push_back({gap, false});
                }

                if (!vtable[index].methodInfo.isCppSlotGap)
                {
                    if (cppSlotConflict.empty())
                    {
                        cppSlotConflict = "'" + mlir::StringRef(method.name).split(CPP_OVERLOAD_SEPARATOR).first.str() + "' and '"
                            + mlir::StringRef(vtable[index].methodInfo.name).split(CPP_OVERLOAD_SEPARATOR).first.str() + "' are both in slot "
                            + std::to_string(index);
                    }

                    continue;
                }

                method.virtualIndex = index;
                methodSlots[method.name] = index;
                vtable[index] = {method, false};
            }
            else if (method.isVirtual)
            {
                method.virtualIndex = vtable.size();
                methodSlots[method.name] = vtable.size();
                vtable.push_back({method, false});
            }
        }

#ifdef ADD_STATIC_MEMBERS_TO_VTABLE
        // static fields (a C++ vtable has none)
        for (auto &staticField : isCppVTable ? llvm::MutableArrayRef<StaticFieldInfo>() : llvm::MutableArrayRef<StaticFieldInfo>(staticFields))
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

    auto hasBase(mlir_ts::ClassType classType) -> bool
    {
        for (auto &base : baseClasses)
        {
            if (base->classType == classType || base->hasBase(classType))
            {
                return true;
            }
        }

        return false;
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
        auto storageClass = cast<mlir_ts::ClassStorageType>(classType.getStorageType());
        return storageClass.size();
    }

    mlir_ts::FieldInfo fieldInfoByIndex(int index)
    {
        if (index >= 0)
        {
            auto storageClass = cast<mlir_ts::ClassStorageType>(classType.getStorageType());
            return storageClass.getFieldInfo(index);
        }

        return mlir_ts::FieldInfo();
    }

    mlir_ts::FieldInfo findField(mlir::Attribute id, bool &foundField)
    {
        foundField = false;
        auto storageClass = cast<mlir_ts::ClassStorageType>(classType.getStorageType());
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

    NamespaceInfo_TypePtr elementNamespace;

    llvm::SmallVector<TypeParameterDOM::TypePtr> typeParams;

    mlir_ts::ClassType classType;

    ClassLikeDeclaration classDeclaration;

    SourceFile sourceFile;
    StringRef fileName;

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

    llvm::StringMap<FunctionEntry> functionMap;

    llvm::StringMap<GenericFunctionInfo::TypePtr> genericFunctionMap;

    llvm::StringMap<VariableDeclarationDOM::TypePtr> globalsMap;

    llvm::StringMap<llvm::StringMap<ts::VariableDeclarationDOM::TypePtr>> captureVarsMap;

    llvm::StringMap<llvm::SmallVector<mlir::typescript::FieldInfo>> localVarsInThisContextMap;

    llvm::StringMap<std::pair<mlir::Type, TypeNode>> typeAliasMap;

    llvm::StringMap<std::pair<llvm::SmallVector<TypeParameterDOM::TypePtr>, TypeNode>> genericTypeAliasMap;

    llvm::StringMap<mlir::StringRef> importEqualsMap;

    // `import { a as b }`: b -> a, a name the imported module declares
    llvm::StringMap<mlir::StringRef> importAliasMap;

    llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> enumsMap;

    llvm::StringMap<ClassInfo::TypePtr> classesMap;

    llvm::StringMap<GenericClassInfo::TypePtr> genericClassesMap;

    llvm::StringMap<InterfaceInfo::TypePtr> interfacesMap;

    llvm::StringMap<GenericInterfaceInfo::TypePtr> genericInterfacesMap;

    llvm::StringMap<NamespaceInfo::TypePtr> namespacesMap;

    bool isFunctionNamespace;

    // `import * as M`: M has nothing in it, its names are the module's, which are at the top level
    bool isModuleImport = false;

    NamespaceInfo::TypePtr parentNamespace;
};

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_MLIRGENSTORE_H_
