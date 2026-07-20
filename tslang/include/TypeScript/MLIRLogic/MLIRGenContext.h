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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "TypeScript/DOM.h"
#include "TypeScript/MLIRLogic/MLIRGenStore.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

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
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

// These types are shared across the MLIRGen translation units (see MLIRGenImpl.h),
// so they need external linkage — an anonymous namespace would give each TU its own type.
namespace typescript
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

    void clearScopeVars()
    {
        passResult = nullptr;
        capturedVars = nullptr;
        usingVars = nullptr;

        currentOperation = nullptr;
        allocateVarsOutsideOfOperation = false;
        allocateUsingVarsOutsideOfOperation = false;
    }

    void clearReceiverTypes()
    {
        receiverType = mlir::Type();
        receiverFuncType = mlir::Type();
    }

    // erases the IR collected during a dummy run (cleanup lists, dummy funcOp) and detaches from it;
    // the lists, passResult and state are owned by the stack frame that set them, not by GenContext
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

            cleanUps->clear();
            cleanUps = nullptr;
        }

        if (cleanUpOps)
        {
            for (auto op : *cleanUpOps)
            {
                op->dropAllDefinedValueUses();
                op->dropAllUses();
                op->dropAllReferences();
                op->erase();
            }

            cleanUpOps->clear();
            cleanUpOps = nullptr;
        }

        passResult = nullptr;
        state = nullptr;

        cleanFuncOp();
    }

    void cleanFuncOp()
    {
        if (funcOp)
        {
            funcOp->dropAllDefinedValueUses();
            funcOp->dropAllUses();
            funcOp->dropAllReferences();
            funcOp->erase();
        }
    }

    void stop() const
    {
        if (stopProcess) return;
        stopProcess = true;
        if (rootContext) rootContext->stop();
    }

    bool isStopped() const
    {
        return stopProcess || rootContext && rootContext->stopProcess;
    }

    bool allowPartialResolve = false;
    bool dummyRun = false;
    bool allowConstEval = false;
    bool allocateVarsInContextThis = false;
    bool allocateVarsOutsideOfOperation = false;
    bool allocateUsingVarsOutsideOfOperation = false;
    bool forceDiscover = false;
    bool discoverParamsOnly = false;
    bool insertIntoParentScope = false;
    mlir::Operation *currentOperation = nullptr;
    mlir_ts::FuncOp funcOp;
    FunctionPrototypeDOM::TypePtr funcProto;
    llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> *capturedVars = nullptr;
    llvm::SmallVector<ts::VariableDeclarationDOM::TypePtr> *usingVars = nullptr;
    mlir::Type thisType;
    mlir_ts::ClassType thisClassType;
    mlir::Type receiverFuncType;
    mlir::Type receiverType;
    mlir::StringRef receiverName;
    bool isGlobalVarReceiver = false;
    // set while generating the initializer of an exported var/let - lets a
    // nested object-literal's method-like properties (processObjectFunctionLike)
    // know they must be forced to public/external linkage, since their
    // function pointer is reachable only indirectly through the exported
    // global's data, not by name; left private, the linker can strip them and
    // the boxed global ends up with a null method slot cross-module.
    bool isExportVarReceiver = false;
    PassResult *passResult = nullptr;
    mlir::SmallVector<mlir::Block *> *cleanUps = nullptr;
    mlir::SmallVector<mlir::Operation *> *cleanUpOps = nullptr;
    NodeArray<Statement> generatedStatements;
    llvm::StringMap<mlir::Type> typeAliasMap;
    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;
    ArrayRef<mlir::Value> callOperands;
    int *state = nullptr;
    bool disableSpreadParams = false;
    const GenContext* parentBlockContext = nullptr;
    const GenContext* rootContext = nullptr;
    bool isLoop = false;
    std::string loopLabel;
    // out-of-band cancellation signal; mutable so stop() stays callable through the const& threading
    mutable bool stopProcess = false;
    mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> *postponedMessages = nullptr;
    bool specialization = false;
    // TODO: special hack to detect initializing specialized class and see that generic methods are not initialized at the same time
    bool instantiateSpecializedFunction = false;
    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> *inferTypes = nullptr;
};

struct ValueOrLogicalResult 
{
    ValueOrLogicalResult() : result(mlir::success()), value(mlir::Value()) {};
    ValueOrLogicalResult(mlir::LogicalResult result) : result(result) {};
    ValueOrLogicalResult(mlir::Value value) : result(mlir::success()), value(value) {};

    mlir::LogicalResult result;
    mlir::Value value;

    operator bool()
    {
        return mlir::succeeded(result);
    }

    bool failed()
    {
        return mlir::failed(result);
    }

    bool failed_or_no_value()
    {
        return failed() || !value;
    }

    operator mlir::LogicalResult()
    {
        return result;
    } 

    operator mlir::Value()
    {
        return value;
    }    
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_MLIRGENCONTEXT_H_
