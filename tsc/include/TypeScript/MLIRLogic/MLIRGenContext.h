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

        if (cleanUpOps)
        {
            for (auto op : *cleanUpOps)
            {
                op->dropAllDefinedValueUses();
                op->dropAllUses();
                op->dropAllReferences();
                op->erase();
            }

            delete cleanUpOps;
            cleanUpOps = nullptr;               
        }

        if (passResult)
        {
            delete passResult;
            passResult = nullptr;
        }

        cleanState();

        cleanFuncOp();
    }

    void cleanState()
    {
        if (state)
        {
            delete state;
            state = nullptr;
        }
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

    void stop()
    {
        if (stopProcess) return;
        stopProcess = true;
        if (rootContext) const_cast<GenContext *>(rootContext)->stop();
    }

    bool isStopped() const
    {
        return stopProcess || rootContext && rootContext->stopProcess;
    }

    bool allowPartialResolve;
    bool dummyRun;
    bool allowConstEval;
    bool allocateVarsInContextThis;
    bool allocateVarsOutsideOfOperation;
    bool allocateUsingVarsOutsideOfOperation;
    bool forceDiscover;
    bool discoverParamsOnly;
    bool insertIntoParentScope;
    mlir::Operation *currentOperation;
    mlir_ts::FuncOp funcOp;
    FunctionPrototypeDOM::TypePtr funcProto;
    llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> *capturedVars;
    llvm::SmallVector<ts::VariableDeclarationDOM::TypePtr> *usingVars;
    mlir::Type thisType;
    mlir::Type receiverFuncType;
    mlir::Type receiverType;
    mlir::StringRef receiverName;
    bool isGlobalVarReceiver;
    PassResult *passResult;
    mlir::SmallVector<mlir::Block *> *cleanUps;
    mlir::SmallVector<mlir::Operation *> *cleanUpOps;
    NodeArray<Statement> generatedStatements;
    llvm::StringMap<mlir::Type> typeAliasMap;
    llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;
    ArrayRef<mlir::Value> callOperands;
    int *state;
    bool disableSpreadParams;
    const GenContext* parentBlockContext;
    const GenContext* rootContext;
    bool isLoop;
    std::string loopLabel;
    bool stopProcess;
    mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> *postponedMessages;
    bool specialization;
    // TODO: special hack to detect initializing specialized class and see that generic methods are not initialized at the same time
    bool instantiateSpecializedFunction;
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

#define V(x) static_cast<mlir::Value>(x)

#define CAST(res_cast, location, to_type, from_value, gen_context) \
    auto cast_result = cast(location, to_type, from_value, gen_context); \
    EXIT_IF_FAILED_OR_NO_VALUE(cast_result) \
    res_cast = V(cast_result);

#define CAST_A(res_cast, location, to_type, from_value, gen_context) \
    mlir::Value res_cast; \
    { \
        auto cast_result = cast(location, to_type, from_value, gen_context); \
        EXIT_IF_FAILED_OR_NO_VALUE(cast_result) \
        res_cast = V(cast_result); \
    }

#define DECLARE(varDesc, varValue) \
    if (mlir::failed(declare(location, varDesc, varValue, genContext))) \
    { \
        return mlir::failure(); \
    }

} // namespace

#endif // MLIR_TYPESCRIPT_MLIRGENCONTEXT_H_
