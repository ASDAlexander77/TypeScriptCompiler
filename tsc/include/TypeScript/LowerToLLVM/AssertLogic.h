#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ASSERTLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ASSERTLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/LocationHelper.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class AssertLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    TypeHelper th;
    LLVMCodeHelper ch;
    CodeLogicHelper clh;
    Location loc;

  protected:
    mlir::Type sizeType;
    mlir::Type typeOfValueType;

  public:
    AssertLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), th(rewriter), ch(op, rewriter, &tch.typeConverter, compileOptions), clh(op, rewriter), loc(loc)
    {
        sizeType = th.getIndexType();
        typeOfValueType = th.getI8PtrType();
    }

    mlir::LogicalResult logic(mlir::Value condValue, std::string msg)
    {
#ifdef WIN32
        return logicWin32(condValue, msg);
#else
        return logicUnix(condValue, msg);
#endif
    }

    mlir::LogicalResult logicWin32(mlir::Value condValue, std::string msg)
    {
        auto unreachable = clh.FindUnreachableBlockOrCreate();

        auto [fileName, lineAndColumn] = LLVMLocationHelper::getLineAndColumnAndFileName(loc);
        auto [line, column] = lineAndColumn;

        // Insert the `_assert` declaration if necessary.
        auto i8PtrTy = th.getI8PtrType();
        auto assertFuncOp =
            ch.getOrInsertFunction("_assert", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, rewriter.getI32Type()}));

        // Split block at `assert` operation.
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // Generate IR to call `assert`.
        auto *failureBlock = rewriter.createBlock(opBlock->getParent());

        std::stringstream msgWithNUL;
        msgWithNUL << msg;

        auto opHash = std::hash<std::string>{}(msgWithNUL.str());

        std::stringstream msgVarName;
        msgVarName << "m_" << opHash;

        std::stringstream fileVarName;
        fileVarName << "f_" << hash_value(fileName);

        std::stringstream fileWithNUL;
        fileWithNUL << fileName.str();

        auto msgCst = ch.getOrCreateGlobalString(msgVarName.str(), msgWithNUL.str());

        auto fileCst = ch.getOrCreateGlobalString(fileVarName.str(), fileName.str());

        // auto nullCst = rewriter.create<LLVM::NullOp>(loc, getI8PtrType(context));

        mlir::Value lineNumberRes = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(line));

        rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{msgCst, fileCst, lineNumberRes});
        // rewriter.create<LLVM::UnreachableOp>(loc);
        rewriter.create<mlir::cf::BranchOp>(loc, unreachable);

        // Generate assertion test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, condValue, continuationBlock, failureBlock);

        return success();
    }

    mlir::LogicalResult logicUnix(mlir::Value condValue, std::string msg)
    {
        auto unreachable = clh.FindUnreachableBlockOrCreate();

        auto [fileName, lineAndColumn] = LLVMLocationHelper::getLineAndColumnAndFileName(loc);
        auto [line, column] = lineAndColumn;

        // Insert the `_assert` declaration if necessary.
        auto i8PtrTy = th.getI8PtrType();
        auto assertFuncOp = ch.getOrInsertFunction(
            "__assert_fail", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, rewriter.getI32Type(), i8PtrTy}));

        // Split block at `assert` operation.
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // Generate IR to call `assert`.
        auto *failureBlock = rewriter.createBlock(opBlock->getParent());

        std::stringstream msgWithNUL;
        msgWithNUL << msg;

        auto opHash = std::hash<std::string>{}(msgWithNUL.str());

        std::stringstream msgVarName;
        msgVarName << "m_" << opHash;

        std::stringstream fileVarName;
        fileVarName << "f_" << hash_value(fileName);

        std::stringstream fileWithNUL;
        fileWithNUL << fileName.str();

        auto msgCst = ch.getOrCreateGlobalString(msgVarName.str(), msgWithNUL.str());

        auto fileCst = ch.getOrCreateGlobalString(fileVarName.str(), fileName.str());

        // auto nullCst = rewriter.create<LLVM::NullOp>(loc, getI8PtrType(context));

        mlir::Value lineNumberRes = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(line));
        mlir::Value funcName = rewriter.create<LLVM::NullOp>(loc, i8PtrTy);

        rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{msgCst, fileCst, lineNumberRes, funcName});
        // rewriter.create<LLVM::UnreachableOp>(loc);
        rewriter.create<mlir::cf::BranchOp>(loc, unreachable);

        // Generate assertion test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, condValue, continuationBlock, failureBlock);

        return success();
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ASSERTLOGIC_H_
