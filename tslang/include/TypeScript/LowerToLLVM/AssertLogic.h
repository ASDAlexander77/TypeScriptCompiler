#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ASSERTLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ASSERTLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"
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
    LLVMCodeHelperBase ch;
    CodeLogicHelper clh;
    Location loc;

  protected:
    mlir::Type sizeType;
    mlir::Type typeOfValueType;

  public:
    AssertLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), th(rewriter), ch(op, rewriter, tch.typeConverter, compileOptions), clh(op, rewriter), loc(loc)
    {
        sizeType = th.getIndexType();
        typeOfValueType = th.getPtrType();
    }

    AssertLogic(Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter, Location loc, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), th(rewriter), ch(op, rewriter, typeConverter, compileOptions), clh(op, rewriter), loc(loc)
    {
        sizeType = th.getIndexType();
        typeOfValueType = th.getPtrType();
    }

    // `message`, when given, is the text known only at run time and is shown instead of `msg`
    mlir::LogicalResult logic(mlir::Value condValue, std::string msg, mlir::Value message = mlir::Value())
    {
        // the test replaces the assert op, which the split left at the top of the continuation
        failUnless(condValue, msg, message);
        rewriter.eraseOp(op);
        return success();
    }

    // A check in the middle of another lowering (#483): execution goes on at the insertion point
    // when `condValue` holds, and stops with `msg` and the operation's file and line, as a failing
    // assert does, when it does not.
    void check(mlir::Value condValue, std::string msg)
    {
        auto *continuationBlock = failUnless(condValue, msg, mlir::Value());
        rewriter.setInsertionPointToStart(continuationBlock);
    }

    // `_assert` and `__assert_fail` abort, which flushes no stream: what the program printed before
    // the failure was lost whenever stdout was not a console (a pipe, a file, the test runner)
    void flushOutput()
    {
        auto fflushFuncOp = ch.getOrInsertFunction("fflush", th.getFunctionType(rewriter.getI32Type(), {th.getPtrType()}));
        mlir::Value allStreams = rewriter.create<LLVM::ZeroOp>(loc, th.getPtrType());
        rewriter.create<LLVM::CallOp>(loc, fflushFuncOp, ValueRange{allStreams});
    }

    // a null string shows the constant message instead
    mlir::Value messageOrConstant(mlir::Value message, mlir::Value msgCst)
    {
        if (!message)
        {
            return msgCst;
        }

        auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, message.getType());
        auto isNull = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, message, nullPtr);
        return rewriter.create<LLVM::SelectOp>(loc, isNull, msgCst, message);
    }

  private:
    // Splits the block at the insertion point and ends the first half with a branch on
    // `condValue`: on to the continuation, which is returned, or to a new block that reports the
    // failure (`_assert` on Windows, `__assert_fail` elsewhere) and never returns.
    mlir::Block *failUnless(mlir::Value condValue, std::string msg, mlir::Value message)
    {
        auto unreachable = clh.FindUnreachableBlockOrCreate();

        auto [fileName, lineAndColumn] = LLVMLocationHelper::getLineAndColumnAndFileName(loc);
        auto [line, column] = lineAndColumn;

        auto i8PtrTy = th.getPtrType();
#ifdef WIN32
        auto assertFuncOp =
            ch.getOrInsertFunction("_assert", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, rewriter.getI32Type()}));
#else
        auto assertFuncOp = ch.getOrInsertFunction(
            "__assert_fail", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, rewriter.getI32Type(), i8PtrTy}));
#endif

        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        auto *failureBlock = rewriter.createBlock(opBlock->getParent());

        auto opHash = std::hash<std::string>{}(msg);

        std::stringstream msgVarName;
        msgVarName << "m_" << opHash;

        std::stringstream fileVarName;
        fileVarName << "f_" << hash_value(fileName);

        auto msgCst = ch.getOrCreateGlobalString(msgVarName.str(), msg);
        auto fileCst = ch.getOrCreateGlobalString(fileVarName.str(), fileName.str());

        mlir::Value lineNumberRes = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(line));

        flushOutput();
#ifdef WIN32
        rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{messageOrConstant(message, msgCst), fileCst, lineNumberRes});
#else
        mlir::Value funcName = rewriter.create<LLVM::ZeroOp>(loc, i8PtrTy);
        rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{messageOrConstant(message, msgCst), fileCst, lineNumberRes, funcName});
#endif
        rewriter.create<mlir::cf::BranchOp>(loc, unreachable);

        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::CondBrOp>(loc, condValue, continuationBlock, failureBlock);

        return continuationBlock;
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ASSERTLOGIC_H_
