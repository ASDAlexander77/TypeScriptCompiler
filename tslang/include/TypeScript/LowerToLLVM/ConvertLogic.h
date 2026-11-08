#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CONVERTERLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CONVERTERLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"

#include <limits>

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class ConvertLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    TypeConverterHelper &tch;
    TypeHelper th;
    LLVMCodeHelperBase ch;
    CodeLogicHelper clh;
    Location loc;
    CompileOptions &compileOptions;

  public:
    ConvertLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter, tch.typeConverter, compileOptions), clh(op, rewriter), loc(loc),
          compileOptions(compileOptions)
    {
    }

    mlir::Value sprintf(int buffSize, std::string format, mlir::Value value)
    {
        auto i8PtrTy = th.getPtrType();

        auto llvmIndexType = tch.convertType(th.getIndexType());
        auto bufferSizeValue = clh.createIndexConstantOf(llvmIndexType, buffSize);

        auto opHash = std::hash<std::string>{}(format);

        std::stringstream formatVarName;
        formatVarName << "frmt_" << opHash;

        auto formatSpecifierCst = ch.getOrCreateGlobalString(formatVarName.str(), format);

        auto newVal = rewriter.create<mlir_ts::ConvertFOp>(loc, th.getStringType(), bufferSizeValue, formatSpecifierCst, ValueRange{value});
        return newVal;
    }

    // A number as JS writes it (Number::toString, ECMA-262 6.1.6.1.20): the fewest digits that read
    // back as the same double, plain decimal for a decimal exponent n in -6 < n <= 21 and `d.ddde+N`
    // otherwise; NaN, Infinity, -Infinity, and "0" for -0. `%g` kept 6 digits (`0.1 + 0.2` printed
    // 0.3, `2 ** 53 + 2` 9.0072e+015), and the MSVC CRT writes 3-digit exponents. The work is done
    // by a routine of the module's own (getOrCreateNumberToStringRoutine), over the CRT's
    // snprintf/sprintf_s and strtod, which every target already links for the other conversions and
    // parseFloat.
    mlir::Value sprintfOfF64(mlir::Value value)
    {
#ifdef NUMBER_F64
        mlir::Value doubleValue = value;
#else
        mlir::Value doubleValue = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), value);
#endif
        auto llvmIndexType = tch.convertType(th.getIndexType());
        auto bufferSizeValue = clh.createIndexConstantOf(llvmIndexType, NUMBER_TO_STRING_BUFFER_SIZE);

        auto routine = getOrCreateNumberToStringRoutine();

        auto buffer = ch.MemoryAlloc(bufferSizeValue, MemoryAllocSet::Atomic);
        rewriter.create<LLVM::CallOp>(loc, routine, ValueRange{buffer, doubleValue});
        return buffer;
    }

    // the longest text is 25 characters: "-0.00000" and 17 digits, or "-d.dddddddddddddddde-308"
    static constexpr int NUMBER_TO_STRING_BUFFER_SIZE = 32;

    // `void __ts_number_to_string(ptr out, f64 v)`, internal to the module (a public one in every
    // module collides when the modules link together), writing at most
    // NUMBER_TO_STRING_BUFFER_SIZE bytes. Digits: for p = 0..16, `%.*e` with p decimals until
    // strtod reads the same double back, which gives the shortest correctly rounded digit string
    // (a shorter one ending in 0 would have matched first, so it has no trailing zeros). The
    // digits and the exponent are then laid out as JS does, by snprintf with `%.*s` pieces.
    LLVM::LLVMFuncOp getOrCreateNumberToStringRoutine()
    {
        constexpr auto name = "__ts_number_to_string";
        auto parentModule = op->getParentOfType<ModuleOp>();
        if (auto existing = parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
        {
            return existing;
        }

        auto ptrTy = th.getPtrType();
        auto i8Ty = th.getI8Type();
        auto i32Ty = rewriter.getI32Type();
        auto f64Ty = rewriter.getF64Type();
        auto llvmIndexType = tch.convertType(th.getIndexType());
        auto noLoc = mlir::UnknownLoc::get(rewriter.getContext());

        // the target's CRT, not the host's, as ConvertF: sprintf_s is MSVC only
        auto printFunc = ch.getOrInsertFunction(compileOptions.isWindows ? "sprintf_s" : "snprintf",
                                                th.getFunctionType(i32Ty, {ptrTy, llvmIndexType, ptrTy}, true));
        auto strtodFunc = ch.getOrInsertFunction("strtod", th.getFunctionType(f64Ty, {ptrTy, ptrTy}));

        OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());

        auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(noLoc, name, th.getFunctionType(th.getVoidType(), {ptrTy, f64Ty}),
                                                        LLVM::Linkage::Internal);

        auto *entry = funcOp.addEntryBlock(rewriter);
        auto &body = funcOp.getBody();
        auto out = entry->getArgument(0);
        auto value = entry->getArgument(1);
        rewriter.setInsertionPointToStart(entry);

        auto newBlock = [&](TypeRange types = {}) {
            SmallVector<mlir::Location> locs(types.size(), noLoc);
            return rewriter.createBlock(&body, body.end(), types, locs);
        };

        auto i32Const = [&](int32_t v) { return clh.createI32ConstantOf(v); };
        auto text = [&](std::string s) { return ch.getOrCreateGlobalString(s); };
        auto bufferSize = clh.createIndexConstantOf(llvmIndexType, NUMBER_TO_STRING_BUFFER_SIZE);

        auto printTo = [&](mlir::Value dest, std::string format, ArrayRef<mlir::Value> args) {
            SmallVector<mlir::Value> values{dest, bufferSize, text(format)};
            values.append(args.begin(), args.end());
            rewriter.create<LLVM::CallOp>(noLoc, printFunc, values);
        };
        auto print = [&](std::string format, ArrayRef<mlir::Value> args) {
            printTo(out, format, args);
            rewriter.create<LLVM::ReturnOp>(noLoc, ValueRange{});
        };
        auto strtod = [&](mlir::Value str) {
            auto nullPtr = rewriter.create<LLVM::ZeroOp>(noLoc, ptrTy);
            return rewriter.create<LLVM::CallOp>(noLoc, strtodFunc, ValueRange{str, nullPtr}).getResult();
        };
        auto at = [&](mlir::Value base, mlir::Value index) -> mlir::Value {
            return rewriter.create<LLVM::GEPOp>(noLoc, ptrTy, i8Ty, base, ArrayRef<LLVM::GEPArg>{index});
        };
        auto icmp = [&](LLVM::ICmpPredicate predicate, mlir::Value l, mlir::Value r) -> mlir::Value {
            return rewriter.create<LLVM::ICmpOp>(noLoc, predicate, l, r);
        };
        auto fcmp = [&](LLVM::FCmpPredicate predicate, mlir::Value l, mlir::Value r) -> mlir::Value {
            return rewriter.create<LLVM::FCmpOp>(noLoc, predicate, l, r);
        };

        // entry: the scratch buffers, then NaN
        auto scratchSize = i32Const(NUMBER_TO_STRING_BUFFER_SIZE);
        auto scientific = rewriter.create<LLVM::AllocaOp>(noLoc, ptrTy, i8Ty, scratchSize);
        auto digits = rewriter.create<LLVM::AllocaOp>(noLoc, ptrTy, i8Ty, scratchSize);
        auto zero = clh.createF64ConstantOf(0.0);
        auto isNegative = fcmp(LLVM::FCmpPredicate::olt, value, zero);

        auto *nanBlock = newBlock();
        auto *notNanBlock = newBlock();
        auto *infinityBlock = newBlock();
        auto *finiteBlock = newBlock();
        auto *zeroBlock = newBlock();
        auto *searchBlock = newBlock();
        auto *loopBlock = newBlock({i32Ty});
        auto *foundBlock = newBlock();
        auto *plainIntegerBlock = newBlock();
        auto *notPlainIntegerBlock = newBlock();
        auto *plainFractionBlock = newBlock();
        auto *notPlainFractionBlock = newBlock();
        auto *smallFractionBlock = newBlock();
        auto *exponentBlock = newBlock();

        rewriter.setInsertionPointToEnd(entry);
        rewriter.create<LLVM::CondBrOp>(noLoc, fcmp(LLVM::FCmpPredicate::uno, value, value), nanBlock, notNanBlock);

        rewriter.setInsertionPointToEnd(nanBlock);
        print("NaN", {});

        rewriter.setInsertionPointToEnd(notNanBlock);
        mlir::Value magnitude = rewriter.create<LLVM::FAbsOp>(noLoc, f64Ty, value);
        auto infinity = clh.createF64ConstantOf(std::numeric_limits<double>::infinity());
        rewriter.create<LLVM::CondBrOp>(noLoc, fcmp(LLVM::FCmpPredicate::oeq, magnitude, infinity), infinityBlock,
                                        finiteBlock);

        rewriter.setInsertionPointToEnd(infinityBlock);
        print("%s", {rewriter.create<LLVM::SelectOp>(noLoc, isNegative, text("-Infinity"), text("Infinity"))});

        // -0 is "0" as well
        rewriter.setInsertionPointToEnd(finiteBlock);
        rewriter.create<LLVM::CondBrOp>(noLoc, fcmp(LLVM::FCmpPredicate::oeq, value, zero), zeroBlock, searchBlock);

        rewriter.setInsertionPointToEnd(zeroBlock);
        print("0", {});

        rewriter.setInsertionPointToEnd(searchBlock);
        mlir::Value sign = rewriter.create<LLVM::SelectOp>(noLoc, isNegative, text("-"), text(""));
        rewriter.create<LLVM::BrOp>(noLoc, ValueRange{i32Const(0)}, loopBlock);

        // loop(p): `d.ddd...e+NN` with p decimals; 17 digits (p = 16) always read back
        rewriter.setInsertionPointToEnd(loopBlock);
        mlir::Value decimals = loopBlock->getArgument(0);
        printTo(scientific, "%.*e", {decimals, magnitude});
        auto readBack = strtod(scientific);
        auto exact = fcmp(LLVM::FCmpPredicate::oeq, readBack, magnitude);
        auto last = icmp(LLVM::ICmpPredicate::sge, decimals, i32Const(16));
        auto done = rewriter.create<LLVM::OrOp>(noLoc, exact, last);
        auto nextDecimals = rewriter.create<LLVM::AddOp>(noLoc, decimals, i32Const(1));
        rewriter.create<LLVM::CondBrOp>(noLoc, done, foundBlock, ValueRange{}, loopBlock, ValueRange{nextDecimals});

        // found: k digits, the first at [0] and the rest from [2]; the exponent after the 'e' at
        // [1] (no decimals) or [p + 2], read by strtod whatever its width ("e+08", MSVC's "e+008")
        rewriter.setInsertionPointToEnd(foundBlock);
        auto digitCount = rewriter.create<LLVM::AddOp>(noLoc, decimals, i32Const(1));
        auto noDecimals = icmp(LLVM::ICmpPredicate::eq, decimals, i32Const(0));
        auto exponentIndex = rewriter.create<LLVM::SelectOp>(noLoc, noDecimals, i32Const(2),
                                                             rewriter.create<LLVM::AddOp>(noLoc, decimals, i32Const(3)));
        auto exponent = rewriter.create<LLVM::FPToSIOp>(noLoc, i32Ty, strtod(at(scientific, exponentIndex)));
        // n: the position of the decimal point relative to the digits
        mlir::Value n = rewriter.create<LLVM::AddOp>(noLoc, exponent, i32Const(1));
        mlir::Value firstDigit = rewriter.create<LLVM::SExtOp>(noLoc, i32Ty, rewriter.create<LLVM::LoadOp>(noLoc, i8Ty, scientific));
        printTo(digits, "%c%.*s", {firstDigit, decimals, at(scientific, i32Const(2))});
        auto zeros = text("000000000000000000000");
        auto upTo21 = icmp(LLVM::ICmpPredicate::sle, n, i32Const(21));
        auto plainInteger = rewriter.create<LLVM::AndOp>(noLoc, icmp(LLVM::ICmpPredicate::sle, digitCount, n), upTo21);
        rewriter.create<LLVM::CondBrOp>(noLoc, plainInteger, plainIntegerBlock, notPlainIntegerBlock);

        // k <= n <= 21: the digits, then n - k zeros
        rewriter.setInsertionPointToEnd(plainIntegerBlock);
        print("%s%s%.*s", {sign, digits, rewriter.create<LLVM::SubOp>(noLoc, n, digitCount), zeros});

        rewriter.setInsertionPointToEnd(notPlainIntegerBlock);
        auto plainFraction = rewriter.create<LLVM::AndOp>(noLoc, icmp(LLVM::ICmpPredicate::sgt, n, i32Const(0)), upTo21);
        rewriter.create<LLVM::CondBrOp>(noLoc, plainFraction, plainFractionBlock, notPlainFractionBlock);

        // 0 < n < k: the first n digits, '.', the rest
        rewriter.setInsertionPointToEnd(plainFractionBlock);
        print("%s%.*s.%s", {sign, n, digits, at(digits, n)});

        rewriter.setInsertionPointToEnd(notPlainFractionBlock);
        auto smallFraction = rewriter.create<LLVM::AndOp>(noLoc, icmp(LLVM::ICmpPredicate::sgt, n, i32Const(-6)),
                                                          icmp(LLVM::ICmpPredicate::sle, n, i32Const(0)));
        rewriter.create<LLVM::CondBrOp>(noLoc, smallFraction, smallFractionBlock, exponentBlock);

        // -6 < n <= 0: "0.", -n zeros, the digits
        rewriter.setInsertionPointToEnd(smallFractionBlock);
        print("%s0.%.*s%s", {sign, rewriter.create<LLVM::SubOp>(noLoc, i32Const(0), n), zeros, digits});

        // otherwise d[.ddd]e+N / e-N
        rewriter.setInsertionPointToEnd(exponentBlock);
        auto dot = rewriter.create<LLVM::SelectOp>(noLoc, icmp(LLVM::ICmpPredicate::sgt, digitCount, i32Const(1)), text("."), text(""));
        print("%s%c%s%se%+d", {sign, firstDigit, dot, at(digits, i32Const(1)), rewriter.create<LLVM::SubOp>(noLoc, n, i32Const(1))});

        // the helpers above take the user's location for the constants and the string addresses;
        // a routine of its own has no debug scope to name
        funcOp->walk([&](mlir::Operation *nested) { nested->setLoc(noLoc); });

        return funcOp;
    }

    mlir::Value sprintfOfInt(mlir::Value valueIn, int width, bool isSigned)
    {
        mlir::Value value = valueIn;

        std::string frm = "%";

        if (isSigned)
        {
            frm += "-";
        }

        switch (width)
        {
            case 8: 
                frm += "hh";
                break;
            case 16: 
                frm += "h";
                break;
            case 64: 
                frm += "ll";
                break;
            default:
                break;
        }

        if (isSigned)
        {
            frm += "i";
        }
        else
        {
            frm += "u";
        }

        if (width < 32)
        {
            if (isSigned)
            {
                value = rewriter.create<LLVM::SExtOp>(loc, tch.convertType(rewriter.getIntegerType(32)), value);
            }
            else
            {
                value = rewriter.create<LLVM::ZExtOp>(loc, tch.convertType(rewriter.getIntegerType(32)), value);
            }
        }

        return sprintf(50, frm, value);
    }

    mlir::Value intToString(mlir::Value value, int width, bool isSigned)
    {
        return sprintfOfInt(value, width, isSigned);
    }

    mlir::Value f64ToString(mlir::Value value)
    {
        return sprintfOfF64(value);
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CONVERTERLOGIC_H_
