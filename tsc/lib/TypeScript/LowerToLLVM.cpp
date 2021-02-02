//===----------------------------------------------------------------------===//
//
// This file implements full lowering of TypeScript operations to LLVM MLIR dialect.
// 'typescript.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the TypeScriptToLLVMLoweringPass. This pass
// lowers the combination of Affine + SCF + Standard dialects to the LLVM one:
//
//                                Affine --
//                                        |
//                                        v
//                                        Standard --> LLVM (Dialect)
//                                        ^
//                                        |
//     'typescript.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TypeScriptToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace
{
    class BaseConversionPattern : public ConversionPattern
    {
    public:
        explicit BaseConversionPattern(StringRef rootName, PatternBenefit benefit, MLIRContext *ctx)
            : ConversionPattern(rootName, benefit, ctx)
        {
            context = ctx;
        }

        /// Return a value representing an access into a global string with the given
        /// name, creating the string if necessary.
        static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                             StringRef name, StringRef value,
                                             ModuleOp module)
        {
            // Create the global at the entry of the module.
            LLVM::GlobalOp global;
            if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name)))
            {
                OpBuilder::InsertionGuard insertGuard(builder);
                builder.setInsertionPointToStart(module.getBody());
                auto type = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), value.size());
                global = builder.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name, builder.getStringAttr(value));
            }

            // Get the pointer to the first character in the global string.
            Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
            Value cst0 = builder.create<LLVM::ConstantOp>(
                loc, IntegerType::get(builder.getContext(), 64),
                builder.getIntegerAttr(builder.getIndexType(), 0));
            return builder.create<LLVM::GEPOp>(
                loc,
                getI8PtrType(builder.getContext()),
                globalPtr, ArrayRef<Value>({cst0, cst0}));
        }

        static LLVM::LLVMFuncOp getOrInsertFunction(
            PatternRewriter &rewriter, ModuleOp module, const StringRef &name, const LLVM::LLVMFunctionType &llvmFnType)
        {
            if (auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
            {
                return funcOp;
            }

            // Insert the printf function into the body of the parent module.
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
        }

        static LLVM::LLVMVoidType getVoidType(MLIRContext *context)
        {
            return LLVM::LLVMVoidType::get(context);
        }

        static IntegerType getI8Type(MLIRContext *context)
        {
            return IntegerType::get(context, 8);
        }

        static IntegerType getI32Type(MLIRContext *context)
        {
            return IntegerType::get(context, 32);
        }

        static IntegerType getI64Type(MLIRContext *context)
        {
            return IntegerType::get(context, 64);
        }

        static LLVM::LLVMPointerType getPointerType(mlir::Type type)
        {
            return LLVM::LLVMPointerType::get(type);
        }

        static LLVM::LLVMPointerType getI8PtrType(MLIRContext *context)
        {
            return getPointerType(getI8Type(context));
        }

        static LLVM::LLVMFunctionType getFunctionType(mlir::Type result, mlir::ResultRange::type_range arguments, bool isVarArg = false)
        {
            return LLVM::LLVMFunctionType::get(result, arguments, isVarArg);
        }

    protected:
        MLIRContext *context;
    };

    /// Lowers `typescript.print` to a loop nest calling `printf` on each of the individual
    /// elements of the array.
    class PrintOpLowering : public BaseConversionPattern
    {
    public:
        explicit PrintOpLowering(MLIRContext *context)
            : BaseConversionPattern(typescript::PrintOp::getOperationName(), 1, context) {}

        LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override
        {
            auto loc = op->getLoc();
            auto parentModule = op->getParentOfType<ModuleOp>();
            auto *context = parentModule.getContext();

            // Get a symbol reference to the printf function, inserting it if necessary.
            auto printfFuncOp =
                getOrInsertFunction(
                    rewriter,
                    parentModule,
                    "printf",
                    getFunctionType(getI32Type(context), getI8PtrType(context), true));

            Value formatSpecifierCst = getOrCreateGlobalString(
                loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);

            Value newLineCst = getOrCreateGlobalString(
                loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

            // print new line
            rewriter.create<LLVM::CallOp>(loc, printfFuncOp, newLineCst);

            // Notify the rewriter that this operation has been removed.
            rewriter.eraseOp(op);

            return success();
        }
    };

    struct AssertOpLowering : public BaseConversionPattern
    {
        explicit AssertOpLowering(MLIRContext *context)
            : BaseConversionPattern(typescript::AssertOp::getOperationName(), 1, context) {}

        LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override
        {
            // llvm::StringMap<pybind11::object>
            auto loc = op->getLoc();

            auto assertOp = cast<typescript::AssertOp>(op);

            auto line = 0;
            auto column = 0;
            auto fileName = StringRef("");
            TypeSwitch<LocationAttr>(loc)
                .Case<FileLineColLoc>([&](FileLineColLoc loc) {
                    fileName = loc.getFilename();
                    line = loc.getLine();
                    column = loc.getColumn();
                });

            auto parentModule = op->getParentOfType<ModuleOp>();
            auto *context = parentModule.getContext();
            typescript::AssertOp::Adaptor transformed(operands);

            // Insert the `_assert` declaration if necessary.
            auto i8PtrTy = getI8PtrType(context);
            auto assertFuncOp =
                getOrInsertFunction(
                    rewriter,
                    parentModule,
                    "_assert",
                    getFunctionType(getVoidType(context), {i8PtrTy, i8PtrTy, getI32Type(context)}));

            // Split block at `assert` operation.
            Block *opBlock = rewriter.getInsertionBlock();
            auto opPosition = rewriter.getInsertionPoint();
            Block *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

            // Generate IR to call `abort`.
            Block *failureBlock = rewriter.createBlock(opBlock->getParent());

            auto opHash = OperationEquivalence::computeHash(op, OperationEquivalence::Flags::IgnoreOperands);

            std::stringstream msgVarName;
            msgVarName << "m_" << opHash;

            std::stringstream msgWithNUL;
            msgWithNUL << assertOp.msg().str() << "\\0";

            std::stringstream fileVarName;
            fileVarName << "f_" << hash_value(fileName);

            std::stringstream fileWithNUL;
            fileWithNUL << fileName.str() << "\\0";

            auto msgCst = getOrCreateGlobalString(
                loc, rewriter, msgVarName.str(), msgWithNUL.str(), parentModule);

            auto fileCst = getOrCreateGlobalString(
                loc, rewriter, fileVarName.str(), fileWithNUL.str(), parentModule);

            //auto nullCst = rewriter.create<LLVM::NullOp>(loc, getI8PtrType(context));

            Value lineNumberRes = 
                rewriter.create<LLVM::ConstantOp>(
                    loc, 
                    getI32Type(context), 
                    rewriter.getI32IntegerAttr(line));

            rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{msgCst, fileCst, lineNumberRes});
            rewriter.create<LLVM::UnreachableOp>(loc);

            // Generate assertion test.
            rewriter.setInsertionPointToEnd(opBlock);
            rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
                op,
                transformed.arg(),
                continuationBlock,
                failureBlock);

            return success();
        }
    };

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TypeScriptToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace
{
    struct TypeScriptToLLVMLoweringPass
        : public PassWrapper<TypeScriptToLLVMLoweringPass, OperationPass<ModuleOp>>
    {
        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
        }
        void runOnOperation() final;
    };
} // end anonymous namespace

void TypeScriptToLLVMLoweringPass::runOnOperation()
{
    // The first thing to define is the conversion target. This will define the
    // final target for this lowering. For this lowering, we are only targeting
    // the LLVM dialect.
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    // During this lowering, we will also be lowering the MemRef types, that are
    // currently being operated on, to a representation in LLVM. To perform this
    // conversion we use a TypeConverter as part of the lowering. This converter
    // details how one type maps to another. This is necessary now that we will be
    // doing more complicated lowerings, involving loop region arguments.
    LLVMTypeConverter typeConverter(&getContext());

    // Now that the conversion target has been defined, we need to provide the
    // patterns used for lowering. At this point of the compilation process, we
    // have a combination of `typescript`, `affine`, and `std` operations. Luckily, there
    // are already exists a set of patterns to transform `affine` and `std`
    // dialects. These patterns lowering in multiple stages, relying on transitive
    // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
    // patterns must be applied to fully transform an illegal operation into a
    // set of legal ones.
    OwningRewritePatternList patterns;
    populateAffineToStdConversionPatterns(patterns, &getContext());
    populateLoopToStdConversionPatterns(patterns, &getContext());
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    // The only remaining operation to lower from the `typescript` dialect, is the
    // PrintOp.
    patterns.insert<PrintOpLowering, AssertOpLowering>(&getContext());

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations the remaining `TypeScript` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::typescript::createLowerToLLVMPass()
{
    return std::make_unique<TypeScriptToLLVMLoweringPass>();
}
