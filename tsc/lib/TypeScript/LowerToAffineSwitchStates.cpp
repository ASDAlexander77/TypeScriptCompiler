#define DEBUG_TYPE "affine"

#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/TypeScriptPassContext.h"
#ifdef WIN_EXCEPTION
#include "TypeScript/MLIRLogic/MLIRRTTIHelperVCWin32.h"
#else
#include "TypeScript/MLIRLogic/MLIRRTTIHelperVCLinux.h"
#endif

#include "TypeScript/LowerToLLVMLogic.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

#include "scanner_enums.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

struct StateLabelOpLowering : public TsPattern<mlir_ts::StateLabelOp>
{
    using TsPattern<mlir_ts::StateLabelOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::StateLabelOp op, PatternRewriter &rewriter) const final
    {
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        rewriter.create<mlir::BranchOp>(op.getLoc(), continuationBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

class SwitchStateOpLowering : public TsPattern<mlir_ts::SwitchStateOp>
{
  public:
    using TsPattern<mlir_ts::SwitchStateOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::SwitchStateOp switchStateOp, PatternRewriter &rewriter) const final
    {
        auto loc = switchStateOp->getLoc();

        if (!tsContext->returnBlock)
        {
            CodeLogicHelper clh(switchStateOp, rewriter);
            tsContext->returnBlock = clh.FindReturnBlock(true);
        }

        assert(tsContext->returnBlock);

        auto defaultBlock = tsContext->returnBlock;

        assert(defaultBlock != nullptr);

        SmallVector<APInt> caseValues;
        SmallVector<mlir::Block *> caseDestinations;

        SmallPtrSet<Operation *, 16> stateLabels;

        auto index = 1;

        // select all states
        auto visitorAllStateLabels = [&](Operation *op) {
            if (auto stateLabelOp = dyn_cast_or_null<mlir_ts::StateLabelOp>(op))
            {
                stateLabels.insert(op);
            }
        };

        switchStateOp->getParentOp()->walk(visitorAllStateLabels);

        {
            mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
            for (auto op : stateLabels)
            {
                auto stateLabelOp = dyn_cast_or_null<mlir_ts::StateLabelOp>(op);
                rewriter.setInsertionPoint(stateLabelOp);

                auto *opBlock = rewriter.getInsertionBlock();
                auto opPosition = rewriter.getInsertionPoint();
                auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

                rewriter.setInsertionPointToEnd(opBlock);

                rewriter.create<mlir::BranchOp>(stateLabelOp.getLoc(), continuationBlock);

                rewriter.setInsertionPointToStart(continuationBlock);

                rewriter.eraseOp(stateLabelOp);

                // add switch
                caseValues.push_back(APInt(32, index++));
                caseDestinations.push_back(continuationBlock);
            }
        }

        // make switch to be terminator
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // insert 0 state label
        caseValues.insert(caseValues.begin(), APInt(32, 0));
        caseDestinations.insert(caseDestinations.begin(), continuationBlock);

        // switch
        rewriter.setInsertionPointToEnd(opBlock);

        rewriter.create<mlir::SwitchOp>(loc, switchStateOp.state(), defaultBlock ? defaultBlock : continuationBlock, ValueRange{},
                                        caseValues, caseDestinations);

        rewriter.eraseOp(switchStateOp);

        rewriter.setInsertionPointToStart(continuationBlock);

        return success();
    }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TypeScriptToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the typescript operations that are
/// computationally intensive (like add+mul for example...) while keeping the
/// rest of the code in the TypeScript dialect.

namespace
{
struct TypeScriptToAffineSwitchStatesLoweringPass : public PassWrapper<TypeScriptToAffineSwitchStatesLoweringPass, TypeScriptFunctionPass>
{
    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<StandardOpsDialect>();
    }

    void runOnFunction() final;

    TSContext tsContext;
};
} // end anonymous namespace.

void TypeScriptToAffineSwitchStatesLoweringPass::runOnFunction()
{
    auto function = getFunction();

    // We only lower the main function as we expect that all other functions have been inlined.
    if (function.getName() == "main")
    {
        auto voidType = mlir_ts::VoidType::get(function.getContext());
        // Verify that the given main has no inputs and results.
        if (function.getNumArguments() || llvm::any_of(function.getType().getResults(), [&](mlir::Type type) { return type != voidType; }))
        {
            function.emitError("expected 'main' to have 0 inputs and 0 results");
            return signalPassFailure();
        }
    }

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine` and `Standard` dialects.
    target.addLegalDialect<StandardOpsDialect>();

    // We also define the TypeScript dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the TypeScript operations that don't want
    // to lower, `typescript.print`, as `legal`.
    target.addIllegalDialect<mlir_ts::TypeScriptDialect>();
    target.addLegalOp<
        mlir_ts::AddressOfOp, mlir_ts::AddressOfConstStringOp, mlir_ts::AddressOfElementOp, mlir_ts::ArithmeticBinaryOp,
        mlir_ts::ArithmeticUnaryOp, mlir_ts::AssertOp, mlir_ts::CaptureOp, mlir_ts::CastOp, mlir_ts::ConstantOp, mlir_ts::ElementRefOp,
        mlir_ts::FuncOp, mlir_ts::GlobalOp, mlir_ts::GlobalResultOp, mlir_ts::HasValueOp, mlir_ts::ValueOp, mlir_ts::NullOp,
        mlir_ts::ParseFloatOp, mlir_ts::ParseIntOp, mlir_ts::PrintOp, mlir_ts::SizeOfOp, mlir_ts::StoreOp, mlir_ts::SymbolRefOp,
        mlir_ts::LengthOfOp, mlir_ts::StringLengthOp, mlir_ts::StringConcatOp, mlir_ts::StringCompareOp, mlir_ts::LoadOp, mlir_ts::NewOp,
        mlir_ts::CreateTupleOp, mlir_ts::DeconstructTupleOp, mlir_ts::CreateArrayOp, mlir_ts::NewEmptyArrayOp, mlir_ts::NewArrayOp,
        mlir_ts::DeleteOp, mlir_ts::PropertyRefOp, mlir_ts::InsertPropertyOp, mlir_ts::ExtractPropertyOp, mlir_ts::LogicalBinaryOp,
        mlir_ts::UndefOp, mlir_ts::VariableOp, mlir_ts::TrampolineOp, mlir_ts::InvokeOp, mlir_ts::ResultOp, mlir_ts::ThisVirtualSymbolRefOp,
        mlir_ts::InterfaceSymbolRefOp, mlir_ts::PushOp, mlir_ts::PopOp, mlir_ts::NewInterfaceOp, mlir_ts::VTableOffsetRefOp,
        mlir_ts::ThisPropertyRefOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp, mlir_ts::TypeOfOp, mlir_ts::DebuggerOp, mlir_ts::LandingPadOp,
        mlir_ts::CompareCatchTypeOp, mlir_ts::BeginCatchOp, mlir_ts::SaveCatchVarOp, mlir_ts::EndCatchOp, mlir_ts::ThrowUnwindOp,
        mlir_ts::ThrowCallOp, mlir_ts::CallInternalOp, mlir_ts::ReturnInternalOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TypeScript operations.
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<SwitchStateOpLowering, StateLabelOpLowering>(&getContext(), &tsContext);

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(function, target, std::move(patterns))))
    {
        signalPassFailure();
    }
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the TypeScript IR.
std::unique_ptr<mlir::Pass> mlir_ts::createLowerToAffineSwitchStatesPass()
{
    return std::make_unique<TypeScriptToAffineSwitchStatesLoweringPass>();
}
