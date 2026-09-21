#include "mlir/Pass/Pass.h"

#include "TypeScript/Passes.h"
#include "TypeScript/Pass/ModulePass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pass"

namespace mlir_ts = mlir::typescript;

namespace
{

// the name upstream's lookupOrCreateAlignedAllocFn declares
constexpr auto ALIGNED_ALLOC_NAME = "aligned_alloc";

// Repairs what the upstream ConvertAsyncToLLVM pass leaves behind for a target whose pointers are
// narrower than 64 bits. That pass is part of the prebuilt LLVM, so it is fixed up here rather
// than patched. Scheduled only when the target's pointer width is below 64, so x64 never sees it.
class AsyncTargetWidthPass : public mlir::PassWrapper<AsyncTargetWidthPass, ModulePass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncTargetWidthPass)

    unsigned pointerBits;

    AsyncTargetWidthPass(unsigned pointerBits) : pointerBits(pointerBits)
    {
    }

    void runOnModule() override
    {
        if (mlir::failed(fixFrameAllocator(getModule())))
        {
            signalPassFailure();
        }
    }

    // Upstream ConvertAsyncToLLVM assumes 64-bit sizes: it allocates coroutine frames with
    // aligned_alloc(i64, i64) whatever the target. aligned_alloc takes size_t, which is pointer
    // width, so on a 32-bit target the callee reads (alignment, 0) and the frame overflows its
    // block. Retype the declaration to the pointer width and truncate the arguments at each call.
    // The values are a coroutine frame's size and alignment, which fit in 32 bits.
    //
    // Runs directly after ConvertAsyncToLLVM and before GCPass, which renames the declaration
    // (and its calls) to GC_memalign and so keeps the corrected signature. Under the other memory
    // models the name resolves to the runtime's aligned_alloc shim, which is (size_t, size_t) too.
    mlir::LogicalResult fixFrameAllocator(mlir::ModuleOp m)
    {
        auto funcOp = m.lookupSymbol<LLVM::LLVMFuncOp>(ALIGNED_ALLOC_NAME);
        if (!funcOp)
        {
            return mlir::success();
        }

        auto *context = m.getContext();
        auto pointerIntType = mlir::IntegerType::get(context, pointerBits);
        auto i64Type = mlir::IntegerType::get(context, 64);

        auto funcType = funcOp.getFunctionType();
        auto params = funcType.getParams();
        auto hasShape = [&](mlir::Type intType) {
            return funcOp.isExternal() && !funcType.isVarArg() && params.size() == 2 && params[0] == intType &&
                   params[1] == intType && isa<LLVM::LLVMPointerType>(funcType.getReturnType());
        };

        if (hasShape(pointerIntType))
        {
            return mlir::success();
        }

        if (!hasShape(i64Type))
        {
            return funcOp.emitError("async target width: unexpected aligned_alloc declaration ")
                   << funcType << ", expected an external (i64, i64) -> ptr";
        }

        llvm::SmallVector<LLVM::CallOp> calls;
        auto walkResult = m.walk([&](LLVM::CallOp callOp) {
            auto callee = callOp.getCallee();
            if (!callee.has_value() || callee.value() != ALIGNED_ALLOC_NAME)
            {
                return mlir::WalkResult::advance();
            }

            auto operands = callOp.getArgOperands();
            if (operands.size() != 2 || operands[0].getType() != i64Type || operands[1].getType() != i64Type)
            {
                callOp.emitError("async target width: unexpected aligned_alloc call, expected two i64 arguments");
                return mlir::WalkResult::interrupt();
            }

            calls.push_back(callOp);
            return mlir::WalkResult::advance();
        });

        if (walkResult.wasInterrupted())
        {
            return mlir::failure();
        }

        funcOp.setFunctionType(
            LLVM::LLVMFunctionType::get(funcType.getReturnType(), {pointerIntType, pointerIntType}));

        mlir::OpBuilder builder(context);
        for (auto callOp : calls)
        {
            builder.setInsertionPoint(callOp);
            auto operands = callOp.getArgOperands();
            auto alignment = builder.create<LLVM::TruncOp>(callOp.getLoc(), pointerIntType, operands[0]);
            auto size = builder.create<LLVM::TruncOp>(callOp.getLoc(), pointerIntType, operands[1]);
            callOp.getArgOperandsMutable().assign(mlir::ValueRange{alignment, size});
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! AsyncTargetWidthPass: retyped aligned_alloc to i" << pointerBits
                                << ", calls: " << calls.size() << "\n";);

        return mlir::success();
    }
};
} // end anonymous namespace

#undef DEBUG_TYPE

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createAsyncTargetWidthPass(unsigned pointerBits)
{
    return std::make_unique<AsyncTargetWidthPass>(pointerBits);
}
