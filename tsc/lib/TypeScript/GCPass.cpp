#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptPassContext.h"
#include "TypeScript/ModulePass.h"

#include "TypeScript/LowerToLLVMLogic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pass"

using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

class GCPass : public mlir::PassWrapper<GCPass, ModulePass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GCPass)

    TSContext tsContext;

    GCPass(CompileOptions &compileOptions) : tsContext(compileOptions)
    {
    }

    void runOnModule() override
    {
        auto m = getModule();

        m.walk([&](mlir::Operation *op) {
            if (auto funcOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(op))
            {
                auto symbolAttr = funcOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
                if (!symbolAttr)
                {
                    return;
                }

                auto name = std::string(symbolAttr.getValue());
                if (!funcOp.getBody().empty())
                {
                    if (name == "main")
                    {
                        injectInit(funcOp);
                    }

                    return;
                }

                renameFunction(name, funcOp);
            }

            if (auto callOp = dyn_cast_or_null<LLVM::CallOp>(op))
            {
                if (!callOp.getCallee().has_value())
                {
                    return;
                }

                auto name = callOp.getCallee().value();
                if (name == "memset")
                {
                    removeRedundantMemSet(callOp);

                    return;
                }

                renameCall(name, callOp);
            }
        });
    }

    bool mapName(StringRef name, StringRef modeName, StringRef &newName)
    {
        if (name == "malloc" || name == "calloc")
        {
            if (modeName == "atomic")
            {
                newName = "GC_malloc_atomic";    
            }
            else
            {
                newName = "GC_malloc";
            }
        }
        else if (name == "aligned_alloc")
        {
            newName = "GC_memalign";
        }
        else if (name == "realloc")
        {
            newName = "GC_realloc";
        }
        else if (name == "free")
        {
            newName = "GC_free";
        }
        else if (name == "aligned_free")
        {
            newName = "GC_free";
        }
        else
        {
            return false;
        }

        return true;
    }

    void renameFunction(StringRef name, LLVM::LLVMFuncOp funcOp)
    {
        StringRef newName;
        StringRef modeAttrValue;

        // this is function declaration
        if (!mapName(name, modeAttrValue, newName))
        {
            return;
        }

        funcOp->setAttr(SymbolTable::getSymbolAttrName(), mlir::StringAttr::get(funcOp->getContext(), newName));
    }

    void renameCall(StringRef name, LLVM::CallOp callOp)
    {
        StringRef newName;
        StringRef modeAttrValue;

        if (auto modeAttr = callOp->getAttr("mode").dyn_cast_or_null<mlir::StringAttr>())
        {
            modeAttrValue = modeAttr.getValue();
        }

        if (!mapName(name, modeAttrValue, newName))
        {
            return;
        }

        if (modeAttrValue == "atomic")
        {
            injectAtomicDeclaration(callOp);
        }

        callOp.setCalleeAttr(::mlir::FlatSymbolRefAttr::get(callOp->getContext(), newName));
    }

    void injectAtomicDeclaration(LLVM::CallOp memSetCallOp)
    {
        ConversionPatternRewriter rewriter(memSetCallOp.getContext());

        TypeHelper th(memSetCallOp.getContext());
        LLVMCodeHelper ch(memSetCallOp, rewriter, nullptr, tsContext.compileOptions);
        auto i8PtrTy = th.getI8PtrType();
        auto gcInitFuncOp = ch.getOrInsertFunction("GC_malloc_atomic", th.getFunctionType(th.getI8PtrType(), mlir::ArrayRef<mlir::Type>{th.getI64Type()}));
    }

    void injectInit(LLVM::LLVMFuncOp funcOp)
    {
        ConversionPatternRewriter rewriter(funcOp.getContext());

        TypeHelper th(rewriter.getContext());
        LLVMCodeHelper ch(funcOp, rewriter, nullptr, tsContext.compileOptions);
        auto i8PtrTy = th.getI8PtrType();
        auto gcInitFuncOp = ch.getOrInsertFunction("GC_init", th.getFunctionType(th.getVoidType(), mlir::ArrayRef<mlir::Type>{}));
        rewriter.create<LLVM::CallOp>(funcOp->getLoc(), gcInitFuncOp, ValueRange{});
    }

    void removeRedundantMemSet(LLVM::CallOp memSetCallOp)
    {
        // this is memset, find out if it is used by GC_malloc
        LLVM_DEBUG(llvm::dbgs() << "DBG: " << memSetCallOp.getOperand(0) << "\n";);
        if (auto probMemAllocCall = dyn_cast_or_null<LLVM::CallOp>(memSetCallOp.getOperand(0).getDefiningOp()))
        {
            if (!probMemAllocCall.getCallee().has_value())
            {
                return;
            }

            auto name = probMemAllocCall.getCallee().value();
            if (name == "GC_malloc")
            {
                ConversionPatternRewriter rewriter(memSetCallOp.getContext());
                rewriter.replaceOp(memSetCallOp, ValueRange{probMemAllocCall.getResult()});
            }
        }
    }
};
} // end anonymous namespace

#undef DEBUG_TYPE

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createGCPass(CompileOptions &compileOptions)
{
    return std::make_unique<GCPass>(compileOptions);
}
