#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptPassContext.h"
#include "TypeScript/Pass/ModulePass.h"

#include "TypeScript/LowerToLLVMLogic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pass"

using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

// what LLVMCodeHelperBase::_MemoryAlloc asks for when it wants a zeroed block
constexpr auto CALLOC_NAME = "calloc";

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

        LLVM_DEBUG(llvm::dbgs() << "\n!! GCPass: BEFORE DUMP: \n" << m << "\n";);

        auto added = false;
        llvm::SmallVector<LLVM::MemsetOp> redundantMemSets;
        llvm::SmallVector<LLVM::CallOp> callocCalls;
        llvm::SmallVector<LLVM::LLVMFuncOp> callocDecls;
        m.walk([&](mlir::Operation *op) {
            // process gctors first
            if (auto funcOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(op))
            {
                auto symbolAttr = funcOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
                if (!symbolAttr)
                {
                    return;
                }

                auto name = std::string(symbolAttr.getValue());
                if (name == CALLOC_NAME)
                {
                    callocDecls.push_back(funcOp);
                    return;
                }

                if (!funcOp.getBody().empty())
                {
                    if (!added)
                    {
                        // we are adding to gctos(as method - only)
                        if (StringRef(name).starts_with(MLIR_GCTORS))
                        {
                            added = true;
                            injectInit(funcOp);
                        }
                    }

                    return;
                }

                renameFunction(name, funcOp);
            }

            if (auto memsetOp = dyn_cast_or_null<LLVM::MemsetOp>(op))
            {
                if (zeroesAGCAllocation(memsetOp))
                {
                    // erased after the walk, not during it
                    redundantMemSets.push_back(memsetOp);
                }

                return;
            }

            if (auto callOp = dyn_cast_or_null<LLVM::CallOp>(op))
            {
                if (!callOp.getCallee().has_value())
                {
                    return;
                }

                auto name = callOp.getCallee().value();
                if (name == CALLOC_NAME)
                {
                    callocCalls.push_back(callOp);
                    return;
                }

                renameCall(name, callOp);
            }
        });

        for (auto memsetOp : redundantMemSets)
        {
            memsetOp.erase();
        }

        replaceCallocWithGCMalloc(m, callocCalls, callocDecls);

        if (!added)
        {
            // process main
            if (auto funcOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(m.lookupSymbol(MAIN_ENTRY_NAME)))
            {
                if (!funcOp.getBody().empty())
                {
                    added = true;
                    injectInit(funcOp);
                }
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! GCPass: AFTER DUMP: \n" << m << "\n";);
    }

    // `calloc` is deliberately absent here: it takes two arguments where GC_malloc takes one, so
    // a rename in place would leave a call whose arity disagrees with its callee. It goes through
    // replaceCallocWithGCMalloc instead.
    bool mapName(StringRef name, StringRef modeName, StringRef &newName)
    {
        if (name == "malloc")
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

        markAsAllocatorIfNeeded(newName, funcOp);
    }

    // GC_malloc/GC_malloc_atomic/GC_memalign are fresh-pointer-per-call allocators, just like
    // plain malloc. Unlike malloc, they aren't libc names LLVM's TargetLibraryInfo recognizes,
    // so without explicit markings, GVN/EarlyCSE at -O3 see two calls with identical arguments
    // (e.g. GC_malloc(0) for two different empty array fields) and no intervening memory
    // clobber, and fold them into one shared allocation - aliasing fields that must stay
    // distinct. memory_effects alone (writing unmodeled "other" memory) is not enough to stop
    // GVN's call-CSE, which special-cases allocator-shaped functions via the `allockind` LLVM
    // attribute (the same mechanism TargetLibraryInfo uses internally for malloc/calloc). Since
    // GC_malloc isn't a recognized libc name, we must attach `allockind("alloc")` explicitly so
    // each call is treated as returning a distinct, non-aliasing pointer.
    void markAsAllocatorIfNeeded(StringRef newName, LLVM::LLVMFuncOp funcOp)
    {
        if (newName != "GC_malloc" && newName != "GC_malloc_atomic" && newName != "GC_memalign")
        {
            return;
        }

        auto *context = funcOp->getContext();
        auto memoryEffects = LLVM::MemoryEffectsAttr::get(context, LLVM::ModRefInfo::Mod, LLVM::ModRefInfo::NoModRef,
                                                            LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
                                                            LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef);
        funcOp.setMemoryEffectsAttr(memoryEffects);

        // AllocFnKind::Alloc = 1<<0, Zeroed = 1<<4. GC_malloc/GC_malloc_atomic zero-fill;
        // GC_memalign (GC_memalign) does not guarantee zeroing, so only mark Alloc for it.
        uint64_t allocKind = newName == "GC_memalign" ? /*Alloc*/ 1 : /*Alloc|Zeroed*/ 1 | (1 << 4);
        auto kindEntry = mlir::ArrayAttr::get(
            context, {mlir::StringAttr::get(context, "allockind"),
                      mlir::StringAttr::get(context, std::to_string(allocKind))});

        llvm::SmallVector<mlir::Attribute> passthrough;
        if (auto existing = funcOp.getPassthroughAttr())
        {
            // one declaration, many call sites: this runs once per call for the injected
            // declarations, and a second `allockind` entry would be a duplicate LLVM attribute
            if (llvm::is_contained(existing, kindEntry))
            {
                return;
            }

            passthrough.append(existing.begin(), existing.end());
        }
        passthrough.push_back(kindEntry);
        funcOp.setPassthroughAttr(mlir::ArrayAttr::get(context, passthrough));
    }

    void renameCall(StringRef name, LLVM::CallOp callOp)
    {
        StringRef newName;
        StringRef modeAttrValue;

        if (auto modeAttr = dyn_cast_or_null<mlir::StringAttr>(callOp->getAttr("mode")))
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
        PatternRewriter rewriter(memSetCallOp.getContext());

        TypeHelper th(memSetCallOp.getContext());
        LLVMCodeHelper ch(memSetCallOp, rewriter, nullptr, tsContext.compileOptions);
        auto i8PtrTy = th.getPtrType();
        auto gcInitFuncOp = ch.getOrInsertFunction("GC_malloc_atomic", th.getFunctionType(th.getPtrType(), mlir::ArrayRef<mlir::Type>{th.getI64Type()}));
        markAsAllocatorIfNeeded("GC_malloc_atomic", gcInitFuncOp);
    }

    // `calloc(1, n)` is how a zeroed block is asked for (LLVMCodeHelperBase::_MemoryAlloc), and
    // GC_malloc already returns zeroed memory - so the count argument is dropped and the size
    // handed straight over. Rewritten rather than renamed because the arity differs; a rename in
    // place would leave a two-argument call to a one-argument callee.
    //
    // The declarations go too. Every `calloc` in the module came from _MemoryAlloc, so once the
    // calls are gone nothing names them, and a declaration left behind would make the linked
    // program depend on libc's allocator for a symbol it never calls.
    void replaceCallocWithGCMalloc(mlir::ModuleOp module, llvm::SmallVector<LLVM::CallOp> &calls,
                                   llvm::SmallVector<LLVM::LLVMFuncOp> &decls)
    {
        if (calls.empty() && decls.empty())
        {
            return;
        }

        PatternRewriter rewriter(module.getContext());
        TypeHelper th(module.getContext());

        for (auto callOp : calls)
        {
            LLVMCodeHelper ch(callOp, rewriter, nullptr, tsContext.compileOptions);
            auto sizeValue = callOp.getOperand(1);
            auto gcMallocFuncOp = ch.getOrInsertFunction(
                "GC_malloc", th.getFunctionType(th.getPtrType(), mlir::ArrayRef<mlir::Type>{sizeValue.getType()}));
            markAsAllocatorIfNeeded("GC_malloc", gcMallocFuncOp);

            rewriter.setInsertionPoint(callOp);
            auto gcMallocCall = rewriter.create<LLVM::CallOp>(callOp->getLoc(), gcMallocFuncOp, ValueRange{sizeValue});
            rewriter.replaceOp(callOp, gcMallocCall.getResults());
        }

        for (auto funcOp : decls)
        {
            funcOp.erase();
        }
    }

    void injectInit(LLVM::LLVMFuncOp funcOp)
    {
        PatternRewriter rewriter(funcOp.getContext());

        TypeHelper th(rewriter.getContext());
        LLVMCodeHelper ch(funcOp, rewriter, nullptr, tsContext.compileOptions);
        auto i8PtrTy = th.getPtrType();
        auto gcInitFuncOp = ch.getOrInsertFunction("GC_init", th.getFunctionType(th.getVoidType(), mlir::ArrayRef<mlir::Type>{}));

        rewriter.setInsertionPointToStart(&*funcOp.getBody().begin());
        rewriter.create<LLVM::CallOp>(funcOp->getLoc(), gcInitFuncOp, ValueRange{});
    }

    // GC_malloc hands back zeroed memory, so zeroing the block it just returned is wasted work.
    bool zeroesAGCAllocation(LLVM::MemsetOp memSetOp)
    {
        LLVM_DEBUG(llvm::dbgs() << "DBG: " << memSetOp.getDst() << "\n";);
        auto probMemAllocCall = dyn_cast_or_null<LLVM::CallOp>(memSetOp.getDst().getDefiningOp());
        if (!probMemAllocCall || !probMemAllocCall.getCallee().has_value())
        {
            return false;
        }

        // The allocation is renamed before this runs - the walk reaches it first, since it
        // defines the pointer being zeroed - so the name to match is the GC one.
        return probMemAllocCall.getCallee().value() == "GC_malloc";
    }
};
} // end anonymous namespace

#undef DEBUG_TYPE

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createGCPass(CompileOptions &compileOptions)
{
    return std::make_unique<GCPass>(compileOptions);
}
