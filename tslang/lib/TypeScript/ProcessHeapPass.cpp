#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Pass/ModulePass.h"
#include "TypeScript/Defines.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pass"

using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

// On Windows, under every memory model but gc: the module's malloc, calloc, realloc, free and
// aligned_alloc become the process-heap allocator of TypeScript/ProcessHeap.h. The C runtime's are
// not one allocator across modules - each links its own copy of the static CRT, a debug CRT keeps
// its blocks' list per copy, and with the prebuilt LLVM tslang.exe's `malloc` is rpmalloc - while a
// block made in one module is routinely freed in another (a library's object destroyed by the
// program, a coroutine frame from the runtime). gc has GCPass, which renames the same set to the
// collector's.
//
// Renamed, not rewritten: the helpers take what the C functions take. The names are not ones LLVM
// knows, so the allocating ones are marked as allocators, as GCPass marks GC_malloc - without that,
// two calls with equal arguments and nothing between them merge into one block.
class ProcessHeapPass : public mlir::PassWrapper<ProcessHeapPass, ModulePass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProcessHeapPass)

    void runOnModule() override
    {
        auto m = getModule();

        llvm::SmallVector<LLVM::LLVMFuncOp> declarations;
        m.walk([&](mlir::Operation *op) {
            if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op))
            {
                if (funcOp.getBody().empty() && !mapName(funcOp.getSymName()).empty())
                {
                    declarations.push_back(funcOp);
                }

                return;
            }

            if (auto callOp = dyn_cast<LLVM::CallOp>(op))
            {
                if (auto callee = callOp.getCallee())
                {
                    if (auto newName = mapName(*callee); !newName.empty())
                    {
                        callOp.setCalleeAttr(mlir::FlatSymbolRefAttr::get(op->getContext(), newName));
                    }
                }

                return;
            }

            // a function's address taken - a destructor slot, say
            if (auto addressOfOp = dyn_cast<LLVM::AddressOfOp>(op))
            {
                if (auto newName = mapName(addressOfOp.getGlobalName()); !newName.empty())
                {
                    addressOfOp.setGlobalNameAttr(mlir::FlatSymbolRefAttr::get(op->getContext(), newName));
                }
            }
        });

        for (auto funcOp : declarations)
        {
            auto newName = mapName(funcOp.getSymName());

            // `free` and `aligned_free` both become the one release; the first declaration stays
            if (auto existing = m.lookupSymbol<LLVM::LLVMFuncOp>(newName); existing && existing != funcOp)
            {
                funcOp.erase();
                continue;
            }

            funcOp.setSymName(newName);
            markAsAllocator(newName, funcOp);
        }
    }

  private:
    static llvm::StringRef mapName(llvm::StringRef name)
    {
        return llvm::StringSwitch<llvm::StringRef>(name)
            .Case("malloc", PROCESS_HEAP_MALLOC)
            .Case("calloc", PROCESS_HEAP_CALLOC)
            .Case("realloc", PROCESS_HEAP_REALLOC)
            .Case("free", PROCESS_HEAP_FREE)
            .Case("aligned_alloc", PROCESS_HEAP_ALIGNED_ALLOC)
            .Case("aligned_free", PROCESS_HEAP_FREE)
            .Default("");
    }

    // What LLVM knows of malloc, calloc, realloc and free by name, said of the helpers: they are
    // one allocator family, each allocation is a block of its own, the release takes the block
    // back, and the allocator's own state is memory nobody else reads. It is not only that two equal
    // allocations must not merge: optimization has to keep treating the program as it treated the C
    // functions - an allocation that is only released, say, is deleted together with its releases.
    static void markAsAllocator(llvm::StringRef name, LLVM::LLVMFuncOp funcOp)
    {
        // AllocFnKind: Alloc = 1 << 0, Realloc = 1 << 1, Free = 1 << 2, Uninitialized = 1 << 3,
        // Zeroed = 1 << 4
        uint64_t allocKind;
        auto takesBlock = false;
        if (name == PROCESS_HEAP_MALLOC || name == PROCESS_HEAP_ALIGNED_ALLOC)
        {
            allocKind = 1 | (1 << 3);
        }
        else if (name == PROCESS_HEAP_CALLOC)
        {
            allocKind = 1 | (1 << 4);
        }
        else if (name == PROCESS_HEAP_REALLOC)
        {
            allocKind = (1 << 1) | (1 << 3);
            takesBlock = true;
        }
        else if (name == PROCESS_HEAP_FREE)
        {
            allocKind = 1 << 2;
            takesBlock = true;
        }
        else
        {
            return;
        }

        auto *context = funcOp->getContext();

        // an allocation writes only the block it makes (other memory, as GCPass says it of
        // GC_malloc) and the allocator's state; a release or a resize reads and writes the block it
        // is given as well
        auto memoryEffects =
            takesBlock ? LLVM::MemoryEffectsAttr::get(context, LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::ModRef,
                                                      LLVM::ModRefInfo::ModRef, LLVM::ModRefInfo::NoModRef,
                                                      LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef)
                       : LLVM::MemoryEffectsAttr::get(context, LLVM::ModRefInfo::Mod, LLVM::ModRefInfo::NoModRef,
                                                      LLVM::ModRefInfo::ModRef, LLVM::ModRefInfo::NoModRef,
                                                      LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef);
        funcOp.setMemoryEffectsAttr(memoryEffects);

        if (takesBlock && funcOp.getNumArguments() > 0)
        {
            funcOp.setArgAttr(0, LLVM::LLVMDialect::getAllocatedPointerAttrName(), mlir::UnitAttr::get(context));
        }

        auto entry = [&](llvm::StringRef key, llvm::StringRef value) {
            return mlir::ArrayAttr::get(context, {mlir::StringAttr::get(context, key), mlir::StringAttr::get(context, value)});
        };

        llvm::SmallVector<mlir::Attribute> passthrough;
        if (auto existing = funcOp.getPassthroughAttr())
        {
            passthrough.append(existing.begin(), existing.end());
        }

        for (auto added : {entry("allockind", std::to_string(allocKind)), entry("alloc-family", "malloc")})
        {
            if (!llvm::is_contained(passthrough, added))
            {
                passthrough.push_back(added);
            }
        }

        funcOp.setPassthroughAttr(mlir::ArrayAttr::get(context, passthrough));
    }
};
} // end anonymous namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createProcessHeapPass()
{
    return std::make_unique<ProcessHeapPass>();
}
