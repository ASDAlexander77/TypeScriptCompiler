#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/DataStructs.h"

#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

// Set TSLANG_REPORT_STALE_LOWERING_CONTEXT to have each run report how many entries it drops from
// the run before (a stale entry that is looked up is always reported, and stops compilation).
inline bool reportStaleLoweringContext()
{
    static const bool report = std::getenv("TSLANG_REPORT_STALE_LOWERING_CONTEXT") != nullptr;
    return report;
}

// An entry written while lowering an earlier function was about to steer this one: the op it
// was written for is gone, and a new op took its address. Carrying on would branch into another
// function's blocks, so stop - in every build, since release is where it went unnoticed.
inline void reportStaleLoweringContextEntry(const char *table, Operation *op)
{
    llvm::errs() << "stale lowering-context entry: " << table << " for '" << op->getName() << "' at "
                 << op->getLoc() << "\n";
    llvm::report_fatal_error("lowering-context entry left by a previous function");
}

// A side table keyed by Operation*. Its entries belong to one run of a lowering pass (one
// function, or one module) and are dropped when the next run begins: the ops they were written
// for are gone by then, and ops created later reuse their addresses. When that happened, an op
// in the next function inherited an `unwind`/`cleanup` entry pointing into the previous
// function's blocks, and the verifier failed with "reference to block defined in another
// region" - rarely, and only where the allocator happened to hand the address back
// (00try_finally_return.ts on Windows CI). Each entry also remembers its run, so a lookup that
// still finds one from an earlier run reports it.
template <typename V> class OpSideTable
{
  public:
    explicit OpSideTable(const char *name) : name(name)
    {
    }

    void beginRun()
    {
        if (reportStaleLoweringContext() && !entries.empty())
        {
            llvm::errs() << "lowering-context: dropping " << entries.size() << " " << name
                         << " entries of the previous run\n";
        }

        entries.clear();
        run++;
    }

    // for writing: `table[op] = value`
    V &operator[](Operation *op)
    {
        auto &entry = entries[op];
        entry.run = run;
        return entry.value;
    }

    V lookup(Operation *op) const
    {
        auto it = entries.find(op);
        if (it == entries.end())
        {
            return V();
        }

        if (it->second.run != run && it->second.value)
        {
            reportStaleLoweringContextEntry(name, op);
        }

        return it->second.value;
    }

    bool contains(Operation *op) const
    {
        auto it = entries.find(op);
        if (it == entries.end())
        {
            return false;
        }

        if (it->second.run != run)
        {
            reportStaleLoweringContextEntry(name, op);
        }

        return true;
    }

  private:
    struct Entry
    {
        V value = V();
        unsigned run = 0;
    };

    const char *name;
    unsigned run = 0;
    mlir::DenseMap<Operation *, Entry> entries;
};

struct TSContext
{
    TSContext(CompileOptions &compileOptions) : compileOptions(compileOptions), returnBlock(nullptr) {};

    // call once before lowering each function (or module): entries written from here on belong to it
    void beginRun()
    {
        jumps.beginRun();
        catchOpData.beginRun();
        unwind.beginRun();
        cleanup.beginRun();
        parentTryOp.beginRun();
        landingBlockOf.beginRun();
        leavesCatch.beginRun();
    }

    // options
    CompileOptions &compileOptions;

    // name, break, continue
    OpSideTable<mlir::Block *> jumps{"jumps"};
    OpSideTable<mlir::Value> catchOpData{"catchOpData"};
    OpSideTable<mlir::Block *> unwind{"unwind"};
    OpSideTable<mlir::Block *> cleanup{"cleanup"};
    OpSideTable<Operation *> parentTryOp{"parentTryOp"};
    OpSideTable<mlir::Block *> landingBlockOf{"landingBlockOf"};
    // Throws that sit inside a catch clause and therefore have to end the active catch before
    // they leave it. `return`, `break` and `continue` carry the same meaning in `unwind`, but
    // a throw cannot: `unwind` already means its invoke destination, which is a different
    // question with a different answer.
    OpSideTable<bool> leavesCatch{"leavesCatch"};
    mlir::Block *returnBlock;
};

struct TSFunctionContext
{
    TSFunctionContext() = default;

    mlir::SmallVector<mlir::Block *> stateLabels;
};

template <typename OpTy> class TsPattern : public OpRewritePattern<OpTy>
{
  public:
    TsPattern<OpTy>(MLIRContext *context, TSContext *tsContext, TSFunctionContext *tsFuncContext, PatternBenefit benefit = 1)
        : OpRewritePattern<OpTy>::OpRewritePattern(context, benefit), tsContext(tsContext), tsFuncContext(tsFuncContext)
    {
    }

  protected:
    TSContext *tsContext;
    TSFunctionContext *tsFuncContext;
};

} // namespace
