#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

struct TSContext
{
    TSContext() = default;

    // name, break, continue
    mlir::DenseMap<Operation *, mlir::Block *> jumps;
    mlir::DenseMap<Operation *, mlir::Value> catchOpData;
    mlir::DenseMap<Operation *, mlir::Block *> unwind;
    mlir::Block *returnBlock;
};

template <typename OpTy> class TsPattern : public OpRewritePattern<OpTy>
{
  public:
    TsPattern<OpTy>(MLIRContext *context, TSContext *tsContext, PatternBenefit benefit = 1)
        : OpRewritePattern<OpTy>::OpRewritePattern(context, benefit), tsContext(tsContext)
    {
    }

  protected:
    TSContext *tsContext;
};

}