#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.cpp.inc"

using namespace mlir;
using namespace mlir::typescript;

//===----------------------------------------------------------------------===//
// xxxxOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IdentifierReference
//===----------------------------------------------------------------------===//

IdentifierReference IdentifierReference::create(Location location, StringRef name)
{
    OperationState state(location, "identifier_reference");
    OpBuilder builder(location->getContext());
    IdentifierReference::build(builder, state, builder.getNoneType(), name);
    return IdentifierReference(Operation::create(state));
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

namespace
{
    struct EraseRedundantAssertions : public OpRewritePattern<AssertOp>
    {
        using OpRewritePattern<AssertOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(AssertOp op,
                                      PatternRewriter &rewriter) const override
        {
            // Erase assertion if argument is constant true.
            if (matchPattern(op.arg(), m_One()))
            {
                rewriter.eraseOp(op);
                return success();

            }
            return failure();
        }
    };
} // namespace

void AssertOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *context)
{
    patterns.insert<EraseRedundantAssertions>(context);
}
