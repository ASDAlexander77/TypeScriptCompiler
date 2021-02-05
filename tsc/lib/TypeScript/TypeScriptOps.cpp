#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"

#define GET_TYPEDEF_CLASSES
#include "TypeScript/TypeScriptOpsTypes.cpp.inc"

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

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr)
    {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }

    FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
    if (!fn)
    {
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";
    }

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getType();
    if (fnType.getNumInputs() != getNumOperands())
    {
        return emitOpError("incorrect number of operands for callee");
    }

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    {
        if (getOperand(i).getType() != fnType.getInput(i))
        {
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided "
                   << getOperand(i).getType() << " for operand number " << i;
        }
    }

    if (fnType.getNumResults() != getNumResults())
    {
        return emitOpError("incorrect number of results for callee");
    }

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    {
        if (getResult(i).getType() != fnType.getResult(i))
        {
            return emitOpError("result type mismatch");
        }
    }

    return success();
}

FunctionType CallOp::getCalleeType()
{
    return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}
