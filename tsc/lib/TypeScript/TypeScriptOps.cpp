#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/Defines.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/FunctionImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "TypeScript/TypeScriptOpsTypes.cpp.inc"

static ::mlir::ParseResult parseFuncOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
static void print(::mlir::typescript::FuncOp op, ::mlir::OpAsmPrinter &p);
static ::mlir::LogicalResult verify(::mlir::typescript::FuncOp op);

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.cpp.inc"

using namespace mlir;
namespace ts = mlir::typescript;

/// Default callback for IfOp builders. Inserts a yield without arguments.
void ts::buildTerminatedBody(OpBuilder &builder, Location loc)
{
    builder.create<ts::YieldOp>(loc);
}

//===----------------------------------------------------------------------===//
// OptionalType
//===----------------------------------------------------------------------===//

Type ts::TypeScriptDialect::parseType(DialectAsmParser &parser) const
{
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    auto stringType = generatedTypeParser(getContext(), parser, "string");
    if (stringType != Type())
    {
        return stringType;
    }

    auto refType = generatedTypeParser(getContext(), parser, "ref");
    if (refType != Type())
    {
        return refType;
    }

    auto optionalType = generatedTypeParser(getContext(), parser, "optional");
    if (optionalType != Type())
    {
        return optionalType;
    }

    parser.emitError(typeLoc, "unknown type in TypeScript dialect");
    return Type();
}

void ts::TypeScriptDialect::printType(Type type, DialectAsmPrinter &os) const
{
    if (failed(generatedTypePrinter(type, os)))
    {
        llvm_unreachable("unexpected 'TypeScript' type kind");
    }
}

LogicalResult ts::OptionalType::verifyConstructionInvariants(Location loc, Type elementType)
{
    return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//
ts::FuncOp ts::FuncOp::create(Location location, StringRef name, FunctionType type,
                              ArrayRef<NamedAttribute> attrs)
{
    OperationState state(location, ts::FuncOp::getOperationName());
    OpBuilder builder(location->getContext());
    ts::FuncOp::build(builder, state, name, type, attrs);
    return cast<ts::FuncOp>(Operation::create(state));
}

ts::FuncOp ts::FuncOp::create(Location location, StringRef name, FunctionType type,
                              iterator_range<dialect_attr_iterator> attrs)
{
    SmallVector<NamedAttribute, 8> attrRef(attrs);
    return create(location, name, type, llvm::makeArrayRef(attrRef));
}

ts::FuncOp ts::FuncOp::create(Location location, StringRef name, FunctionType type,
                              ArrayRef<NamedAttribute> attrs,
                              ArrayRef<DictionaryAttr> argAttrs)
{
    auto func = create(location, name, type, attrs);
    func.setAllArgAttrs(argAttrs);
    return func;
}

void ts::FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                       FunctionType type, ArrayRef<NamedAttribute> attrs,
                       ArrayRef<DictionaryAttr> argAttrs)
{
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(name));
    state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();

    if (argAttrs.empty())
    {
        return;
    }

    assert(type.getNumInputs() == argAttrs.size());
    SmallString<8> argAttrName;
    for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
    {
        if (DictionaryAttr argDict = argAttrs[i])
        {
            state.addAttribute(getArgAttrName(i, argAttrName), argDict);
        }
    }
}

ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result)
{
    auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                            ArrayRef<Type> results, impl::VariadicFlag,
                            std::string &) {
        return builder.getFunctionType(argTypes, results);
    };

    return impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/false, buildFuncType);
}

void print(ts::FuncOp op, OpAsmPrinter &p)
{
    FunctionType fnType = op.getType();
    impl::printFunctionLikeOp(p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

LogicalResult verify(ts::FuncOp op)
{
    // If this function is external there is nothing to do.
    if (op.isExternal())
        return success();

    // Verify that the argument list of the function and the arg list of the entry
    // block line up.  The trait already verified that the number of arguments is
    // the same between the signature and the block.
    auto fnInputTypes = op.getType().getInputs();
    Block &entryBlock = op.front();
    for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
        if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
            return op.emitOpError("type of entry block argument #")
                   << i << '(' << entryBlock.getArgument(i).getType()
                   << ") must match the type of the corresponding argument in "
                   << "function signature(" << fnInputTypes[i] << ')';

    return success();
}

//===----------------------------------------------------------------------===//
// IdentifierReference
//===----------------------------------------------------------------------===//

ts::IdentifierReference ts::IdentifierReference::create(Location location, StringRef name)
{
    OperationState state(location, ts::IdentifierReference::getOperationName());
    OpBuilder builder(location->getContext());
    ts::IdentifierReference::build(builder, state, builder.getNoneType(), name);
    return cast<ts::IdentifierReference>(Operation::create(state));
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

namespace
{
    struct EraseRedundantAssertions : public OpRewritePattern<ts::AssertOp>
    {
        using OpRewritePattern<ts::AssertOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ts::AssertOp op,
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

void ts::AssertOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                               MLIRContext *context)
{
    patterns.insert<EraseRedundantAssertions>(context);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult ts::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr)
    {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }

    auto fn = symbolTable.lookupNearestSymbolFrom<ts::FuncOp>(*this, fnAttr);
    if (!fn)
    {
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";
    }

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getType();

    auto optionalFromValue = -1;
    auto optionalFrom = fn->getAttrOfType<IntegerAttr>(FUNC_OPTIONAL_ATTR_NAME);
    if (optionalFrom)
    {
        optionalFromValue = *optionalFrom.getValue().getRawData();
    }

    if (optionalFromValue == -1 && fnType.getNumInputs() != getNumOperands())
    {
        return emitOpError("incorrect number of operands for callee");
    }

    for (unsigned i = 0, e = optionalFromValue == -1 ? fnType.getNumInputs() : getOperands().size(); i != e; ++i)
    {
        if (getOperand(i).getType() != fnType.getInput(i))
        {
            /*
            OptionalType optType;
            TypeSwitch<Type>(fnType.getInput(i))
                .Case<OptionalType>([&](auto node) { optType = node; });

            if (!optType || optType.getElementType() != getOperand(i).getType())
            {
            */
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided "
                   << getOperand(i).getType() << " for operand number " << i;
            /*
            }
            */
        }
    }

    if (optionalFromValue == -1 && fnType.getNumResults() != getNumResults())
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

FunctionType ts::CallOp::getCalleeType()
{
    return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void ts::IfOp::build(OpBuilder &builder, OperationState &result, Value cond, bool withElseRegion)
{
    build(builder, result, /*resultTypes=*/llvm::None, cond, withElseRegion);
}

void ts::IfOp::build(OpBuilder &builder, OperationState &result, TypeRange resultTypes, Value cond, bool withElseRegion)
{
    auto addTerminator = [&](OpBuilder &nested, Location loc) {
        if (resultTypes.empty())
        {
            ts::IfOp::ensureTerminator(*nested.getInsertionBlock()->getParent(), nested, loc);
        }
    };

    build(builder, result, resultTypes, cond, addTerminator,
          withElseRegion ? addTerminator
                         : function_ref<void(OpBuilder &, Location)>());
}

void ts::IfOp::build(OpBuilder &builder, OperationState &result, TypeRange resultTypes, Value cond,
                     function_ref<void(OpBuilder &, Location)> thenBuilder,
                     function_ref<void(OpBuilder &, Location)> elseBuilder)
{
    assert(thenBuilder && "the builder callback for 'then' must be present");

    result.addOperands(cond);
    result.addTypes(resultTypes);

    OpBuilder::InsertionGuard guard(builder);
    Region *thenRegion = result.addRegion();
    builder.createBlock(thenRegion);
    thenBuilder(builder, result.location);

    Region *elseRegion = result.addRegion();
    if (!elseBuilder)
        return;

    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
}

void ts::IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                     function_ref<void(OpBuilder &, Location)> thenBuilder,
                     function_ref<void(OpBuilder &, Location)> elseBuilder)
{
    build(builder, result, TypeRange(), cond, thenBuilder, elseBuilder);
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op, Region &region, ValueRange blockArgs = {})
{
    assert(llvm::hasSingleElement(region) && "expected single-region block");
    Block *block = &region.front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.mergeBlockBefore(block, op, blockArgs);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ts::IfOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions)
{
    // The `then` and the `else` region branch back to the parent operation.
    if (index.hasValue())
    {
        regions.push_back(RegionSuccessor(getResults()));
        return;
    }

    // Don't consider the else region if it is empty.
    Region *elseRegion = &this->elseRegion();
    if (elseRegion->empty())
    {
        elseRegion = nullptr;
    }

    // Otherwise, the successor is dependent on the condition.
    bool condition;
    if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
    {
        condition = condAttr.getValue().isOneValue();
    }
    else
    {
        // If the condition isn't constant, both regions may be executed.
        regions.push_back(RegionSuccessor(&thenRegion()));
        regions.push_back(RegionSuccessor(elseRegion));
        return;
    }

    // Add the successor regions using the condition.
    regions.push_back(RegionSuccessor(condition ? &thenRegion() : elseRegion));
}

namespace
{
    // Pattern to remove unused IfOp results.
    struct RemoveUnusedResults : public OpRewritePattern<ts::IfOp>
    {
        using OpRewritePattern<ts::IfOp>::OpRewritePattern;

        void transferBody(Block *source, Block *dest, ArrayRef<OpResult> usedResults,
                          PatternRewriter &rewriter) const
        {
            // Move all operations to the destination block.
            rewriter.mergeBlocks(source, dest);
            // Replace the yield op by one that returns only the used values.
            auto yieldOp = cast<ts::YieldOp>(dest->getTerminator());
            SmallVector<Value, 4> usedOperands;
            llvm::transform(
                usedResults,
                std::back_inserter(usedOperands),
                [&](OpResult result) {
                    return yieldOp.getOperand(result.getResultNumber());
                });
            rewriter.updateRootInPlace(yieldOp, [&]() { yieldOp->setOperands(usedOperands); });
        }

        LogicalResult matchAndRewrite(ts::IfOp op, PatternRewriter &rewriter) const override
        {
            // Compute the list of used results.
            SmallVector<OpResult, 4> usedResults;
            llvm::copy_if(
                op.getResults(),
                std::back_inserter(usedResults),
                [](OpResult result) { return !result.use_empty(); });

            // Replace the operation if only a subset of its results have uses.
            if (usedResults.size() == op.getNumResults())
            {
                return failure();
            }

            // Compute the result types of the replacement operation.
            SmallVector<Type, 4> newTypes;
            llvm::transform(
                usedResults,
                std::back_inserter(newTypes),
                [](OpResult result) { return result.getType(); });

            // Create a replacement operation with empty then and else regions.
            auto emptyBuilder = [](OpBuilder &, Location) {};
            auto newOp = rewriter.create<ts::IfOp>(
                op.getLoc(),
                newTypes,
                op.condition(),
                emptyBuilder,
                emptyBuilder);

            // Move the bodies and replace the terminators (note there is a then and
            // an else region since the operation returns results).
            transferBody(op.getBody(0), newOp.getBody(0), usedResults, rewriter);
            transferBody(op.getBody(1), newOp.getBody(1), usedResults, rewriter);

            // Replace the operation by the new one.
            SmallVector<Value, 4> repResults(op.getNumResults());
            for (auto en : llvm::enumerate(usedResults))
            {
                repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
            }

            rewriter.replaceOp(op, repResults);
            return success();
        }
    };

    struct RemoveStaticCondition : public OpRewritePattern<ts::IfOp>
    {
        using OpRewritePattern<ts::IfOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ts::IfOp op, PatternRewriter &rewriter) const override
        {
            auto constant = op.condition().getDefiningOp<mlir::ConstantOp>();
            if (!constant)
            {
                return failure();
            }

            if (constant.getValue().cast<BoolAttr>().getValue())
            {
                replaceOpWithRegion(rewriter, op, op.thenRegion());
            }
            else if (!op.elseRegion().empty())
            {
                replaceOpWithRegion(rewriter, op, op.elseRegion());
            }
            else
            {
                rewriter.eraseOp(op);
            }

            return success();
        }
    };
} // namespace

void ts::IfOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnusedResults, RemoveStaticCondition>(context);
}
