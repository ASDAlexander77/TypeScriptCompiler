#define DEBUG_TYPE "mlir"

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

/// Default callback for IfOp builders. Inserts a yield without arguments.
void mlir_ts::buildTerminatedBody(OpBuilder &builder, Location loc)
{
    builder.create<mlir_ts::ResultOp>(loc);
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OptionalType
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EnumType
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstArrayType
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// ConstTupleType
//===----------------------------------------------------------------------===//

/// Return the number of element types.
size_t mlir_ts::ConstTupleType::size() const
{
    return getFields().size();
}

//===----------------------------------------------------------------------===//
/// TupleType
//===----------------------------------------------------------------------===//

/// Return the number of element types.
size_t mlir_ts::TupleType::size() const
{
    return getFields().size();
}

//===----------------------------------------------------------------------===//
/// ClassStorageType
//===----------------------------------------------------------------------===//

/// Return the number of element types.
size_t mlir_ts::ClassStorageType::size() const
{
    return getFields().size();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
OpFoldResult mlir_ts::ConstantOp::fold(ArrayRef<Attribute> operands)
{
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

namespace
{
template <typename T> struct RemoveUnused : public OpRewritePattern<T>
{
    using OpRewritePattern<T>::OpRewritePattern;

    LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const override
    {
        if (op->getResult(0).use_empty())
        {
            rewriter.eraseOp(op);
        }

        return success();
    }
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// UnresolvedSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::UnresolvedSymbolRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::UnresolvedSymbolRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// SymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::SymbolRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::SymbolRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// ThisSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ThisSymbolRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ThisSymbolRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// ThisVirtualSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ThisVirtualSymbolRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ThisVirtualSymbolRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// AccessorRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::AccessorRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::AccessorRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// ThisAccessorRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ThisAccessorRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::AccessorRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// InterfaceSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::InterfaceSymbolRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::InterfaceSymbolRefOp>>(context);
}

void mlir_ts::InterfaceSymbolRefOp::getAsmResultNames(llvm::function_ref<void(Value, StringRef)> setNameFn)
{
    SmallString<4> buffer;
    llvm::raw_svector_ostream os(buffer);
    os << "func";
    setNameFn(getResult(0), os.str());
    SmallString<4> buffer2;
    llvm::raw_svector_ostream os2(buffer2);
    os2 << "this";
    setNameFn(getResult(1), os2.str());
}

//===----------------------------------------------------------------------===//
// TypeRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::TypeRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::TypeRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// ClassRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ClassRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ClassRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// InterfaceRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::InterfaceRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::InterfaceRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// NamespaceRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::NamespaceRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::NamespaceRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void mlir_ts::LoadOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::LoadOp>>(context);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult verify(mlir_ts::CastOp op)
{
    // funcType -> funcType
    auto inFuncType = op.in().getType().dyn_cast_or_null<mlir::FunctionType>();
    auto resFuncType = op.res().getType().dyn_cast_or_null<mlir::FunctionType>();
    if (inFuncType && resFuncType)
    {
        if (inFuncType.getInputs().size() != resFuncType.getInputs().size())
        {
            return op.emitOpError("can't cast function type to other function type with different count of parameters ")
                   << '(' << inFuncType << ") must match the "
                   << "function type(" << resFuncType << ')';
        }

        for (unsigned i = 0, e = inFuncType.getInputs().size(); i != e; ++i)
        {
            if (inFuncType.getInput(i) != resFuncType.getInput(i))
            {
                return op.emitOpError("can't cast function type to other function type with different parameters #")
                       << i << '(' << inFuncType.getInput(i) << ") must match the type of the corresponding argument in "
                       << "function argument(" << resFuncType.getInput(i) << ')';
            }
        }

        auto inRetCount = inFuncType.getResults().size();
        auto resRetCount = resFuncType.getResults().size();

        auto noneType = mlir::NoneType::get(op.getContext());
        auto voidType = mlir_ts::VoidType::get(op.getContext());

        for (auto retType : inFuncType.getResults())
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                inRetCount--;
            }
        }

        for (auto retType : resFuncType.getResults())
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                resRetCount--;
            }
        }

        if (inRetCount != resRetCount)
        {
            return op.emitOpError("can't cast function type to other function type with different count of returns ")
                   << '(' << inFuncType << ") must match the "
                   << "function type(" << resFuncType << ')';
        }

        for (unsigned i = 0, e = inFuncType.getResults().size(); i != e; ++i)
        {
            auto inRetType = inFuncType.getResult(i);
            auto resRetType = resFuncType.getResult(i);

            auto isInVoid = !inRetType || inRetType == noneType || inRetType == voidType;
            auto isResVoid = !resRetType || resRetType == noneType || resRetType == voidType;
            if (!isInVoid && !isResVoid && inRetType != resRetType)
            {
                return op.emitOpError("can't cast function type to other function type with different return types #")
                       << i << '(' << inFuncType.getResult(i) << ") must match the return type of the corresponding return in "
                       << "function return(" << resFuncType.getResult(i) << ')';
            }
        }
    }

    return success();
}

namespace
{
struct NormalizeCast : public OpRewritePattern<mlir_ts::CastOp>
{
    using OpRewritePattern<mlir_ts::CastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir_ts::CastOp castOp, PatternRewriter &rewriter) const override
    {
        // TODO: finish it
        auto in = castOp.in();
        auto res = castOp.res();

        if (in.getType() == res.getType())
        {
            rewriter.replaceOp(castOp, in);
            return success();
        }

        if (auto stringType = res.getType().isa<mlir_ts::StringType>())
        {
            if (auto classType = in.getType().dyn_cast_or_null<mlir_ts::ClassType>())
            {
                auto className = classType.getName().getValue();
                auto fullToStringName = className + ".toString";
                auto stringType = mlir_ts::StringType::get(rewriter.getContext());
                auto callRes =
                    rewriter.create<mlir_ts::CallOp>(castOp->getLoc(), fullToStringName.str(), TypeRange{stringType}, ValueRange{in});
                auto res = callRes.getResult(0);
                rewriter.replaceOp(castOp, res);
                return success();
            }
        }

        // null -> interface cast
        auto anyType = in.getType().dyn_cast_or_null<mlir_ts::AnyType>();
        auto interfaceType = res.getType().dyn_cast_or_null<mlir_ts::InterfaceType>();
        if (anyType && interfaceType)
        {
            if (auto nullOp = in.getDefiningOp<mlir_ts::NullOp>())
            {
                auto undef = rewriter.create<mlir_ts::UndefOp>(castOp->getLoc(), interfaceType);
                rewriter.replaceOp(castOp, ValueRange{undef});
                rewriter.eraseOp(nullOp);
                return success();
            }
        }

        return success();
    }
};
} // end anonymous namespace.

void mlir_ts::CastOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<NormalizeCast>(context);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

mlir_ts::FuncOp mlir_ts::FuncOp::create(Location location, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs)
{
    OperationState state(location, mlir_ts::FuncOp::getOperationName());
    OpBuilder builder(location->getContext());
    mlir_ts::FuncOp::build(builder, state, name, type, attrs);
    return cast<mlir_ts::FuncOp>(Operation::create(state));
}

mlir_ts::FuncOp mlir_ts::FuncOp::create(Location location, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs,
                                        ArrayRef<DictionaryAttr> argAttrs)
{
    auto func = create(location, name, type, attrs);
    func.setAllArgAttrs(argAttrs);
    return func;
}

void mlir_ts::FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs,
                            ArrayRef<DictionaryAttr> argAttrs)
{
    state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();

    if (argAttrs.empty())
    {
        return;
    }

    assert(type.getNumInputs() == argAttrs.size());
    function_like_impl::addArgAndResultAttrs(builder, state, argAttrs, /*resultAttrs=*/llvm::None);
}

LogicalResult verify(mlir_ts::FuncOp op)
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
                   << i << '(' << entryBlock.getArgument(i).getType() << ") must match the type of the corresponding argument in "
                   << "function signature(" << fnInputTypes[i] << ')';

    return success();
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

LogicalResult verify(mlir_ts::InvokeOp op)
{
    if (op.getNumResults() > 1)
    {
        return op.emitOpError("must have 0 or 1 result");
    }

    Block *unwindDest = op.unwindDest();
    if (unwindDest->empty())
    {
        return op.emitError("must have at least one operation in unwind destination");
    }

    // In unwind destination, first operation must be LandingpadOp
    /*
    if (!isa<LandingpadOp>(unwindDest->front()))
    {
        return op.emitError("first operation in unwind destination should be a "
                            "llvm.landingpad operation");
    }
    */

    return success();
}

Optional<MutableOperandRange> mlir_ts::InvokeOp::getMutableSuccessorOperands(unsigned index)
{
    assert(index < getNumSuccessors() && "invalid successor index");
    return index == 0 ? normalDestOperandsMutable() : unwindDestOperandsMutable();
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

namespace
{
struct EraseRedundantAssertions : public OpRewritePattern<mlir_ts::AssertOp>
{
    using OpRewritePattern<mlir_ts::AssertOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir_ts::AssertOp op, PatternRewriter &rewriter) const override
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

void mlir_ts::AssertOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns, MLIRContext *context)
{
    patterns.insert<EraseRedundantAssertions>(context);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult mlir_ts::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr)
    {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }

    auto fn = symbolTable.lookupNearestSymbolFrom<mlir_ts::FuncOp>(*this, fnAttr);
    if (!fn)
    {
        return emitOpError() << "'" << fnAttr.getValue() << "' does not reference a valid function";
    }

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getType();

    auto optionalFromValue = (int)fnType.getNumInputs() - (int)getNumOperands();
    for (unsigned i = 0, e = optionalFromValue == -1 ? fnType.getNumInputs() : getOperands().size(); i != e; ++i)
    {
        if (auto symRef = dyn_cast_or_null<mlir_ts::SymbolRefOp>(getOperand(i).getDefiningOp()))
        {
            auto val = symbolTable.lookupNearestSymbolFrom(*this, symRef.identifierAttr());
            if (!val)
            {
                return emitOpError() << "can't find variable:  '" << symRef.identifierAttr() << "'";
            }
        }

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
                   << fnType.getInput(i) << ", but provided " << getOperand(i).getType() << " for operand number " << i;
            /*
            }
            */
        }
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

FunctionType mlir_ts::CallOp::getCalleeType()
{
    return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult mlir_ts::CallIndirectOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Verify that the operand and result types match the callee.
    auto fnType = getCallee().getType().cast<mlir::FunctionType>();

    auto optionalFromValue = (int)fnType.getNumInputs() - (int)getNumOperands();
    for (unsigned i = 0, e = optionalFromValue == -1 ? fnType.getNumInputs() : getOperands().size(); i != e; ++i)
    {
        if (auto symRef = dyn_cast_or_null<mlir_ts::SymbolRefOp>(getOperand(i + 1).getDefiningOp()))
        {
            auto val = symbolTable.lookupNearestSymbolFrom(*this, symRef.identifierAttr());
            if (!val)
            {
                return emitOpError() << "can't find variable:  '" << symRef.identifierAttr() << "'";
            }
        }

        if (getOperand(i + 1).getType() != fnType.getInput(i))
        {
            /*
            OptionalType optType;
            TypeSwitch<Type>(fnType.getInput(i))
                .Case<OptionalType>([&](auto node) { optType = node; });

            if (!optType || optType.getElementType() != getOperand(i).getType())
            {
            */
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided " << getOperand(i).getType() << " for operand number " << i;
            /*
            }
            */
        }
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

namespace
{
/// Fold indirect calls that have a constant function as the callee operand.
struct SimplifyIndirectCallWithKnownCallee : public OpRewritePattern<mlir_ts::CallIndirectOp>
{
    using OpRewritePattern<mlir_ts::CallIndirectOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir_ts::CallIndirectOp indirectCall, PatternRewriter &rewriter) const override
    {
        // Check that the callee is a constant callee.
        if (auto symbolRefOp = indirectCall.getCallee().getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            // Replace with a direct call.
            rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, symbolRefOp.identifierAttr(), indirectCall.getResultTypes(),
                                                         indirectCall.getArgOperands());
            return success();
        }

        if (auto thisSymbolRefOp = indirectCall.getCallee().getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            // Replace with a direct call.
            rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, thisSymbolRefOp.identifierAttr(), indirectCall.getResultTypes(),
                                                         indirectCall.getArgOperands());
            return success();
        }

        if (auto trampolineOp = indirectCall.getCallee().getDefiningOp<mlir_ts::TrampolineOp>())
        {
            if (auto symbolRefOp = trampolineOp.callee().getDefiningOp<mlir_ts::SymbolRefOp>())
            {
                // Replace with a direct call.
                SmallVector<mlir::Value> args;
                args.push_back(trampolineOp.data_reference());
                args.append(indirectCall.getArgOperands().begin(), indirectCall.getArgOperands().end());
                rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, symbolRefOp.identifierAttr(), indirectCall.getResultTypes(),
                                                             args);

                LLVM_DEBUG(for (auto &arg : args) { llvm::dbgs() << "\n\n SimplifyIndirectCallWithKnownCallee arg: " << arg << "\n"; });

                LLVM_DEBUG(llvm::dbgs() << "\nSimplifyIndirectCallWithKnownCallee: args: " << args.size() << "\n";);
                LLVM_DEBUG(for (auto &use
                                : trampolineOp->getUses()) { llvm::dbgs() << "\n use number:" << use.getOperandNumber() << "\n"; });

                if (trampolineOp.use_empty())
                {
                    rewriter.eraseOp(trampolineOp);
                }

                return success();
            }
        }

        return failure();
    }
};
} // end anonymous namespace.

void mlir_ts::CallIndirectOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<SimplifyIndirectCallWithKnownCallee>(context);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void mlir_ts::IfOp::build(OpBuilder &builder, OperationState &result, Value cond, bool withElseRegion)
{
    build(builder, result, /*resultTypes=*/llvm::None, cond, withElseRegion);
}

void mlir_ts::IfOp::build(OpBuilder &builder, OperationState &result, TypeRange resultTypes, Value cond, bool withElseRegion)
{
    auto addTerminator = [&](OpBuilder &nested, Location loc) {
        if (resultTypes.empty())
        {
            mlir_ts::IfOp::ensureTerminator(*nested.getInsertionBlock()->getParent(), nested, loc);
        }
    };

    build(builder, result, resultTypes, cond, addTerminator, withElseRegion ? addTerminator : function_ref<void(OpBuilder &, Location)>());
}

void mlir_ts::IfOp::build(OpBuilder &builder, OperationState &result, TypeRange resultTypes, Value cond,
                          function_ref<void(OpBuilder &, Location)> thenBuilder, function_ref<void(OpBuilder &, Location)> elseBuilder)
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

void mlir_ts::IfOp::build(OpBuilder &builder, OperationState &result, Value cond, function_ref<void(OpBuilder &, Location)> thenBuilder,
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
void mlir_ts::IfOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions)
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

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::WhileOp::getSuccessorEntryOperands(unsigned index)
{
    assert(index == 0 && "WhileOp is expected to branch only to the first region");

    return inits();
}

void mlir_ts::WhileOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands,
                                           SmallVectorImpl<RegionSuccessor> &regions)
{
    (void)operands;

    if (!index.hasValue())
    {
        regions.emplace_back(&cond(), cond().getArguments());
        return;
    }

    assert(*index < 2 && "there are only two regions in a WhileOp");
    if (*index == 0)
    {
        regions.emplace_back(&body(), body().getArguments());
        regions.emplace_back(getResults());
        return;
    }

    regions.emplace_back(&cond(), cond().getArguments());
}

//===----------------------------------------------------------------------===//
// DoWhileOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::DoWhileOp::getSuccessorEntryOperands(unsigned index)
{
    assert(index == 0 && "DoWhileOp is expected to branch only to the first region");

    return inits();
}

void mlir_ts::DoWhileOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands,
                                             SmallVectorImpl<RegionSuccessor> &regions)
{
    (void)operands;

    if (!index.hasValue())
    {
        regions.emplace_back(&cond(), cond().getArguments());
        return;
    }

    assert(*index < 2 && "there are only two regions in a DoWhileOp");
    if (*index == 0)
    {
        regions.emplace_back(&body(), body().getArguments());
        regions.emplace_back(getResults());
        return;
    }

    regions.emplace_back(&cond(), cond().getArguments());
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::ForOp::getSuccessorEntryOperands(unsigned index)
{
    assert(index == 0 && "ForOp is expected to branch only to the first region");

    return inits();
}

void mlir_ts::ForOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions)
{
    (void)operands;

    if (!index.hasValue())
    {
        regions.emplace_back(&cond(), cond().getArguments());
        return;
    }

    assert(*index < 2 && "there are only two regions in a ForOp");
    if (*index == 0)
    {
        regions.emplace_back(&incr(), incr().getArguments());
        regions.emplace_back(&body(), body().getArguments());
        regions.emplace_back(getResults());
        return;
    }

    regions.emplace_back(&cond(), cond().getArguments());
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

void mlir_ts::SwitchOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands,
                                            SmallVectorImpl<RegionSuccessor> &regions)
{
    regions.push_back(RegionSuccessor(&casesRegion()));
}

Block *mlir_ts::SwitchOp::getHeaderBlock()
{
    assert(!casesRegion().empty() && "op region should not be empty!");
    // The first block is the loop header block.
    return &casesRegion().front();
}

Block *mlir_ts::SwitchOp::getMergeBlock()
{
    assert(!casesRegion().empty() && "op region should not be empty!");
    // The last block is the loop merge block.
    return &casesRegion().back();
}

void mlir_ts::SwitchOp::addMergeBlock()
{
    assert(casesRegion().empty() && "entry and merge block already exist");
    auto *mergeBlock = new Block();
    casesRegion().push_back(mergeBlock);
    OpBuilder builder = OpBuilder::atBlockEnd(mergeBlock);

    // Add a ts.merge op into the merge block.
    builder.create<mlir_ts::MergeOp>(getLoc());
}

namespace
{
// Pattern to remove unused IfOp results.
struct RemoveUnusedResults : public OpRewritePattern<mlir_ts::IfOp>
{
    using OpRewritePattern<mlir_ts::IfOp>::OpRewritePattern;

    void transferBody(Block *source, Block *dest, ArrayRef<OpResult> usedResults, PatternRewriter &rewriter) const
    {
        // Move all operations to the destination block.
        rewriter.mergeBlocks(source, dest);
        // Replace the yield op by one that returns only the used values.
        auto yieldOp = cast<mlir_ts::ResultOp>(dest->getTerminator());
        SmallVector<Value, 4> usedOperands;
        llvm::transform(usedResults, std::back_inserter(usedOperands),
                        [&](OpResult result) { return yieldOp.getOperand(result.getResultNumber()); });
        rewriter.updateRootInPlace(yieldOp, [&]() { yieldOp->setOperands(usedOperands); });
    }

    LogicalResult matchAndRewrite(mlir_ts::IfOp op, PatternRewriter &rewriter) const override
    {
        // Compute the list of used results.
        SmallVector<OpResult, 4> usedResults;
        llvm::copy_if(op.getResults(), std::back_inserter(usedResults), [](OpResult result) { return !result.use_empty(); });

        // Replace the operation if only a subset of its results have uses.
        if (usedResults.size() == op.getNumResults())
        {
            return failure();
        }

        // Compute the result types of the replacement operation.
        SmallVector<Type, 4> newTypes;
        llvm::transform(usedResults, std::back_inserter(newTypes), [](OpResult result) { return result.getType(); });

        // Create a replacement operation with empty then and else regions.
        auto emptyBuilder = [](OpBuilder &, Location) {};
        auto newOp = rewriter.create<mlir_ts::IfOp>(op.getLoc(), newTypes, op.condition(), emptyBuilder, emptyBuilder);

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

struct RemoveStaticCondition : public OpRewritePattern<mlir_ts::IfOp>
{
    using OpRewritePattern<mlir_ts::IfOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir_ts::IfOp op, PatternRewriter &rewriter) const override
    {
        auto constant = op.condition().getDefiningOp<mlir_ts::ConstantOp>();
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

void mlir_ts::IfOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnusedResults, RemoveStaticCondition>(context);
}

void mlir_ts::GlobalOp::build(OpBuilder &builder, OperationState &result, Type type, bool isConstant, StringRef name, Attribute value,
                              ArrayRef<NamedAttribute> attrs)
{
    result.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    result.addAttribute("type", TypeAttr::get(type));
    if (isConstant)
    {
        result.addAttribute("constant", builder.getUnitAttr());
    }

    if (value)
    {
        result.addAttribute("value", value);
    }

    result.attributes.append(attrs.begin(), attrs.end());
    result.addRegion();
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::TryOp::getSuccessorEntryOperands(unsigned index)
{
    assert(index == 0 && "TryOp is expected to branch only to the first region");

    return getODSOperands(0);
}

void mlir_ts::TryOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions)
{
    regions.push_back(RegionSuccessor(&catches()));
    regions.push_back(RegionSuccessor(&finallyBlock()));
}
