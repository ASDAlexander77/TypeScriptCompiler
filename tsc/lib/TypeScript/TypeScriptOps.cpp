#define DEBUG_TYPE "mlir"

#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptDialect.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

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
/// FunctionType
//===----------------------------------------------------------------------===//

bool mlir_ts::FunctionType::isVarArg() const
{
    return getIsVarArg();
}

unsigned mlir_ts::FunctionType::getNumInputs() const
{
    return getInputs().size();
}

unsigned mlir_ts::FunctionType::getNumResults() const
{
    return getResults().size();
}

mlir::Type mlir_ts::FunctionType::getReturnType()
{
    return getResults().empty() ? mlir::Type() : getResults().front();
}

unsigned mlir_ts::FunctionType::getNumParams()
{
    return getNumInputs();
}

mlir::Type mlir_ts::FunctionType::getParamType(unsigned i)
{
    return getInputs()[i];
}

ArrayRef<mlir::Type> mlir_ts::FunctionType::getParams()
{
    return getInputs();
}

//===----------------------------------------------------------------------===//
/// HybridFunctionType
//===----------------------------------------------------------------------===//
bool mlir_ts::HybridFunctionType::isVarArg() const
{
    return getIsVarArg();
}

unsigned mlir_ts::HybridFunctionType::getNumInputs() const
{
    return getInputs().size();
}

unsigned mlir_ts::HybridFunctionType::getNumResults() const
{
    return getResults().size();
}

//===----------------------------------------------------------------------===//
/// BoundFunctionType
//===----------------------------------------------------------------------===//
bool mlir_ts::BoundFunctionType::isVarArg() const
{
    return getIsVarArg();
}

unsigned mlir_ts::BoundFunctionType::getNumInputs() const
{
    return getInputs().size();
}

unsigned mlir_ts::BoundFunctionType::getNumResults() const
{
    return getResults().size();
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

void mlir_ts::AccessorOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::AccessorOp>>(context);
}

//===----------------------------------------------------------------------===//
// ThisAccessorRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ThisAccessorOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ThisAccessorOp>>(context);
}

//===----------------------------------------------------------------------===//
// InterfaceSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::InterfaceSymbolRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::InterfaceSymbolRefOp>>(context);
}

/*
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
*/

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
// NullOp
//===----------------------------------------------------------------------===//

void mlir_ts::NullOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::NullOp>>(context);
}

//===----------------------------------------------------------------------===//
// UndefOp
//===----------------------------------------------------------------------===//

void mlir_ts::UndefOp::getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::UndefOp>>(context);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult verify(mlir_ts::CastOp op)
{
    auto inType = op.in().getType();
    auto resType = op.res().getType();

    // funcType -> funcType
    auto inFuncType = inType.dyn_cast_or_null<mlir_ts::FunctionType>();
    auto resFuncType = resType.dyn_cast_or_null<mlir_ts::FunctionType>();
    if (inFuncType && resFuncType)
    {
        ::typescript::MLIRTypeHelper mth(op.getContext());
        auto result = mth.TestFunctionTypesMatchWithObjectMethods(inFuncType, resFuncType);
        if (::typescript::MatchResultType::Match != result.result)
        {
            op->emitError("can't cast function type ") << inFuncType << "[" << inFuncType.getNumInputs() << "] to type " << resFuncType
                                                       << "[" << resFuncType.getNumInputs() << "]";
            return failure();
        }

        return success();
    }

    // optional<T> -> <T>
    if (auto inOptType = inType.dyn_cast_or_null<mlir_ts::OptionalType>())
    {
        if (inOptType.getElementType() == resType)
        {
            return success();
        }
    }

    if (auto resOptType = resType.dyn_cast_or_null<mlir_ts::OptionalType>())
    {
        if (resOptType.getElementType() == inType)
        {
            return success();
        }
    }

    // check if we can cast type to union type
    auto inUnionType = inType.dyn_cast_or_null<mlir_ts::UnionType>();
    auto resUnionType = resType.dyn_cast_or_null<mlir_ts::UnionType>();
    if (inUnionType || resUnionType)
    {
        ::typescript::MLIRTypeHelper mth(op.getContext());
        auto cmpTypes = [&](mlir::Type t1, mlir::Type t2) { return mth.isCastableTypes(t1, t2); };

        if (inUnionType && !resUnionType)
        {
            auto pred = [&](auto &item) { return cmpTypes(item, resType); };
            auto types = inUnionType.getTypes();
            if (std::find_if(types.begin(), types.end(), pred) == types.end())
            {
                return op.emitOpError("type [") << inUnionType << "] does not have [" << resType << "] type";
            }

            return success();
        }

        if (!inUnionType && resUnionType)
        {
            auto pred = [&](auto &item) { return cmpTypes(inType, item); };
            auto types = resUnionType.getTypes();
            if (std::find_if(types.begin(), types.end(), pred) == types.end())
            {
                return op.emitOpError("type [") << inType << "] can't be stored in [" << resUnionType << "]";
            }

            return success();
        }

        auto resUnionTypes = resUnionType.getTypes();

        auto predForInUnion = [&](auto &inUnionItem) {
            auto pred = [&](auto &resUnionItem) { return cmpTypes(inUnionItem, resUnionItem); };
            return std::find_if(resUnionTypes.begin(), resUnionTypes.end(), pred) != resUnionTypes.end();
        };

        auto inUnionTypes = inUnionType.getTypes();
        if (std::find_if(inUnionTypes.begin(), inUnionTypes.end(), predForInUnion) == inUnionTypes.end())
        {
            return op.emitOpError("type [") << inUnionType << "] can't be stored in [" << resUnionType << ']';
        }

        return success();
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

        auto loc = castOp->getLoc();

        // any support
        if (res.getType().isa<mlir_ts::AnyType>())
        {
            // TODO: boxing, finish it, need to send TypeOf
            auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
            auto boxedValue = rewriter.create<mlir_ts::BoxOp>(loc, mlir_ts::AnyType::get(rewriter.getContext()), in, typeOfValue);
            rewriter.replaceOp(castOp, ValueRange{boxedValue});
            return success();
        }

        if (in.getType().isa<mlir_ts::AnyType>())
        {
            auto unboxedValue = rewriter.create<mlir_ts::UnboxOp>(loc, res.getType(), in);
            rewriter.replaceOp(castOp, ValueRange{unboxedValue});
            return success();
        }

        // union support
        auto isInUnionType = in.getType().isa<mlir_ts::UnionType>();
        auto isResUnionType = res.getType().isa<mlir_ts::UnionType>();
        if (isResUnionType && !isInUnionType)
        {
            // TODO: boxing, finish it, need to send TypeOf
            auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
            auto unionValue = rewriter.create<mlir_ts::CreateUnionInstanceOp>(loc, res.getType(), in, typeOfValue);
            rewriter.replaceOp(castOp, ValueRange{unionValue});
            return success();
        }

        if (isInUnionType && !isResUnionType)
        {
            auto value = rewriter.create<mlir_ts::GetValueFromUnionOp>(loc, res.getType(), in);
            rewriter.replaceOp(castOp, ValueRange{value});
            return success();
        }

        /*
        if (isInUnionType && isResUnionType)
        {
            auto maxStoreType = ? ? ? ;
            auto value = rewriter.create<mlir_ts::GetValueFromUnionOp>(loc, maxStoreType, in);
            auto typeOfValue = rewriter.create<mlir_ts::GetTypeInfoFromUnionOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
            auto unionValue = rewriter.create<mlir_ts::CreateUnionInstanceOp>(loc, res.getType(), value, typeOfValue);
            rewriter.replaceOp(castOp, ValueRange{unionValue});
            return success();
        }
        */

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

        // const tuple -> const tuple, for example { value: undefined, done: true } -> { value: <int>, done: <boolean> }
        if (auto constTupleIn = in.getType().dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            ::typescript::MLIRTypeHelper mth(rewriter.getContext());

            if (auto constTupleRes = res.getType().dyn_cast_or_null<mlir_ts::ConstTupleType>())
            {
                if (mth.isCastableTypesLogic(constTupleIn, constTupleRes))
                {
                    // create other const tuple from source const tuple
                    if (auto constOp = in.getDefiningOp<mlir_ts::ConstantOp>())
                    {
                        auto newConstOp = rewriter.create<mlir_ts::ConstantOp>(loc, constTupleRes, constOp.valueAttr());
                        rewriter.replaceOp(castOp, ValueRange{newConstOp});
                        rewriter.eraseOp(constOp);
                    }
                }
            }
            else if (auto tupleRes = res.getType().dyn_cast_or_null<mlir_ts::TupleType>())
            {
                if (mlir_ts::TupleType::get(rewriter.getContext(), constTupleIn.getFields()) != tupleRes &&
                    mth.isCastableTypesLogic(constTupleIn, tupleRes))
                {
                    // create other const tuple from source const tuple
                    if (auto constOp = in.getDefiningOp<mlir_ts::ConstantOp>())
                    {
                        ::typescript::MLIRTypeHelper mth(rewriter.getContext());
                        auto constTupleType = mth.convertTupleTypeToConstTupleType(tupleRes);

                        auto newConstOp = rewriter.create<mlir_ts::ConstantOp>(loc, constTupleType, constOp.valueAttr());
                        rewriter.replaceOp(constOp, ValueRange{newConstOp});
                        auto newCastOp = rewriter.create<mlir_ts::CastOp>(loc, tupleRes, newConstOp);
                        rewriter.replaceOp(castOp, ValueRange{newCastOp});
                    }
                }
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

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool mlir_ts::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
    {
        // not supporting N->N cast
        return false;
    }

    // for now all are true
    return true;
}

//===----------------------------------------------------------------------===//
// DialectCastOp
//===----------------------------------------------------------------------===//

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool mlir_ts::DialectCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
    {
        // not supporting N->N cast
        return false;
    }

    // for now all are true
    return true;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

mlir::Block *mlir_ts::FuncOp::addEntryBlock()
{
    assert(empty() && "function already has an entry block");
    // assert(!isVarArg() && "unimplemented: non-external variadic functions");

    auto *entry = new Block;
    push_back(entry);

    mlir_ts::FunctionType type = getType();
    for (unsigned i = 0, e = type.getNumParams(); i < e; ++i)
    {
        entry->addArgument(type.getParamType(i));
    }

    return entry;
}

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
// InvokeHybridOp
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange> mlir_ts::InvokeHybridOp::getMutableSuccessorOperands(unsigned index)
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
        if (getOperand(i).getType() != fnType.getInput(i))
        {
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided " << getOperand(i).getType() << " for operand number " << i;
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

mlir_ts::FunctionType mlir_ts::CallOp::getCalleeType()
{
    SmallVector<mlir::Type> oper(getOperandTypes());
    SmallVector<mlir::Type> res(getResultTypes());
    return mlir_ts::FunctionType::get(getContext(), oper, res);
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult mlir_ts::CallIndirectOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    mlir::ArrayRef<mlir::Type> input;
    mlir::ArrayRef<mlir::Type> results;

    // Verify that the operand and result types match the callee.
    if (auto funcType = getCallee().getType().dyn_cast<mlir_ts::FunctionType>())
    {
        input = funcType.getInputs();
        results = funcType.getResults();
    }
    else if (auto hybridFuncType = getCallee().getType().dyn_cast<mlir_ts::HybridFunctionType>())
    {
        input = hybridFuncType.getInputs();
        results = hybridFuncType.getResults();
    }

    auto optionalFromValue = (int)input.size() - (int)getNumOperands();
    for (unsigned i = 0, e = optionalFromValue == -1 ? (int)input.size() : getOperands().size(); i != e; ++i)
    {
        if (getOperand(i + 1).getType() != input[i])
        {
            return emitOpError("operand type mismatch: expected operand type ")
                   << input[i] << ", but provided " << getOperand(i + 1).getType() << " for operand number " << i;
        }
    }

    for (unsigned i = 0, e = results.size(); i != e; ++i)
    {
        if (getResult(i).getType() != results[i])
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
            SmallVector<mlir::Value> args;
            args.push_back(thisSymbolRefOp.thisVal());
            args.append(indirectCall.getArgOperands().begin(), indirectCall.getArgOperands().end());

            // Replace with a direct call.
            rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, thisSymbolRefOp.identifierAttr(), indirectCall.getResultTypes(),
                                                         args);
            return success();
        }

        // supporting trumpoline
#ifndef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
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
#endif

        if (auto getMethodOp = indirectCall.getCallee().getDefiningOp<mlir_ts::GetMethodOp>())
        {
            if (auto createBoundFunctionOp = getMethodOp.boundFunc().getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
            {
                if (auto symbolRefOp = createBoundFunctionOp.func().getDefiningOp<mlir_ts::SymbolRefOp>())
                {
                    auto getThisVal = indirectCall.getArgOperands().front().getDefiningOp<mlir_ts::GetThisOp>();

                    auto thisVal = createBoundFunctionOp.thisVal();
                    if (auto castOp = thisVal.getDefiningOp<mlir_ts::CastOp>())
                    {
                        thisVal = castOp.in();
                    }

                    auto hasThis = !isa<mlir_ts::NullOp>(thisVal.getDefiningOp());

                    // Replace with a direct call.
                    SmallVector<mlir::Value> args;
                    if (hasThis)
                    {
                        args.push_back(thisVal);
                    }

                    args.append(indirectCall.getArgOperands().begin() + (hasThis ? 1 : 0), indirectCall.getArgOperands().end());
                    rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, symbolRefOp.identifierAttr(), indirectCall.getResultTypes(),
                                                                 args);

                    LLVM_DEBUG(for (auto &arg : args) { llvm::dbgs() << "\n\n SimplifyIndirectCallWithKnownCallee arg: " << arg << "\n"; });

                    LLVM_DEBUG(llvm::dbgs() << "\nSimplifyIndirectCallWithKnownCallee: args: " << args.size() << "\n";);
                    LLVM_DEBUG(
                        for (auto &use
                             : createBoundFunctionOp->getUses()) { llvm::dbgs() << "\n use number:" << use.getOperandNumber() << "\n"; });

                    if (getMethodOp.use_empty())
                    {
                        rewriter.eraseOp(getMethodOp);
                    }

                    if (getThisVal)
                    {
                        if (getThisVal.use_empty())
                        {
                            rewriter.eraseOp(getThisVal);
                        }
                    }

                    if (createBoundFunctionOp.use_empty())
                    {
                        rewriter.eraseOp(createBoundFunctionOp);
                    }

                    return success();
                }
            }
            else if (auto thisSymbolRef = getMethodOp.boundFunc().getDefiningOp<mlir_ts::ThisSymbolRefOp>())
            {
                auto getThisVal = indirectCall.getArgOperands().front().getDefiningOp<mlir_ts::GetThisOp>();

                auto thisVal = thisSymbolRef.thisVal();

                // Replace with a direct call.
                SmallVector<mlir::Value> args;
                args.push_back(thisVal);
                args.append(indirectCall.getArgOperands().begin() + 1, indirectCall.getArgOperands().end());
                rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, thisSymbolRef.identifierAttr(), indirectCall.getResultTypes(),
                                                             args);

                LLVM_DEBUG(for (auto &arg : args) { llvm::dbgs() << "\n\n SimplifyIndirectCallWithKnownCallee arg: " << arg << "\n"; });

                LLVM_DEBUG(llvm::dbgs() << "\nSimplifyIndirectCallWithKnownCallee: args: " << args.size() << "\n";);
                LLVM_DEBUG(for (auto &use
                                : thisSymbolRef->getUses()) { llvm::dbgs() << "\n use number:" << use.getOperandNumber() << "\n"; });

                if (getMethodOp.use_empty())
                {
                    rewriter.eraseOp(getMethodOp);
                }

                if (getThisVal.use_empty())
                {
                    rewriter.eraseOp(getThisVal);
                }

                if (thisSymbolRef.use_empty())
                {
                    rewriter.eraseOp(thisSymbolRef);
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
// SymbolCallInternalOp
//===----------------------------------------------------------------------===//

LogicalResult mlir_ts::SymbolCallInternalOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // nothing to do here
    return success();
}

//===----------------------------------------------------------------------===//
// CallHybridInternalOp
//===----------------------------------------------------------------------===//

LogicalResult mlir_ts::CallHybridInternalOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // nothing to do here
    return success();
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
// !!! Results should be removed when IfOp is processed as it is terminator
/*
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
*/

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
    results.insert</*RemoveUnusedResults,*/ RemoveStaticCondition>(context);
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

//===----------------------------------------------------------------------===//
// LabelOp
//===----------------------------------------------------------------------===//

void mlir_ts::LabelOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands,
                                           SmallVectorImpl<RegionSuccessor> &regions)
{
    regions.push_back(RegionSuccessor(&labelRegion()));
}

Block *mlir_ts::LabelOp::getMergeBlock()
{
    assert(!labelRegion().empty() && "op region should not be empty!");
    // The last block is the loop merge block.
    return &labelRegion().back();
}

void mlir_ts::LabelOp::addMergeBlock()
{
    assert(labelRegion().empty() && "entry and merge block already exist");
    auto *mergeBlock = new Block();
    labelRegion().push_back(mergeBlock);
    OpBuilder builder = OpBuilder::atBlockEnd(mergeBlock);

    // Add a ts.merge op into the merge block.
    builder.create<mlir_ts::MergeOp>(getLoc());
}

//===----------------------------------------------------------------------===//
// BodyInternalOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::BodyInternalOp::getSuccessorEntryOperands(unsigned index)
{
    assert(index == 0 && "BodyInternalOp is expected to branch only to the first region");

    return getODSOperands(0);
}

void mlir_ts::BodyInternalOp::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands,
                                                  SmallVectorImpl<RegionSuccessor> &regions)
{
    regions.push_back(RegionSuccessor(&body()));
}