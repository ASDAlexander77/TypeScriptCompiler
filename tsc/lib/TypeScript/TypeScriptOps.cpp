#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptDialect.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

/// Default callback for IfOp builders. Inserts a yield without arguments.
void mlir_ts::buildTerminatedBody(OpBuilder &builder, Location loc)
{
    builder.create<mlir_ts::ResultOp>(loc);
}

// util
bool mlir_ts::isTrue(mlir::Region &condtion)
{
    if (!condtion.hasOneBlock())
    {
        return false;
    }

    if (auto condOp = dyn_cast<mlir_ts::ConditionOp>(condtion.back().back()))
    {
        mlir::Value condVal = condOp.getCondition();
        if (auto castVal = condVal.getDefiningOp<mlir_ts::CastOp>())
        {
            condVal = castVal.getIn();
        }

        if (auto constOp = condVal.getDefiningOp<mlir_ts::ConstantOp>())
        {
            if (auto boolAttr = constOp.getValueAttr().dyn_cast<mlir::BoolAttr>())
            {
                return boolAttr.getValue();
            }
        }
    }

    return false;
}

bool mlir_ts::isEmpty(mlir::Region &condtion)
{
    if (!condtion.hasOneBlock())
    {
        return false;
    }

    if (auto noCondOp = dyn_cast<mlir_ts::NoConditionOp>(condtion.back().back()))
    {
        return true;
    }

    return false;
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

mlir_ts::FunctionType mlir_ts::FunctionType::clone(mlir::TypeRange inputs, mlir::TypeRange results) const {
    SmallVector<mlir::Type> clonedInputs;
    for (auto type : inputs)
    {
        clonedInputs.push_back(type);
    }

    SmallVector<mlir::Type> clonedResults;
    for (auto type : results)
    {
        clonedResults.push_back(type);
    }

    return get(getContext(), clonedInputs, clonedResults, isVarArg());
}

/// Returns a new function type with the specified arguments and results
/// inserted.
mlir_ts::FunctionType mlir_ts::FunctionType::getWithArgsAndResults(ArrayRef<unsigned> argIndices, TypeRange argTypes,
                                                                   ArrayRef<unsigned> resultIndices,
                                                                   TypeRange resultTypes)
{
    SmallVector<Type> argStorage, resultStorage;
    TypeRange newArgTypes = function_interface_impl::insertTypesInto(getInputs(), argIndices, argTypes, argStorage);
    TypeRange newResultTypes =
        function_interface_impl::insertTypesInto(getResults(), resultIndices, resultTypes, resultStorage);
    return clone(newArgTypes, newResultTypes);
}

/// Returns a new function type without the specified arguments and results.
mlir_ts::FunctionType mlir_ts::FunctionType::getWithoutArgsAndResults(const BitVector &argIndices,
                                                                      const BitVector &resultIndices)
{
    SmallVector<Type> argStorage, resultStorage;
    TypeRange newArgTypes = function_interface_impl::filterTypesOut(getInputs(), argIndices, argStorage);
    TypeRange newResultTypes = function_interface_impl::filterTypesOut(getResults(), resultIndices, resultStorage);
    return clone(newArgTypes, newResultTypes);
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
/// ConstructFunctionType
//===----------------------------------------------------------------------===//
bool mlir_ts::ConstructFunctionType::isVarArg() const
{
    return getIsVarArg();
}

unsigned mlir_ts::ConstructFunctionType::getNumInputs() const
{
    return getInputs().size();
}

unsigned mlir_ts::ConstructFunctionType::getNumResults() const
{
    return getResults().size();
}

//===----------------------------------------------------------------------===//
/// ExtensionFunctionType
//===----------------------------------------------------------------------===//
bool mlir_ts::ExtensionFunctionType::isVarArg() const
{
    return getIsVarArg();
}

unsigned mlir_ts::ExtensionFunctionType::getNumInputs() const
{
    return getInputs().size();
}

unsigned mlir_ts::ExtensionFunctionType::getNumResults() const
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
OpFoldResult mlir_ts::ConstantOp::fold(FoldAdaptor adaptor)
{
    ArrayRef<Attribute> operands = adaptor.getOperands();
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
// SymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::SymbolRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::SymbolRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// ThisSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ThisSymbolRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ThisSymbolRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// VirtualSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::VirtualSymbolRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::VirtualSymbolRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// ThisVirtualSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ThisVirtualSymbolRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ThisVirtualSymbolRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// AccessorRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::AccessorOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::AccessorOp>>(context);
}

//===----------------------------------------------------------------------===//
// ThisAccessorRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ThisAccessorOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ThisAccessorOp>>(context);
}

//===----------------------------------------------------------------------===//
// InterfaceSymbolRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::InterfaceSymbolRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
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

void mlir_ts::TypeRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::TypeRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// ClassRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::ClassRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ClassRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// InterfaceRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::InterfaceRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::InterfaceRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// NamespaceRefOp
//===----------------------------------------------------------------------===//

void mlir_ts::NamespaceRefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::NamespaceRefOp>>(context);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void mlir_ts::LoadOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::LoadOp>>(context);
}

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

void mlir_ts::ValueOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::ValueOp>>(context);
}

//===----------------------------------------------------------------------===//
// NullOp
//===----------------------------------------------------------------------===//

void mlir_ts::NullOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::NullOp>>(context);
}

//===----------------------------------------------------------------------===//
// CreateExtensionFunction
//===----------------------------------------------------------------------===//

void mlir_ts::CreateExtensionFunctionOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<RemoveUnused<mlir_ts::CreateExtensionFunctionOp>>(context);
}

//===----------------------------------------------------------------------===//
// UndefOp
//===----------------------------------------------------------------------===//

/*
namespace
{
struct NormalizeUndefTypes : public OpRewritePattern<mlir_ts::UndefOp>
{
    using OpRewritePattern<mlir_ts::UndefOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir_ts::UndefOp undefOp, PatternRewriter &rewriter) const override
    {
        if (undefOp.getType() == mlir_ts::UndefinedType::get(rewriter.getContext()))
        {
            for (auto user : undefOp.getResult().getUsers())
            {
                for (auto operand : user->getOperands())
                {
                    if (operand == undefOp && operand.getType() != undefOp.getType())
                    {
                        // replace
                        auto newOp = rewriter.create<mlir_ts::UndefOp>(undefOp.getLoc(), operand.getType());
                        operand.replaceAllUsesWith(newOp);
                    }
                }
            }

            if (undefOp->use_empty())
            {
                rewriter.eraseOp(undefOp);
            }
        }        
        else if (undefOp->getResult(0).use_empty())
        {
            // remove unsed
            rewriter.eraseOp(undefOp);
        }

        return success();
    }
};

} // end anonymous namespace.
*/

void mlir_ts::UndefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    //results.insert<NormalizeUndefTypes>(context);
    results.insert<RemoveUnused<mlir_ts::UndefOp>>(context);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult mlir_ts::CastOp::verify()
{
    auto inType = getIn().getType();
    auto resType = getRes().getType();

    // funcType -> funcType
    auto inFuncType = inType.dyn_cast<mlir_ts::FunctionType>();
    auto resFuncType = resType.dyn_cast<mlir_ts::FunctionType>();
    if (inFuncType && resFuncType)
    {
        ::typescript::MLIRTypeHelper mth(getContext());
        auto result = mth.TestFunctionTypesMatchWithObjectMethods(inFuncType, resFuncType);
        if (::typescript::MatchResultType::Match != result.result)
        {
            emitError("can't cast function type ") << inFuncType << "[" << inFuncType.getNumInputs() << "] to type " << resFuncType
                                                       << "[" << resFuncType.getNumInputs() << "]";
            return failure();
        }

        return success();
    }

    // optional<T> -> <T>
    if (auto inOptType = inType.dyn_cast<mlir_ts::OptionalType>())
    {
        if (inOptType.getElementType() == resType)
        {
            return success();
        }
    }

    if (auto resOptType = resType.dyn_cast<mlir_ts::OptionalType>())
    {
        if (resOptType.getElementType() == inType)
        {
            return success();
        }
    }

    if (resType.isa<mlir_ts::AnyType>())
    {
        return success();
    }

    // check if we can cast type to union type
    auto inUnionType = inType.dyn_cast<mlir_ts::UnionType>();
    auto resUnionType = resType.dyn_cast<mlir_ts::UnionType>();
    if (inUnionType || resUnionType)
    {
        ::typescript::MLIRTypeHelper mth(getContext());
        auto cmpTypes = [&](mlir::Type t1, mlir::Type t2) { return mth.canCastFromTo(t1, t2); };

        if (inUnionType && !resUnionType)
        {
            auto pred = [&](auto &item) { 
                return cmpTypes(item, resType); 
            };
            auto types = inUnionType.getTypes();
            if (!std::all_of(types.begin(), types.end(), pred))
            {
                ::typescript::MLIRTypeHelper mth(getContext());
                mlir::Type baseType;
                if (!mth.isUnionTypeNeedsTag(inUnionType, baseType)/* && mth.canCastFromTo(baseType, resType)*/)
                {
                    // we need to ignore this case, for example if union<int, int, int> -> string, we need cast int to string
                    return success();
                }

                return emitOpError("not all types in [") << inUnionType << "] can be casted to [" << resType << "] type";
            }

            return success();
        }

        if (!inUnionType && resUnionType)
        {
            // TODO: review using "undefined", use proper union type
            auto effectiveInType = mth.stripOptionalType(inType);

            if (!effectiveInType.isa<mlir_ts::UndefinedType>())
            {
                auto pred = [&](auto &item) { 
                    return cmpTypes(effectiveInType, item); 
                };
                auto types = resUnionType.getTypes();
                if (std::find_if(types.begin(), types.end(), pred) == types.end())
                {
                    LLVM_DEBUG(llvm::dbgs() << "!! Location of CastOp: " << getLoc() << "\n";);

                    return emitOpError("type [") << inType << "] can't be stored in [" << resUnionType << "]";
                }
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
            return emitOpError("type [") << inUnionType << "] can't be stored in [" << resUnionType << ']';
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
        auto in = castOp.getIn();
        auto res = castOp.getRes();

        if (in.getType() == res.getType())
        {
            rewriter.replaceOp(castOp, in);
            return success();
        }

        auto loc = castOp->getLoc();

        // remove chain calls
        /*
        for (auto user : res.getUsers())
        {
            auto any = false;
            if (auto chainCast = dyn_cast_or_null<mlir_ts::CastOp>(user))
            {
                if (chainCast.getIn().getType() == res.getType())
                {
                    // we need to 
                    auto newCastOp = rewriter.create<mlir_ts::CastOp>(loc, chainCast.getRes().getType(), in);
                    rewriter.replaceOp(chainCast, ValueRange{newCastOp});
                    if (castOp.getRes().use_empty())
                    {
                        rewriter.eraseOp(castOp);
                    }

                    any = true;
                }
            }

            if (any)
            {
                return success();
            }
        } 
        */            

        // union support
        // TODO: review this code, should it be in "cast" logic?
        if (res.getType().isa<mlir_ts::AnyType>())
        {
            return success();
        }

        auto resUnionType = res.getType().dyn_cast<mlir_ts::UnionType>();
        auto inUnionType = in.getType().dyn_cast<mlir_ts::UnionType>();
        if (resUnionType && !inUnionType)
        {
            ::typescript::MLIRTypeHelper mth(rewriter.getContext());
            if (mth.isUnionTypeNeedsTag(resUnionType))
            {
                // TODO: boxing, finish it, need to send TypeOf
                auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
                auto unionValue = rewriter.create<mlir_ts::CreateUnionInstanceOp>(loc, res.getType(), in, typeOfValue);
                rewriter.replaceOp(castOp, ValueRange{unionValue});
            }

            return success();
        }

        // TODO: review it, if you still need it as we are should be using "safeCast"
        if (inUnionType && !resUnionType)
        {
            ::typescript::MLIRTypeHelper mth(rewriter.getContext());
            if (mth.isUnionTypeNeedsTag(inUnionType))
            {
                auto value = rewriter.create<mlir_ts::GetValueFromUnionOp>(loc, res.getType(), in);
                rewriter.replaceOp(castOp, ValueRange{value});
            }

            return success();
        }

        return success();
    }
};

} // end anonymous namespace.

void mlir_ts::CastOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
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

    mlir_ts::FunctionType type = getFunctionType();
    for (unsigned i = 0, e = type.getNumParams(); i < e; ++i)
    {
        entry->addArgument(type.getParamType(i), getLoc());
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
    state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();

    if (argAttrs.empty())
    {
        return;
    }

    assert(type.getNumInputs() == argAttrs.size());
    function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
        getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name)
    );
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void mlir_ts::FuncOp::cloneInto(FuncOp dest, IRMapping &mapper)
{
    // Add the attributes of this function to dest.
    llvm::MapVector<StringAttr, Attribute> newAttrMap;
    for (const auto &attr : dest->getAttrs())
        newAttrMap.insert({attr.getName(), attr.getValue()});
    for (const auto &attr : (*this)->getAttrs())
        newAttrMap.insert({attr.getName(), attr.getValue()});

    auto newAttrs = llvm::to_vector(llvm::map_range(newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
    }));
    dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

    // Clone the body.
    getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
mlir_ts::FuncOp mlir_ts::FuncOp::clone(IRMapping &mapper)
{
    // Create the new function.
    FuncOp newFunc = cast<FuncOp>(getOperation()->cloneWithoutRegions());

    // If the function has a body, then the user might be deleting arguments to
    // the function by specifying them in the mapper. If so, we don't add the
    // argument to the input type vector.
    if (!isExternal())
    {
        FunctionType oldType = getFunctionType();

        unsigned oldNumArgs = oldType.getNumInputs();
        SmallVector<Type, 4> newInputs;
        newInputs.reserve(oldNumArgs);
        for (unsigned i = 0; i != oldNumArgs; ++i)
            if (!mapper.contains(getArgument(i)))
                newInputs.push_back(oldType.getInput(i));

        /// If any of the arguments were dropped, update the type and drop any
        /// necessary argument attributes.
        if (newInputs.size() != oldNumArgs)
        {
            newFunc.setType(FunctionType::get(oldType.getContext(), newInputs, oldType.getResults(), oldType.isVarArg()));

            if (ArrayAttr argAttrs = getAllArgAttrs())
            {
                SmallVector<Attribute> newArgAttrs;
                newArgAttrs.reserve(newInputs.size());
                for (unsigned i = 0; i != oldNumArgs; ++i)
                    if (!mapper.contains(getArgument(i)))
                        newArgAttrs.push_back(argAttrs[i]);
                newFunc.setAllArgAttrs(newArgAttrs);
            }
        }
    }

    /// Clone the current function into the new one and return it.
    cloneInto(newFunc, mapper);
    return newFunc;
}

mlir_ts::FuncOp mlir_ts::FuncOp::clone()
{
    IRMapping mapper;
    return clone(mapper);
}

LogicalResult mlir_ts::FuncOp::verify()
{
    // If this function is external there is nothing to do.
    if (isExternal())
        return success();

    // Verify that the argument list of the function and the arg list of the entry
    // block line up.  The trait already verified that the number of arguments is
    // the same between the signature and the block.
    auto fnInputTypes = getFunctionType().getInputs();
    Block &entryBlock = front();
    for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
        if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
            return emitOpError("type of entry block argument #")
                   << i << '(' << entryBlock.getArgument(i).getType() << ") must match the type of the corresponding argument in "
                   << "function signature(" << fnInputTypes[i] << ')';

    return success();
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

LogicalResult mlir_ts::InvokeOp::verify()
{
    if (getNumResults() > 1)
    {
        return emitOpError("must have 0 or 1 result");
    }

    Block *unwindDestBlock = getUnwindDest();
    if (unwindDestBlock->empty())
    {
        return emitError("must have at least one operation in unwind destination");
    }

    // In unwind destination, first operation must be LandingpadOp
    /*
    if (!isa<LandingpadOp>(unwindDestBlock->front()))
    {
        return emitError("first operation in unwind destination should be a "
                            "llvm.landingpad operation");
    }
    */

    return success();
}

SuccessorOperands mlir_ts::InvokeOp::getSuccessorOperands(unsigned index)
{
    assert(index <= 1 && "invalid successor index");
    if (index == 1)
        return SuccessorOperands(getNormalDestOperandsMutable());
    return SuccessorOperands(getUnwindDestOperandsMutable());
}

//===----------------------------------------------------------------------===//
// InvokeHybridOp
//===----------------------------------------------------------------------===//

SuccessorOperands mlir_ts::InvokeHybridOp::getSuccessorOperands(unsigned index)
{
    assert(index <= 1 && "invalid successor index");
    if (index == 1)
        return SuccessorOperands(getNormalDestOperandsMutable());
    return SuccessorOperands(getUnwindDestOperandsMutable());
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
        if (matchPattern(op.getArg(), m_One()))
        {
            rewriter.eraseOp(op);
            return success();
        }

        return failure();
    }
};
} // namespace

void mlir_ts::AssertOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context)
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
    auto fnType = fn.getFunctionType();
    auto isVarArgAttr = fn->getAttrOfType<BoolAttr>("func.varargs");
    auto isVarArg = (isVarArgAttr) ? isVarArgAttr.getValue() : false;

    if (!isVarArg && fnType.getNumInputs() != getNumOperands())
    {
        return emitOpError("Expected ") << fnType.getNumInputs() << " arguments, but got " << getNumOperands() << ".";
    }

    auto e = (int) fnType.getNumInputs();
    for (auto i = 0; i < e; ++i)
    {
        if (getOperand(i).getType() != fnType.getInput(i))
        {
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided " << getOperand(i).getType() << " for operand number " << i;
        }
    }

    e = fnType.getNumResults();
    for (auto i = 0; i < e; ++i)
    {
        if (getResult(i).getType() != fnType.getResult(i))
        {
            return emitOpError("result type mismatch");
        }
    }

    return success();
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

    auto e = (int)input.size(); 
    for (auto i = 0; i < e; ++i)
    {
        if (getOperand(i + 1).getType() != input[i])
        {
            return emitOpError("operand type mismatch: expected operand type ")
                   << input[i] << ", but provided " << getOperand(i + 1).getType() << " for operand number " << i;
        }
    }

    // it is matched in TypesMatchWith in .td file
    /*
    for (auto i = 0, e = results.size(); i != e; ++i)
    {
        if (getResult(i).getType() != results[i])
        {
            return emitOpError("result type mismatch");
        }
    }
    */

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
            rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, symbolRefOp.getIdentifierAttr(), indirectCall.getResultTypes(),
                                                         indirectCall.getArgOperands());
            return success();
        }

        if (auto thisSymbolRefOp = indirectCall.getCallee().getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            SmallVector<mlir::Value> args;
            args.push_back(thisSymbolRefOp.getThisVal());
            args.append(indirectCall.getArgOperands().begin(), indirectCall.getArgOperands().end());

            // Replace with a direct call.
            rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, thisSymbolRefOp.getIdentifierAttr(), indirectCall.getResultTypes(),
                                                         args);
            return success();
        }

        // supporting trumpoline
        if (auto getMethodOp = indirectCall.getCallee().getDefiningOp<mlir_ts::GetMethodOp>())
        {
            if (auto createBoundFunctionOp = getMethodOp.getBoundFunc().getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
            {
                if (auto symbolRefOp = createBoundFunctionOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>())
                {
                    auto getThisVal = indirectCall.getArgOperands().front().getDefiningOp<mlir_ts::GetThisOp>();

                    auto thisVal = createBoundFunctionOp.getThisVal();
                    auto thisValStripedCast = thisVal;
                    if (auto castOp = thisVal.getDefiningOp<mlir_ts::CastOp>())
                    {
                        thisValStripedCast = castOp.getIn();
                    }

                    auto hasThis = !isa<mlir_ts::NullOp>(thisValStripedCast.getDefiningOp());

                    // Replace with a direct call.
                    SmallVector<mlir::Value> args;
                    if (hasThis)
                    {
                        auto neededThisForCall = 
                            createBoundFunctionOp.getType().cast<mlir_ts::BoundFunctionType>().getInput(0) == thisValStripedCast.getType() 
                                ? thisValStripedCast 
                                : thisVal;

                        args.push_back(neededThisForCall);
                    }

                    args.append(indirectCall.getArgOperands().begin() + (hasThis ? 1 : 0), indirectCall.getArgOperands().end());
                    rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, symbolRefOp.getIdentifierAttr(), indirectCall.getResultTypes(),
                                                                 args);

                    // LLVM_DEBUG(for (auto &arg : args) { llvm::dbgs() << "\n\n SimplifyIndirectCallWithKnownCallee arg: " << arg << "\n"; });

                    // LLVM_DEBUG(llvm::dbgs() << "\nSimplifyIndirectCallWithKnownCallee: args: " << args.size() << "\n";);
                    // LLVM_DEBUG(
                    //     for (auto &use
                    //          : createBoundFunctionOp->getUses()) { llvm::dbgs() << "\n use number:" << use.getOperandNumber() << "\n"; });

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
            else if (auto thisSymbolRef = getMethodOp.getBoundFunc().getDefiningOp<mlir_ts::ThisSymbolRefOp>())
            {
                auto getThisVal = indirectCall.getArgOperands().front().getDefiningOp<mlir_ts::GetThisOp>();

                auto thisVal = thisSymbolRef.getThisVal();

                // Replace with a direct call.
                SmallVector<mlir::Value> args;
                args.push_back(thisVal);
                args.append(indirectCall.getArgOperands().begin() + 1, indirectCall.getArgOperands().end());
                rewriter.replaceOpWithNewOp<mlir_ts::CallOp>(indirectCall, thisSymbolRef.getIdentifierAttr(), indirectCall.getResultTypes(),
                                                             args);

                // LLVM_DEBUG(for (auto &arg : args) { llvm::dbgs() << "\n\n SimplifyIndirectCallWithKnownCallee arg: " << arg << "\n"; });

                // LLVM_DEBUG(llvm::dbgs() << "\nSimplifyIndirectCallWithKnownCallee: args: " << args.size() << "\n";);
                // LLVM_DEBUG(for (auto &use
                //                 : thisSymbolRef->getUses()) { llvm::dbgs() << "\n use number:" << use.getOperandNumber() << "\n"; });

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

void mlir_ts::CallIndirectOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
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
    build(builder, result, /*resultTypes=*/std::nullopt, cond, withElseRegion);
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
    mlir::Block *block = &region.front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.inlineBlockBefore(block, op, blockArgs);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void mlir_ts::IfOp::getSuccessorRegions(std::optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions)
{
    // The `then` and the `else` region branch back to the parent operation.
    if (index)
    {
        regions.push_back(RegionSuccessor(getResults()));
        return;
    }

    // Don't consider the else region if it is empty.
    Region *elseRegion = &this->getElseRegion();
    if (elseRegion->empty())
    {
        elseRegion = nullptr;
    }

    // Otherwise, the successor is dependent on the condition.
    bool condition;
    if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
    {
        condition = condAttr.getValue().isOne();
    }
    else
    {
        // If the condition isn't constant, both regions may be executed.
        regions.push_back(RegionSuccessor(&getThenRegion()));
        regions.push_back(RegionSuccessor(elseRegion));
        return;
    }

    // Add the successor regions using the condition.
    regions.push_back(RegionSuccessor(condition ? &getThenRegion() : elseRegion));
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::WhileOp::getSuccessorEntryOperands(std::optional<unsigned int> index)
{
    assert((!index || *index == 0) && "WhileOp is expected to branch only to the first region");

    return getInits();
}

void mlir_ts::WhileOp::getSuccessorRegions(std::optional<unsigned> index, ArrayRef<Attribute> operands,
                                           SmallVectorImpl<RegionSuccessor> &regions)
{
    (void)operands;

    if (!index)
    {
        regions.emplace_back(&getCond(), getCond().getArguments());
        return;
    }

    assert(*index < 2 && "there are only two regions in a WhileOp");
    if (*index == 0)
    {
        regions.emplace_back(&getBody(), getBody().getArguments());
        regions.emplace_back(getResults());
        return;
    }

    regions.emplace_back(&getCond(), getCond().getArguments());
}

//===----------------------------------------------------------------------===//
// DoWhileOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::DoWhileOp::getSuccessorEntryOperands(std::optional<unsigned int> index)
{
    // TODO: review it
    assert((!index || *index == 1) && "DoWhileOp is expected to branch only to the first region");

    return getInits();
}

void mlir_ts::DoWhileOp::getSuccessorRegions(std::optional<unsigned> index, ArrayRef<Attribute> operands,
                                             SmallVectorImpl<RegionSuccessor> &regions)
{
    (void)operands;

    if (!index)
    {
        regions.emplace_back(&getCond(), getCond().getArguments());
        return;
    }

    // TODO: review it
    assert(*index < 2 && "there are only two regions in a DoWhileOp");
    if (*index == 1)
    {
        regions.emplace_back(&getBody(), getBody().getArguments());
        regions.emplace_back(getResults());
        return;
    }

    regions.emplace_back(&getCond(), getCond().getArguments());
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::ForOp::getSuccessorEntryOperands(std::optional<unsigned int> index)
{
    assert((!index || *index == 0) && "ForOp is expected to branch only to the first region");

    return getInits();
}

void mlir_ts::ForOp::getSuccessorRegions(std::optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions)
{
    (void)operands;

    if (!index)
    {
        regions.emplace_back(&getCond(), getCond().getArguments());
        return;
    }

    // TODO: review it
    //assert(*index < 2 && "there are only two regions in a ForOp");
    if (*index == 0)
    {
        regions.emplace_back(&getIncr(), getIncr().getArguments());
        regions.emplace_back(&getBody(), getBody().getArguments());
        regions.emplace_back(getResults());
        return;
    }

    regions.emplace_back(&getCond(), getCond().getArguments());
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

void mlir_ts::SwitchOp::getSuccessorRegions(std::optional<unsigned> index, ArrayRef<Attribute> operands,
                                            SmallVectorImpl<RegionSuccessor> &regions)
{
    regions.push_back(RegionSuccessor(&getCasesRegion()));
}

mlir::Block *mlir_ts::SwitchOp::getHeaderBlock()
{
    assert(!getCasesRegion().empty() && "op region should not be empty!");
    // The first block is the loop header block.
    return &getCasesRegion().front();
}

mlir::Block *mlir_ts::SwitchOp::getMergeBlock()
{
    assert(!getCasesRegion().empty() && "op region should not be empty!");
    // The last block is the loop merge block.
    return &getCasesRegion().back();
}

void mlir_ts::SwitchOp::addMergeBlock()
{
    assert(getCasesRegion().empty() && "entry and merge block already exist");
    auto *mergeBlock = new Block();
    getCasesRegion().push_back(mergeBlock);
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
            repResults[en.value().getResultNumber()] = newOp.getResult(en.getIndex());
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
        LLVM_DEBUG(llvm::dbgs() << "\n\n\t IfOp RemoveStaticCondition: \n\n" << op << "'\n";);

        auto constant = op.getCondition().getDefiningOp<mlir_ts::ConstantOp>();
        if (!constant)
        {
            return failure();
        }

        if (constant.getValue().cast<BoolAttr>().getValue())
        {
            replaceOpWithRegion(rewriter, op, op.getThenRegion());
        }
        else if (!op.getElseRegion().empty())
        {
            replaceOpWithRegion(rewriter, op, op.getElseRegion());
        }
        else
        {
            rewriter.eraseOp(op);
        }

        return success();
    }
};
} // namespace

void mlir_ts::IfOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert</*RemoveUnusedResults,*/ RemoveStaticCondition>(context);
}

namespace
{

struct SimplifyStaticExpression : public OpRewritePattern<mlir_ts::LogicalBinaryOp>
{
    using OpRewritePattern<mlir_ts::LogicalBinaryOp>::OpRewritePattern;

    mlir::Attribute UnwrapConstant(mlir::Value op1) const
    {
        mlir::Value opConst = op1;
        while (opConst && opConst.getDefiningOp() && llvm::isa<mlir_ts::CastOp>(opConst.getDefiningOp())) 
        {
            opConst = opConst.getDefiningOp<mlir_ts::CastOp>().getOperand();
        }

        if (!opConst || !opConst.getDefiningOp())
        {
            return mlir::Attribute();
        }

        if (auto constOp = opConst.getDefiningOp<mlir_ts::ConstantOp>())
        {
            return constOp.getValue();
        }

        return mlir::Attribute();
    }

    // TODO: complete it
    std::optional<bool> logicalOpResultOfConstants(unsigned int opCode, mlir::Attribute op1, mlir::Attribute op2) const {

        auto op1Typed = op1.dyn_cast<mlir::TypedAttr>();
        auto op2Typed = op2.dyn_cast<mlir::TypedAttr>();
        if (!op1Typed || !op2Typed)
        {
            return {};
        }
        
        if (op1Typed.getType() != op2Typed.getType())
        {
            return {};
        }

        // strings
        if (op1Typed.isa<mlir::StringAttr>())
        {
            switch ((SyntaxKind)opCode)
            {
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
                return op1Typed.cast<mlir::StringAttr>().getValue().equals(op2Typed.cast<mlir::StringAttr>().getValue());
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
                return !op1Typed.cast<mlir::StringAttr>().getValue().equals(op2Typed.cast<mlir::StringAttr>().getValue());
            }
        }

        return {};
    }

    LogicalResult matchAndRewrite(mlir_ts::LogicalBinaryOp op, PatternRewriter &rewriter) const override
    {
        auto op1 = op.getOperand1();
        auto op2 = op.getOperand2();
        
        auto attrVal1 = UnwrapConstant(op1);
        if (!attrVal1)
        {
            return mlir::failure();
        }

        auto attrVal2 = UnwrapConstant(op2);
        if (!attrVal2)
        {
            return mlir::failure();
        }

        auto result = logicalOpResultOfConstants(op.getOpCode(), attrVal1, attrVal2);
        if (result.has_value())
        {
            rewriter.replaceOpWithNewOp<mlir_ts::ConstantOp>(op, mlir_ts::BooleanType::get(rewriter.getContext()), rewriter.getBoolAttr(result.value()));
            return mlir::success();
        }

        return mlir::failure();        
    }
};
} // namespace

void mlir_ts::LogicalBinaryOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
    results.insert<SimplifyStaticExpression>(context);
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

OperandRange mlir_ts::TryOp::getSuccessorEntryOperands(std::optional<unsigned int> index)
{
    assert((!index || *index < 4) && "TryOp is expected to branch only into 3 regions");

    return getOperation()->getOperands();
}

void mlir_ts::TryOp::getSuccessorRegions(std::optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions)
{
    regions.push_back(RegionSuccessor(&getCleanup()));
    regions.push_back(RegionSuccessor(&getCatches()));
    regions.push_back(RegionSuccessor(&getFinally()));
}

//===----------------------------------------------------------------------===//
// LabelOp
//===----------------------------------------------------------------------===//

void mlir_ts::LabelOp::getSuccessorRegions(std::optional<unsigned> index, ArrayRef<Attribute> operands,
                                           SmallVectorImpl<RegionSuccessor> &regions)
{
    regions.push_back(RegionSuccessor(&getLabelRegion()));
}

mlir::Block *mlir_ts::LabelOp::getMergeBlock()
{
    assert(!getLabelRegion().empty() && "op region should not be empty!");
    // The last block is the loop merge block.
    return &getLabelRegion().back();
}

void mlir_ts::LabelOp::addMergeBlock()
{
    assert(getLabelRegion().empty() && "entry and merge block already exist");
    auto *mergeBlock = new Block();
    getLabelRegion().push_back(mergeBlock);
    OpBuilder builder = OpBuilder::atBlockEnd(mergeBlock);

    // Add a ts.merge op into the merge block.
    builder.create<mlir_ts::MergeOp>(getLoc());
}

//===----------------------------------------------------------------------===//
// BodyInternalOp
//===----------------------------------------------------------------------===//

OperandRange mlir_ts::BodyInternalOp::getSuccessorEntryOperands(std::optional<unsigned int> index)
{
    assert((!index || *index == 0) && "BodyInternalOp is expected to branch only to the first region");

    return getODSOperands(0);
}

void mlir_ts::BodyInternalOp::getSuccessorRegions(std::optional<unsigned> index, ArrayRef<Attribute> operands,
                                                  SmallVectorImpl<RegionSuccessor> &regions)
{
    regions.push_back(RegionSuccessor(&getBody()));
}