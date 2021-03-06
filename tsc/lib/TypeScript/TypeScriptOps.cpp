#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/Defines.h"

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
