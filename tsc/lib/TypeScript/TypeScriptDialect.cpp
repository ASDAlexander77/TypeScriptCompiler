#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScript dialect.
//===----------------------------------------------------------------------===//
#include "TypeScript/TypeScriptOpsDialect.cpp.inc"

#ifndef DISABLE_CUSTOM_CLASSSTORAGESTORAGE
namespace mlir
{
namespace typescript
{    

FlatSymbolRefAttr ObjectStorageType::getName() const
{
    return getImpl()->name;
}

::llvm::ArrayRef<::mlir::typescript::FieldInfo> ObjectStorageType::getFields() const
{
    return getImpl()->fields;
}

LogicalResult ObjectStorageType::setFields(::llvm::ArrayRef<::mlir::typescript::FieldInfo> newFields)
{
    return Base::mutate(newFields);
}

// ClassStorage
FlatSymbolRefAttr ClassStorageType::getName() const
{
    return getImpl()->name;
}

::llvm::ArrayRef<::mlir::typescript::FieldInfo> ClassStorageType::getFields() const
{
    return getImpl()->fields;
}

LogicalResult ClassStorageType::setFields(::llvm::ArrayRef<::mlir::typescript::FieldInfo> newFields)
{
    return Base::mutate(newFields);
}

} // namespace typescript
} // namespace mlir
#endif

#define GET_TYPEDEF_CLASSES
#include "TypeScript/TypeScriptOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.cpp.inc"

Type mlir_ts::ConstTupleType::parse(AsmParser &parser)
{
    SmallVector<FieldInfo, 4> parameters;
    if (parser.parseLess())
        return Type();
    while (mlir::succeeded(parser.parseOptionalLBrace()))
    {
        Attribute id;
        if (parser.parseAttribute(id))
            return Type();
        if (parser.parseComma())
            return Type();
        Type type;
        if (parser.parseType(type))
            return Type();
        if (parser.parseRBrace())
            return Type();
        parameters.push_back(FieldInfo{id, type, false});
        if (parser.parseOptionalComma())
            break;
    }
    if (parser.parseGreater())
        return Type();
    return get(parser.getContext(), parameters);
}

void mlir_ts::ConstTupleType::print(AsmPrinter &printer) const
{
    printer << "<";
    for (size_t i = 0, e = getImpl()->fields.size(); i < e; i++)
    {
        const auto &field = getImpl()->fields[i];
        if (field.id)
        {
            printer << "{" << field.id << "," << field.type << "}";
        }
        else
        {
            printer << field.type;
        }

        if (i < getImpl()->fields.size() - 1)
            printer << ",";
    }
    printer << ">";
}

Type mlir_ts::TupleType::parse(AsmParser &parser)
{
    SmallVector<FieldInfo, 4> parameters;
    if (parser.parseLess())
        return Type();
    while (mlir::succeeded(parser.parseOptionalLBrace()))
    {
        Attribute id;
        if (parser.parseAttribute(id))
            return Type();
        if (parser.parseComma())
            return Type();
        Type type;
        if (parser.parseType(type))
            return Type();
        if (parser.parseRBrace())
            return Type();
        parameters.push_back(FieldInfo{id, type});
        if (parser.parseOptionalComma())
            break;
    }
    if (parser.parseGreater())
        return Type();
    return get(parser.getContext(), parameters);
}

void mlir_ts::TupleType::print(AsmPrinter &printer) const
{
    printer << "<";
    for (size_t i = 0, e = getImpl()->fields.size(); i < e; i++)
    {
        const auto &field = getImpl()->fields[i];
        if (field.id)
        {
            printer << "{" << field.id << "," << field.type << "}";
        }
        else
        {
            printer << field.type;
        }

        if (i < getImpl()->fields.size() - 1)
            printer << ",";
    }
    printer << ">";
}

//===----------------------------------------------------------------------===//
// TypeScriptInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with TypeScript
/// operations.

struct TypeScriptInlinerInterface : public mlir::DialectInlinerInterface
{
    TypeScriptInlinerInterface(Dialect *dialect) : DialectInlinerInterface(dialect)
    {
    }

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// All call operations within TypeScript(but recursive) can be inlined.
    // TODO: find out how to prevent recursive calls in better way
    // TODO: something happening when inlining class methods
    bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable, bool wouldBeCloned) const final
    {
        LLVM_DEBUG(llvm::dbgs() << "!! Legal To Inline(call): TRUE = " << *call << "\n";);
        return true;
    }

    bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned, IRMapping &valueMapping) const final
    {
        if (auto funcOp = src->getParentOfType<mlir_ts::FuncOp>())
        {
            if (funcOp->getAttr("noinline"))
            {
                LLVM_DEBUG(llvm::dbgs() << "!! is Legal To Inline (region): FALSE" << funcOp << "\n";);
                return false;
            }
        }

        if (auto funcOp = src->getParentOfType<mlir_ts::FuncOp>())
        {
            auto condition = !(funcOp.getPersonality().has_value() && funcOp.getPersonality().value());
            LLVM_DEBUG(llvm::dbgs() << "!! is Legal To Inline (region): " << (condition ? "TRUE" : "FALSE") << " " << funcOp << " = "
                                    << "\n";);
            return condition;
        }

        return false;
    }

    /// here if we return false for any of op, then whole funcOp will not be inlined
    /// needed to decided if to allow inlining or not
    bool isLegalToInline(mlir::Operation *op, mlir::Region *region, bool, IRMapping &) const final
    {
        // auto condition = true;
        // ignore all functions until you find out how to resolve issue with recursive calls
        auto condition = !isa<mlir_ts::SymbolCallInternalOp>(op);

        // do not inline if func body has TryOp
        condition &= !isa<mlir_ts::TryOp>(op);
        condition &= !isa<mlir_ts::CatchOp>(op);
        condition &= !isa<mlir_ts::ThrowOp>(op);
        condition &= !isa<mlir_ts::LandingPadOp>(op);
        condition &= !isa<mlir_ts::BeginCatchOp>(op);
        condition &= !isa<mlir_ts::EndCatchOp>(op);

        LLVM_DEBUG(llvm::dbgs() << "!! is Legal To Inline (op): " << (condition ? "TRUE" : "FALSE") << " " << *op << " = "
                                << "\n";);

        return condition;
    }

    //===--------------------------------------------------------------------===//
    // Transformation Hooks
    //===--------------------------------------------------------------------===//

    void handleTerminator(mlir::Operation *op, mlir::ArrayRef<Value> valuesToRepl) const final
    {
        LLVM_DEBUG(llvm::dbgs() << "!! handleTerminator: " << *op << "\n";);

        // we need to handle it when inlining function
        // Only "ts.returnVal" needs to be handled here.
        if (auto returnOp = dyn_cast<mlir_ts::ReturnInternalOp>(op))
        {
            LLVM_DEBUG(llvm::dbgs() << "!! handleTerminator counts: Ret ops: " << returnOp.getNumOperands()
                                    << ", values to replace: " << valuesToRepl.size() << "\n";);
            // Replace the values directly with the return operands.
            if (returnOp.getNumOperands() == valuesToRepl.size())
            {
                for (const auto &it : llvm::enumerate(returnOp.getOperands()))
                {
                    valuesToRepl[it.index()].replaceAllUsesWith(it.value());
                }
            }
            else
            {
                OpBuilder builder(op);
                for (const auto &it : llvm::enumerate(valuesToRepl))
                {
                    auto undefVal = builder.create<mlir_ts::UndefOp>(op->getLoc(), it.value().getType());
                    valuesToRepl[it.index()].replaceAllUsesWith(undefVal);
                }
            }
        }
    }

    void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final
    {
        LLVM_DEBUG(llvm::dbgs() << "!! handleTerminator: " << *op << "\n"
                                << "!! Block: ";
                   newDest->dump(); llvm::dbgs() << "\n";);

        auto voidType = mlir_ts::VoidType::get(op->getContext());

        // remove all args with ts.Void
        for (unsigned int argIndex = 0; argIndex < newDest->getNumArguments(); argIndex++)
        {
            if (newDest->getArgument(argIndex).getType() == voidType)
            {
                newDest->eraseArgument(argIndex);
                argIndex--;
            }
        }

        // we need to handle it when inlining function
        // Only "ts.returnVal" needs to be handled here.
        if (auto returnOp = dyn_cast<mlir_ts::ReturnInternalOp>(op))
        {
            // Replace the values directly with the return operands.
            OpBuilder builder(op);
            builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
            op->erase();
        }
    }

    /// Attempts to materialize a conversion for a type mismatch between a call
    /// from this dialect, and a callable region. This method should generate an
    /// operation that takes 'input' as the only operand, and produces a single
    /// result of 'resultType'. If a conversion can not be generated, nullptr
    /// should be returned.
    mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input, mlir::Type resultType,
                                               mlir::Location conversionLoc) const final
    {
        return builder.create<mlir_ts::CastOp>(conversionLoc, resultType, input);
    }
};

void mlir_ts::TypeScriptDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "TypeScript/TypeScriptOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "TypeScript/TypeScriptOpsTypes.cpp.inc"
        >();
    addInterfaces<TypeScriptInlinerInterface>();
}

// The functions don't need to be in the header file, but need to be in the mlir
// namespace. Declare them here, then define them immediately below. Separating
// the declaration and definition adheres to the LLVM coding standards.
namespace mlir
{
namespace typescript
{
// FieldInfo is used as part of a parameter, so equality comparison is compulsory.
static bool operator==(const FieldInfo &a, const FieldInfo &b);
// FieldInfo is used as part of a parameter, so a hash will be computed.
static llvm::hash_code hash_value(const FieldInfo &fi);
} // namespace typescript
} // namespace mlir

// FieldInfo is used as part of a parameter, so equality comparison is
// compulsory.
static bool mlir_ts::operator==(const FieldInfo &a, const FieldInfo &b)
{
    return a.id == b.id && a.type == b.type;
}

// FieldInfo is used as part of a parameter, so a hash will be computed.
static llvm::hash_code mlir_ts::hash_value(const FieldInfo &fi)
{
    return llvm::hash_combine(fi.id, fi.type);
}
