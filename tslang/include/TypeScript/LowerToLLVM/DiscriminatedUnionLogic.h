#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_DISCRIMINATEDUNIONLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_DISCRIMINATEDUNIONLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

// A union of object shapes told apart by their first field (MLIRTypeHelper::isFirstFieldDiscriminatedUnion) is
// stored as its largest shape alone: the field is its tag. Which member a value is, is read from that field -
// the first word of every member, of the same type in each - compared with each member's literal.
class DiscriminatedUnionLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    TypeConverterHelper &tch;
    Location loc;
    CompileOptions &compileOptions;

  public:
    DiscriminatedUnionLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc,
                            CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), tch(tch), loc(loc), compileOptions(compileOptions)
    {
    }

    static mlir_ts::LiteralType discriminantOf(mlir::Type member)
    {
        return mlir::cast<mlir_ts::LiteralType>(mlir::cast<mlir_ts::TupleType>(member).getFields().front().type);
    }

    // The index of the member the value at `payloadPtr` holds, an i32: -1 when its first field is no member's
    // (a union that holds nothing yet, zeroed).
    mlir::Value memberIndex(mlir_ts::UnionType unionType, mlir::Value payloadPtr)
    {
        TypeHelper th(rewriter);
        LLVMCodeHelperBase ch(op, rewriter, tch.typeConverter, compileOptions);

        auto members = unionType.getTypes();
        auto elementType = discriminantOf(members.front()).getElementType();
        auto fieldType = tch.convertType(elementType);
        mlir::Value field = rewriter.create<LLVM::LoadOp>(loc, fieldType, payloadPtr);

        auto i32Type = th.getI32Type();
        mlir::Value index = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(-1));

        // a string is compared by its text (a literal of another module is another global); a field never set is null
        mlir::Value text;
        if (isa<mlir_ts::StringType>(elementType))
        {
            auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, th.getPtrType());
            auto isSet = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, field, nullPtr);
            text = rewriter.create<LLVM::SelectOp>(loc, isSet, field, ch.getOrCreateGlobalString(std::string()));
        }

        for (auto [memberIndex, member] : llvm::enumerate(members))
        {
            auto value = discriminantOf(member).getValue();
            mlir::Value isMember;
            if (text)
            {
                auto strcmpFuncOp = ch.getOrInsertFunction(
                    "strcmp", th.getFunctionType(th.getI32Type(), {th.getPtrType(), th.getPtrType()}));
                auto literal = ch.getOrCreateGlobalString(mlir::cast<mlir::StringAttr>(value).getValue().str());
                auto compared = rewriter.create<LLVM::CallOp>(loc, strcmpFuncOp, ValueRange{text, literal});
                auto zero = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
                isMember = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, compared.getResult(), zero);
            }
            else if (auto floatType = dyn_cast<mlir::FloatType>(fieldType))
            {
                mlir::APFloat number(0.0);
                if (auto floatAttr = dyn_cast<mlir::FloatAttr>(value))
                {
                    number = floatAttr.getValue();
                }
                else
                {
                    number = mlir::APFloat((double)mlir::cast<mlir::IntegerAttr>(value).getInt());
                }

                bool losesInfo = false;
                number.convert(floatType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
                auto literal = rewriter.create<LLVM::ConstantOp>(loc, fieldType, rewriter.getFloatAttr(fieldType, number));
                isMember = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::oeq, field, literal);
            }
            else
            {
                auto intType = mlir::cast<mlir::IntegerType>(fieldType);
                auto number = isa<mlir::FloatAttr>(value)
                    ? (int64_t)mlir::cast<mlir::FloatAttr>(value).getValueAsDouble()
                    : mlir::cast<mlir::IntegerAttr>(value).getInt();
                auto literal = rewriter.create<LLVM::ConstantOp>(loc, fieldType, rewriter.getIntegerAttr(intType, number));
                isMember = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, field, literal);
            }

            auto thisIndex = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr((int32_t)memberIndex));
            index = rewriter.create<LLVM::SelectOp>(loc, isMember, thisIndex, index);
        }

        return index;
    }

    // ...of a value: put where its first field can be read
    mlir::Value memberIndexOfValue(mlir_ts::UnionType unionType, mlir::Value value)
    {
        TypeHelper th(rewriter);
        mlir::Value slot;
        {
            // in the function's entry block, as every alloca: one in a loop grows the stack each iteration
            OpBuilder::InsertionGuard guard(rewriter);
            auto *function = rewriter.getInsertionBlock()->getParentOp();
            while (function->getParentOp() && !function->hasTrait<OpTrait::IsIsolatedFromAbove>())
            {
                function = function->getParentOp();
            }

            rewriter.setInsertionPointToStart(&function->getRegion(0).front());
            auto one = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(1));
            slot = rewriter.create<LLVM::AllocaOp>(loc, th.getPtrType(), value.getType(), one);
        }

        rewriter.create<LLVM::StoreOp>(loc, value, slot);
        return memberIndex(unionType, slot);
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_DISCRIMINATEDUNIONLOGIC_H_
