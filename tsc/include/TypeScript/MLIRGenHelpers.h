#ifndef MLIR_TYPESCRIPT_MLIRGENHELPERS_H_
#define MLIR_TYPESCRIPT_MLIRGENHELPERS_H_

namespace typescript
{
    template <typename TFrom, typename TTo>
    TTo conv(TFrom value)
    {
        return value.getValue();
    }

    template <>
    int32_t conv<mlir::IntegerAttr, int32_t>(mlir::IntegerAttr value)
    {
        return static_cast<int32_t>(value.getValue().getSExtValue());
    }    

    template <>
    float conv<mlir::FloatAttr, float>(mlir::FloatAttr value)
    {
        return static_cast<float>(value.getValue().convertToFloat());
    }    

    template <typename TVal, typename TValAttr>
    struct CreateArrayAttrFromConstantOpsHelper
    {
        SmallVector<TVal> createArray(SmallVector<mlir::Value> &arrayValues)
        {
            SmallVector<TVal> values;
            for (auto &item : arrayValues)
            {
                auto constOp = cast<mlir_ts::ConstantOp>(item.getDefiningOp());
                if (!constOp)
                {
                    llvm_unreachable("array literal is not implemented(1)");
                    continue;
                }

                auto constValue = constOp.getValue();
                if (!constValue)
                {
                    llvm_unreachable("array literal is not implemented(2)");
                    continue;
                }

                auto valAttr = constOp.getValue().dyn_cast_or_null<TValAttr>();
                if (!valAttr)
                {
                    llvm_unreachable("array literal is not implemented(3)");
                    continue;
                }

                auto value = conv<decltype(valAttr), TVal>(valAttr);

                values.push_back(value);

                item.getDefiningOp()->erase();            
            }

            return values;
        }
    };
}

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
