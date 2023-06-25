#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_

#include "TypeScript/LowerToLLVM/LocationHelper.h"

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMDebugInfoHelper
{
  public:
    LLVMDebugInfoHelper(MLIRContext *context) : context(context)
    {
    }

    LLVM::DITypeAttr getDIType(mlir::Type type)
    {
        LLVM::DITypeAttr diTypeAttr;

        mlir::TypeSwitch<mlir::Type>(type)
            .Case<mlir::IntegerType>([&](auto intType) {  
                diTypeAttr = LLVM::DIBasicTypeAttr::get(context, dwarf::DW_TAG_base_type, StringAttr::get(context, "integer"), intType.getIntOrFloatBitWidth(), 
                    intType.getIntOrFloatBitWidth() == 8 ? dwarf::DW_ATE_signed_char : dwarf::DW_ATE_signed);
            })
            .Case<mlir::FloatType>([&](auto floatType) {  
                diTypeAttr = LLVM::DIBasicTypeAttr::get(context, dwarf::DW_TAG_base_type, StringAttr::get(context, "float"), floatType.getIntOrFloatBitWidth(), dwarf::DW_ATE_float);
            })
            .Case<LLVM::LLVMPointerType>([&](auto pointerType) {  
                diTypeAttr = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_pointer_type, StringAttr::get(context, "pointer"), getDIType(pointerType.getElementType()), 64, 8, 0);
            })
            .Default([&](auto type) { });

        return diTypeAttr;
    }

  private:
    MLIRContext *context;
};

}

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_