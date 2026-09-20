#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class TypeHelper
{
    MLIRContext *context;

    // Optional on purpose. TypeHelper is constructed at ~152 sites, nearly all of which only ask
    // for width-independent types (getI8Type, getPtrType, getVoidType). Threading CompileOptions
    // through all of them would be a large diff for no benefit, so only the sites that ask for a
    // target width pass it - and getSizeType() asserts rather than guessing when they have not.
    const CompileOptions *compileOptions = nullptr;

  public:
    TypeHelper(OpBuilder &rewriter) : context(rewriter.getContext())
    {
    }

    TypeHelper(MLIRContext *context) : context(context)
    {
    }

    TypeHelper(OpBuilder &rewriter, const CompileOptions &compileOptions)
        : context(rewriter.getContext()), compileOptions(&compileOptions)
    {
    }

    TypeHelper(MLIRContext *context, const CompileOptions &compileOptions)
        : context(context), compileOptions(&compileOptions)
    {
    }

    mlir::Type getBooleanType()
    {
        return mlir_ts::BooleanType::get(context);
    }

    mlir::Type getStringType()
    {
        return mlir_ts::StringType::get(context);
    }

    mlir::Type getCharType()
    {
        return mlir_ts::CharType::get(context);
    }

    mlir::Type getUndefinedType()
    {
        return mlir_ts::UndefinedType::get(context);
    }

    mlir::Type getI8Type()
    {
        return mlir::IntegerType::get(context, 8);
    }

    mlir::Type getI32Type()
    {
        return mlir::IntegerType::get(context, 32);
    }

    mlir::Type getI64Type()
    {
        return mlir::IntegerType::get(context, 64);
    }

    // An integer as wide as the target's pointer: object sizes, byte counts, GEP indices.
    // Distinct from getI64Type(), which is for values that are 64-bit whatever the target is -
    // the language's own 64-bit integers and runtime ABI parameters declared int64_t.
    mlir::Type getSizeType()
    {
        assert(compileOptions && "getSizeType() needs the target: construct TypeHelper with CompileOptions");
        return mlir::IntegerType::get(context, compileOptions->targetInfo.pointerBits);
    }

    // The integer a pointer converts to under ptrtoint on this target.
    mlir::Type getPointerIntType()
    {
        return getSizeType();
    }

    mlir::Type getF32Type()
    {
        return Float32Type::get(context);
    }

    mlir::Type getF64Type()
    {
        return Float64Type::get(context);
    }

    mlir::Type getIndexType()
    {
        return IndexType::get(context);
    }

    mlir::IntegerAttr getStructIndexAttrValue(int32_t value)
    {
        return IntegerAttr::get(getI32Type(), APInt(32, value));
    }

    IntegerAttr getIndexAttrValue(mlir::Type llvmIndexType, int64_t value)
    {
        return IntegerAttr::get(llvmIndexType, APInt(llvmIndexType.getIntOrFloatBitWidth(), value));
    }

    mlir::Type getLLVMBoolType()
    {
        return mlir::IntegerType::get(context, 1 /*, IntegerType::SignednessSemantics::Unsigned*/);
    }

    LLVM::LLVMVoidType getVoidType()
    {
        return LLVM::LLVMVoidType::get(context);
    }

    LLVM::LLVMPointerType getPtrType()
    {
        return LLVM::LLVMPointerType::get(context);
    }

    LLVM::LLVMArrayType getI8Array(unsigned size)
    {
        return LLVM::LLVMArrayType::get(getI8Type(), size);
    }

    LLVM::LLVMArrayType getI32Array(unsigned size)
    {
        return LLVM::LLVMArrayType::get(getI32Type(), size);
    }

    LLVM::LLVMArrayType getArrayType(mlir::Type elementType, size_t size)
    {
        return LLVM::LLVMArrayType::get(elementType, size);
    }

    LLVM::LLVMFunctionType getFunctionType(mlir::Type result, ArrayRef<mlir::Type> arguments, bool isVarArg = false)
    {
        return LLVM::LLVMFunctionType::get(result, arguments, isVarArg);
    }

    LLVM::LLVMFunctionType getFunctionType(ArrayRef<mlir::Type> arguments, bool isVarArg = false)
    {
        return LLVM::LLVMFunctionType::get(getVoidType(), arguments, isVarArg);
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEHELPER_H_
