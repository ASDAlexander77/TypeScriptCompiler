#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPECORE_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPECORE_H_

#include "TypeScript/TypeScriptOps.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir"

namespace mlir_ts = mlir::typescript;

namespace typescript
{

class MLIRTypeCore
{  
public:    

    static bool canHaveToPrimitiveMethod(mlir::Type type)
    {
        return isa<mlir_ts::ClassType>(type) 
                || isa<mlir_ts::InterfaceType>(type)
                || isa<mlir_ts::ObjectType>(type)
                || isa<mlir_ts::TupleType>(type)
                || isa<mlir_ts::ConstTupleType>(type);
    }

    static bool shouldCastToNumber(mlir::Type type)
    {
        return isa<mlir::IntegerType>(type) 
            || isa<mlir::FloatType>(type);
    }

    static bool isNullableTypeNoUnion(mlir::Type typeIn)
    {
        if (isa<mlir_ts::NullType>(typeIn) 
            || isa<mlir_ts::StringType>(typeIn) 
            || isa<mlir_ts::ObjectType>(typeIn) 
            || isa<mlir_ts::AnyType>(typeIn) 
            || isa<mlir_ts::ClassType>(typeIn) 
            || isa<mlir_ts::UnknownType>(typeIn)
            || isa<mlir_ts::RefType>(typeIn)
            || isa<mlir_ts::ValueRefType>(typeIn)
            || isa<mlir_ts::OpaqueType>(typeIn))
        {
            return true;            
        }

        return false;
    }

    static bool isNullableType(mlir::Type typeIn)
    {
        if (MLIRTypeCore::isNullableTypeNoUnion(typeIn))
        {
            return true;            
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(typeIn))
        {
            return llvm::any_of(unionType.getTypes(), [&](mlir::Type t) { return isNullableType(t); });
        }

        return false;
    }

    // TODO: should be in separate static class
    static bool isNullableOrOptionalType(mlir::Type typeIn)
    {
        if (isa<mlir_ts::NullType>(typeIn) 
            || isa<mlir_ts::UndefinedType>(typeIn) 
            || isa<mlir_ts::StringType>(typeIn) 
            || isa<mlir_ts::ObjectType>(typeIn) 
            || isa<mlir_ts::ClassType>(typeIn) 
            || isa<mlir_ts::OptionalType>(typeIn)
            || isa<mlir_ts::AnyType>(typeIn)
            || isa<mlir_ts::UnknownType>(typeIn)
            || isa<mlir_ts::RefType>(typeIn)
            || isa<mlir_ts::ValueRefType>(typeIn))
        {
            return true;            
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(typeIn))
        {
            return llvm::any_of(unionType.getTypes(), [&](mlir::Type t) { return isNullableOrOptionalType(t); });
        }

        return false;
    }
};

} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPECORE_H_
