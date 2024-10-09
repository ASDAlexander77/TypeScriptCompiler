#include "enums.h"
#include "dump.h"

#include "TypeScript/MLIRGen.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "mlir/IR/MLIRContext.h"

#include "gmock/gmock.h"

#define DEBUG_TYPE "test"

namespace mlir_ts = mlir::typescript;

struct TypeToNameTest : public testing::Test
{
public:
    void SetUp() override
    {
        context.getOrLoadDialect<mlir::typescript::TypeScriptDialect>();
    }

    std::string getTypeString(mlir::Type type)
    {
        std::stringstream exports;
        MLIRTypeHelper mth(&context);
        mth.printType<std::ostream>(exports, type);
        return exports.str();
    }

    void test(mlir::Type type, std::string expect)
    {
        EXPECT_THAT( getTypeString(type), expect );
    }

    mlir::MLIRContext *getContext()
    {
        return &context;
    }

    template <typename T> T get()
    {
        return T::get(getContext());
    }

    template <typename T, typename E> T getE()
    {
        return T::get(getContext(), get<E>());
    }

    template <typename T> T getTs(::llvm::ArrayRef<mlir::Type> types)
    {
        return T::get(
            getContext(),
            types
        );
    }    

    template <typename T> T getNamedTs(StringRef name, ::llvm::ArrayRef<mlir::Type> types)
    {
        return T::get(
            getContext(),
            mlir::FlatSymbolRefAttr::get(getContext(), name),
            types
        );
    }    

    template <typename T = mlir_ts::FunctionType> T getF(::llvm::ArrayRef<mlir::Type> types, ::llvm::ArrayRef<mlir::Type> results, bool isVar = false)
    {
        return T::get(getContext(), types, results, isVar);    
    }

    template <typename E> mlir_ts::ArrayType getArray()
    {
        return getE<mlir_ts::ArrayType, E>();
    }

    template <typename E> mlir_ts::OptionalType getOpt()
    {
        return getE<mlir_ts::OptionalType, E>();
    }

    mlir_ts::TupleType getTuple(ArrayRef<::mlir::typescript::FieldInfo> fields)
    {
        return mlir_ts::TupleType::get(getContext(), fields);
    }

    mlir_ts::TypeReferenceType getTypeRef(StringRef name)
    {
        return getNamedTs<mlir_ts::TypeReferenceType>(name, {});
    }

    mlir_ts::TypeReferenceType getTypeRef(StringRef name, ::llvm::ArrayRef<mlir::Type> types)
    {
        return getNamedTs<mlir_ts::TypeReferenceType>(name, types);
    }

    mlir_ts::InterfaceType getInterface(StringRef name)
    {
        return mlir_ts::InterfaceType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), name));
    }

    mlir_ts::ClassType getClass(StringRef name)
    {
        return mlir_ts::ClassType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), name), mlir_ts::ClassStorageType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), name)));
    }

    mlir::MLIRContext context;
};

TEST_F(TypeToNameTest, basic_names) {

    test(get<mlir_ts::UndefinedType>(), "undefined");
    test(get<mlir_ts::NullType>(), "null");
    test(get<mlir_ts::BooleanType>(), "boolean");
    test(get<mlir_ts::NumberType>(), "number");
    test(get<mlir_ts::StringType>(), "string");
    test(get<mlir_ts::AnyType>(), "any");
    test(getE<mlir_ts::ObjectType, mlir_ts::AnyType>(), "object");
    test(get<mlir_ts::NeverType>(), "never");
    test(get<mlir_ts::UnknownType>(), "unknown");
    test(get<mlir_ts::VoidType>(), "void");
    // support types
    test(get<mlir_ts::OpaqueType>(), "Opaque");
}

TEST_F(TypeToNameTest, array_name) {

    test(getArray<mlir_ts::BooleanType>(), "boolean[]");
    test(getArray<mlir_ts::NumberType>(), "number[]");
    test(getArray<mlir_ts::StringType>(), "string[]");
    test(getArray<mlir_ts::AnyType>(), "any[]");
}

TEST_F(TypeToNameTest, tuple_name) {

    SmallVector<::mlir::typescript::FieldInfo> fields;
    fields.push_back({ mlir::Attribute(), get<mlir_ts::NumberType>(), false });
    fields.push_back({ mlir::Attribute(), get<mlir_ts::StringType>(), false });
    test(getTuple(fields), "[number, string]");
}

TEST_F(TypeToNameTest, tuple_with_names) {

    SmallVector<::mlir::typescript::FieldInfo> fields;
    fields.push_back({ mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), 1), get<mlir_ts::NumberType>(), false });
    fields.push_back({ mlir::StringAttr::get(getContext(), "size"), get<mlir_ts::NumberType>(), false });
    fields.push_back({ mlir::StringAttr::get(getContext(), "name"), get<mlir_ts::StringType>(), false });
    test(getTuple(fields), "[1:number, size:number, name:string]");
}

TEST_F(TypeToNameTest, optinal_name) {

    test(getOpt<mlir_ts::BooleanType>(), "boolean | undefined");
}

TEST_F(TypeToNameTest, func_name) {

    auto funcTypeVoid = getF({}, {});
    test(funcTypeVoid, "() => void");

    auto funcType2 = getF( 
        {
            get<mlir_ts::NumberType>(), 
            getArray<mlir_ts::AnyType>()
        }, 
        {
            get<mlir_ts::StringType>()
        }
    );

    test(funcType2, "(p0: number, p1: any[]) => string");
}

TEST_F(TypeToNameTest, func_variadic_name) {

    auto funcType = getF(
        {
            get<mlir_ts::NumberType>(), 
            getArray<mlir_ts::AnyType>()
        }, 
        {
            get<mlir_ts::StringType>()
        }, 
        true);

    test(funcType, "(p0: number, ...p1: any[]) => string");
}

TEST_F(TypeToNameTest, union_names) {

    test(getTs<mlir_ts::UnionType>(
        {   
            get<mlir_ts::NumberType>(), 
            get<mlir_ts::StringType>()
        }), "number | string");
}

TEST_F(TypeToNameTest, intersect_names) {

    test(getTs<mlir_ts::IntersectionType>( 
        {   
            get<mlir_ts::NumberType>(), 
            get<mlir_ts::StringType>()
        }), "number & string");
}

TEST_F(TypeToNameTest, typeref_names) {

    test(getTypeRef("type1"), "type1");

    test(getTypeRef("type1", {   
                get<mlir_ts::NumberType>(), 
                get<mlir_ts::StringType>()
        }),
        "type1<number, string>"
    );
}

TEST_F(TypeToNameTest, interface_name) {

    test(getInterface("type1"), "type1");
}

TEST_F(TypeToNameTest, class_name) {

    test(getClass("type1"), "type1");
}