#define DEBUG_TYPE "test"

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

    mlir::MLIRContext context;
};

TEST_F(TypeToNameTest, basic_names) {

    test(mlir_ts::UndefinedType::get(getContext()), "undefined");
    test(mlir_ts::NullType::get(getContext()), "null");
    test(mlir_ts::BooleanType::get(getContext()), "boolean");
    test(mlir_ts::NumberType::get(getContext()), "number");
    test(mlir_ts::StringType::get(getContext()), "string");
    test(mlir_ts::AnyType::get(getContext()), "any");
    test(mlir_ts::ObjectType::get(getContext(), mlir_ts::AnyType::get(getContext())), "object");
    test(mlir_ts::NeverType::get(getContext()), "never");
    test(mlir_ts::UnknownType::get(getContext()), "unknown");
    test(mlir_ts::VoidType::get(getContext()), "void");
    // support types
    test(mlir_ts::OpaqueType::get(getContext()), "Opaque");
}

TEST_F(TypeToNameTest, array_name) {

    test(mlir_ts::ArrayType::get(getContext(), mlir_ts::BooleanType::get(getContext())), "boolean[]");
    test(mlir_ts::ArrayType::get(getContext(), mlir_ts::NumberType::get(getContext())), "number[]");
    test(mlir_ts::ArrayType::get(getContext(), mlir_ts::StringType::get(getContext())), "string[]");
    test(mlir_ts::ArrayType::get(getContext(), mlir_ts::AnyType::get(getContext())), "any[]");
}

TEST_F(TypeToNameTest, tuple_name) {

    SmallVector<::mlir::typescript::FieldInfo> fields;
    fields.push_back({ mlir::Attribute(), mlir_ts::NumberType::get(getContext()) });
    fields.push_back({ mlir::Attribute(), mlir_ts::StringType::get(getContext()) });
    test(mlir_ts::TupleType::get(getContext(), fields), "[number, string]");
}

TEST_F(TypeToNameTest, tuple_with_names) {

    SmallVector<::mlir::typescript::FieldInfo> fields;
    fields.push_back({ mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), 1), mlir_ts::NumberType::get(getContext()) });
    fields.push_back({ mlir::StringAttr::get(getContext(), "size"), mlir_ts::NumberType::get(getContext()) });
    fields.push_back({ mlir::StringAttr::get(getContext(), "name"), mlir_ts::StringType::get(getContext()) });
    test(mlir_ts::TupleType::get(getContext(), fields), "[1:number, size:number, name:string]");
}

TEST_F(TypeToNameTest, optinal_name) {

    test(mlir_ts::OptionalType::get(getContext(), mlir_ts::BooleanType::get(getContext())), "boolean | undefined");
}

TEST_F(TypeToNameTest, func_name) {

    auto funcTypeVoid = mlir_ts::FunctionType::get(
        getContext(), 
        {
        }, 
        {
        }, false);


    test(funcTypeVoid, "() => void");

    auto funcType2 = mlir_ts::FunctionType::get(
        getContext(), 
        {
            mlir_ts::NumberType::get(getContext()), 
            mlir_ts::ArrayType::get(getContext(), mlir_ts::AnyType::get(getContext()))
        }, 
        {
            mlir_ts::StringType::get(getContext())
        }, false);

    test(funcType2, "(number, any[]) => string");
}

TEST_F(TypeToNameTest, func_variadic_name) {

    auto funcType = mlir_ts::FunctionType::get(
        getContext(), 
        {
            mlir_ts::NumberType::get(getContext()), 
            mlir_ts::ArrayType::get(getContext(), mlir_ts::AnyType::get(getContext()))
        }, 
        {
            mlir_ts::StringType::get(getContext())
        }, true);

    test(funcType, "(number, ...any[]) => string");
}

TEST_F(TypeToNameTest, union_names) {

    test(mlir_ts::UnionType::get(
        getContext(), 
        {   
            mlir_ts::NumberType::get(getContext()), 
            mlir_ts::StringType::get(getContext())
        }), "number | string");
}

TEST_F(TypeToNameTest, intersect_names) {

    test(mlir_ts::IntersectionType::get(
        getContext(), 
        {   
            mlir_ts::NumberType::get(getContext()), 
            mlir_ts::StringType::get(getContext())
        }), "number & string");
}

TEST_F(TypeToNameTest, typeref_names) {

    test(
        mlir_ts::TypeReferenceType::get(
            getContext(),
            mlir::FlatSymbolRefAttr::get(getContext(), "type1"),
            {   
            }
        ), 
        "type1"
    );

    test(
        mlir_ts::TypeReferenceType::get(
            getContext(),
            mlir::FlatSymbolRefAttr::get(getContext(), "type1"),
            {   
                mlir_ts::NumberType::get(getContext()), 
                mlir_ts::StringType::get(getContext())
            }
        ), 
        "type1<number, string>"
    );
}

TEST_F(TypeToNameTest, interface_name) {

    test(
        mlir_ts::InterfaceType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), "type1")),
        "type1"
    );
}

TEST_F(TypeToNameTest, class_name) {

    test(
        mlir_ts::ClassType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), "type1"), mlir_ts::ClassStorageType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), "type1"))),
        "type1"
    );
}