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
    fields.push_back({ mlir::StringAttr::get(getContext(), "size"), mlir_ts::NumberType::get(getContext()) });
    fields.push_back({ mlir::StringAttr::get(getContext(), "name"), mlir_ts::StringType::get(getContext()) });
    test(mlir_ts::TupleType::get(getContext(), fields), "[size:number, name:string]");
}
