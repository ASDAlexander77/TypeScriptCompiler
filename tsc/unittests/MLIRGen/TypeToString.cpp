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

TEST_F(TypeToNameTest, bool_name) {
    test(mlir_ts::BooleanType::get(getContext()), "boolean");
}
