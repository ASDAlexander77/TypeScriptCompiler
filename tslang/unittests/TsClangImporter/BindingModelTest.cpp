// The model's helpers: how a mapped type is spelled, which declarations it refers to, and the
// names that must not reach TS unchanged.

#include "TsClangImporter/BindingModel.h"

#include "gmock/gmock.h"

using namespace tsbindgen;

TEST(BindingModel, RendersBuiltinNamedAndReference)
{
    auto nameOf = [](const std::string &key) { return "TS_" + nameFromKey(key); };

    EXPECT_EQ(renderType(TsType::builtin("s32"), nameOf), "s32");
    EXPECT_EQ(renderType(TsType::named("struct:Point"), nameOf), "TS_Point");
    EXPECT_EQ(renderType(TsType::reference(TsType::reference(TsType::named("struct:Point"))), nameOf),
              "Reference<Reference<TS_Point>>");
}

TEST(BindingModel, CollectsEachKeyOnce)
{
    std::vector<std::string> keys;
    collectKeys(TsType::reference(TsType::named("struct:Point")), keys);
    collectKeys(TsType::named("struct:Point"), keys);
    collectKeys(TsType::named("enum:Color"), keys);
    collectKeys(TsType::builtin("string"), keys);

    EXPECT_THAT(keys, testing::ElementsAre("struct:Point", "enum:Color"));
}

TEST(BindingModel, KeysNameTheirDeclaration)
{
    EXPECT_EQ(nameFromKey("typedef:callback_t"), "callback_t");
    EXPECT_EQ(nameFromKey("plain"), "plain");
}

TEST(BindingModel, ReservedWordsGetASuffix)
{
    EXPECT_EQ(escapeReserved("delete"), "delete_");
    EXPECT_EQ(escapeReserved("function"), "function_");
    EXPECT_EQ(escapeReserved("yield"), "yield_");
    EXPECT_EQ(escapeReserved("value"), "value");
    EXPECT_EQ(escapeReserved("len"), "len");
}

TEST(BindingModel, SkippedTypeKeepsItsReason)
{
    auto type = TsType::skip("long double");
    EXPECT_TRUE(type.skipped());
    EXPECT_EQ(type.skipReason, "long double");
    EXPECT_FALSE(TsType::builtin("f64").skipped());
}
