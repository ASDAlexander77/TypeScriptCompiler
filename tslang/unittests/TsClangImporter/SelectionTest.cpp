// What gets emitted: without --filter, what the input file declares; with it, what matches; and in
// both cases the types those use, wherever they are declared.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::HasSubstr;
using testing::Not;

namespace
{

const std::vector<std::pair<std::string, std::string>> files = {
    {"other.h", "typedef struct Shared { int id; } Shared;\nint other_unused(void);\n#define OTHER_VALUE 1\n"},
    {"sys/system.h", "struct sys_stat { int size; };\nint sys_call(struct sys_stat *s);\n#define SYS_VALUE 2\n"},
};

} // namespace

TEST(Selection, ByDefaultOnlyTheInputFilesDeclarations)
{
    auto out = generate("#include \"other.h\"\n#include <system.h>\nint mine(void);\n", {}, windowsTarget, files).text;
    EXPECT_THAT(out, HasSubstr("declare function mine(): s32;"));
    EXPECT_THAT(out, Not(HasSubstr("other_unused")));
    EXPECT_THAT(out, Not(HasSubstr("OTHER_VALUE")));
    EXPECT_THAT(out, Not(HasSubstr("sys_call")));
    EXPECT_THAT(out, Not(HasSubstr("SYS_VALUE")));
    EXPECT_THAT(out, Not(HasSubstr("Shared")));
}

TEST(Selection, UsedTypesArePulledInFromAnyHeader)
{
    auto out = generate("#include \"other.h\"\n#include <system.h>\nint use(Shared *s, struct sys_stat *st);\n", {},
                        windowsTarget, files)
                   .text;
    EXPECT_THAT(out, HasSubstr("type Shared = [id: s32];"));
    EXPECT_THAT(out, HasSubstr("type sys_stat = [size: s32];"));
    EXPECT_THAT(out, HasSubstr("declare function use(s: Reference<Shared>, st: Reference<sys_stat>): s32;"));
    EXPECT_THAT(out, Not(HasSubstr("other_unused")));
}

TEST(Selection, FilterSelectsByCNameAnywhere)
{
    tsbindgen::PrintOptions options;
    options.filters = {"sys_*", "mine"};
    auto out = generate("#include \"other.h\"\n#include <system.h>\nint mine(void);\nint yours(void);\n", options,
                        windowsTarget, files)
                   .text;
    EXPECT_THAT(out, HasSubstr("declare function mine(): s32;"));
    EXPECT_THAT(out, HasSubstr("declare function sys_call(s: Reference<sys_stat>): s32;"));
    EXPECT_THAT(out, HasSubstr("type sys_stat = [size: s32];"));
    EXPECT_THAT(out, Not(HasSubstr("yours")));
    EXPECT_THAT(out, Not(HasSubstr("SYS_VALUE")));
}

TEST(Selection, NothingSelectedIsAWarning)
{
    auto out = generate("#include \"other.h\"\n", {}, windowsTarget, files);
    EXPECT_EQ(out.text, "");
    ASSERT_EQ(out.warnings.size(), 1u);
    EXPECT_THAT(out.warnings.front(), HasSubstr("nothing to emit"));

    tsbindgen::PrintOptions options;
    options.filters = {"no_such_*"};
    EXPECT_THAT(generate("int f(void);\n", options).warnings, testing::Contains(HasSubstr("no declaration matches")));
}

TEST(Selection, TypesComeBeforeTheFunctionsThatUseThem)
{
    auto out = text("int first(struct Late *l);\nstruct Late { int x; };\n");
    auto type = out.find("type Late");
    auto function = out.find("declare function first");
    ASSERT_NE(type, std::string::npos);
    ASSERT_NE(function, std::string::npos);
    EXPECT_LT(type, function);
}
