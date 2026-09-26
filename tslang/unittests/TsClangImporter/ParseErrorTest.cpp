// A header clang rejects, or a command line clang rejects, is an error - never an empty success.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

namespace
{

llvm::Expected<tsbindgen::HeaderModel> parse(const std::string &code, std::vector<std::string> extraArgs = {})
{
    tsbindgen::ParseInput input;
    input.code = code;
    input.fileName = "input.h";
    input.args = {"-x", "c", "--target=x86_64-pc-windows-msvc", "-ffreestanding", "-nostdlibinc", "-resource-dir",
                  TSBINDGEN_TEST_RESOURCE_DIR};
    input.args.insert(input.args.end(), extraArgs.begin(), extraArgs.end());
    return tsbindgen::parseHeader(input);
}

} // namespace

TEST(ParseError, SyntaxError)
{
    auto model = parse("int broken(\n");
    ASSERT_FALSE(static_cast<bool>(model));
    EXPECT_THAT(llvm::toString(model.takeError()), testing::HasSubstr("clang could not parse 'input.h'"));
}

TEST(ParseError, MissingInclude)
{
    auto model = parse("#include \"nowhere.h\"\nint f(void);\n");
    ASSERT_FALSE(static_cast<bool>(model));
    llvm::consumeError(model.takeError());
}

TEST(ParseError, ArgumentClangRejects)
{
    auto model = parse("int f(void);\n", {"--no-such-clang-flag"});
    ASSERT_FALSE(static_cast<bool>(model));
    llvm::consumeError(model.takeError());
}

TEST(ParseError, WarningsAreNotErrors)
{
    auto model = parse("int f(struct Late *l);\n");
    ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
}
