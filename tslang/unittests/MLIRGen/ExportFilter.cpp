#include "TypeScript/MLIRLogic/MLIRExportFilter.h"

#include "gtest/gtest.h"

using namespace typescript;

namespace
{
bool exported(std::vector<std::string> filters, llvm::StringRef name, llvm::StringRef fullName, bool keyword)
{
    return isExportedByFilters(filters, name, fullName, keyword);
}
} // namespace

TEST(ExportFilterTest, no_filters_follow_export_keyword)
{
    EXPECT_TRUE(exported({}, "add", "M.add", true));
    EXPECT_FALSE(exported({}, "add", "M.add", false));
}

TEST(ExportFilterTest, all_and_none_keep_their_meaning)
{
    EXPECT_TRUE(exported({"all"}, "add", "M.add", false));
    EXPECT_FALSE(exported({"none"}, "add", "M.add", true));
}

TEST(ExportFilterTest, name_filter_exports_only_matching_names)
{
    EXPECT_TRUE(exported({"add"}, "add", "M.add", false));
    EXPECT_TRUE(exported({"M.add"}, "add", "M.add", false));
    EXPECT_FALSE(exported({"add"}, "sub", "M.sub", true));
    EXPECT_TRUE(exported({"add", "sub"}, "sub", "M.sub", false));
}

TEST(ExportFilterTest, glob_filter_matches_short_or_namespaced_name)
{
    EXPECT_TRUE(exported({"M.*"}, "add", "M.add", false));
    EXPECT_FALSE(exported({"N.*"}, "add", "M.add", true));
    EXPECT_TRUE(exported({"calc_?"}, "calc_1", "M.calc_1", false));
}

TEST(ExportFilterTest, exclusion_only_filters_the_export_keyword)
{
    EXPECT_TRUE(exported({"!sub"}, "add", "M.add", true));
    EXPECT_FALSE(exported({"!sub"}, "add", "M.add", false));
    EXPECT_FALSE(exported({"!sub"}, "sub", "M.sub", true));
}

TEST(ExportFilterTest, exclusion_wins_over_inclusion)
{
    EXPECT_FALSE(exported({"all", "!add"}, "add", "M.add", true));
    EXPECT_TRUE(exported({"all", "!add"}, "sub", "M.sub", false));
    EXPECT_FALSE(exported({"M.*", "!M.internal*"}, "internalHelper", "M.internalHelper", true));
    EXPECT_FALSE(exported({"!add", "add"}, "add", "M.add", true));
}

TEST(ExportFilterTest, unnamed_declaration_matches_only_all)
{
    EXPECT_TRUE(exported({"all"}, "", "", false));
    EXPECT_FALSE(exported({"*"}, "", "", true));
    EXPECT_TRUE(exported({"!add"}, "", "", true));
}
