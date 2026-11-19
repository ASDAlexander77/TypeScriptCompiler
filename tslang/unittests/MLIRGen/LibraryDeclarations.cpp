#include "TypeScript/LibraryDeclarations.h"

#include "gtest/gtest.h"

using namespace typescript;

// The file of declarations beside a shared library (TypeScript/LibraryDeclarations.h): what is
// printed reads back the same, the blocks of several objects joined into one file read as one
// library, and a file that is not one is told apart.

namespace
{

LibraryModuleDeclarations module(std::string suffix, std::string model)
{
    LibraryModuleDeclarations m;
    m.suffix = std::move(suffix);
    m.memoryModel = std::move(model);
    return m;
}

} // namespace

TEST(LibraryDeclarationsTest, path_beside_the_binary)
{
    EXPECT_EQ(libraryDeclarationsPath("out/foo.dll"), "out/foo.d.ts");
    EXPECT_EQ(libraryDeclarationsPath("libfoo.so"), "libfoo.d.ts");
    EXPECT_EQ(libraryDeclarationsPath("foo.obj"), "foo.d.ts");
}

TEST(LibraryDeclarationsTest, round_trip)
{
    auto m = module("m1_123", "own");
    m.hasGCtors = true;
    m.declarations = "@dllimport\nfunction add(p0 : number) : number;\n";
    m.genericDeclarations = "export class Box<T> { constructor(public v: T) {} }\n";
    m.ownFacts = "add\nP.get\n";

    auto parsed = parseLibraryDeclarations(printLibraryDeclarations({m}));
    ASSERT_TRUE(parsed.has_value());
    ASSERT_EQ(parsed->size(), 1u);
    auto &r = parsed->front();
    EXPECT_EQ(r.suffix, "m1_123");
    EXPECT_EQ(r.memoryModel, "own");
    EXPECT_TRUE(r.hasGCtors);
    EXPECT_EQ(r.declarations, m.declarations);
    EXPECT_EQ(r.genericDeclarations, m.genericDeclarations);
    EXPECT_EQ(r.ownFacts, m.ownFacts);
}

// test-runner joins the files --emit=obj wrote beside each object, and --emit=dll appends them
TEST(LibraryDeclarationsTest, joined_blocks_are_one_library)
{
    auto a = module("a_1", "rc");
    a.declarations = "@dllimport\nfunction fa() : void;\n";
    auto b = module("b_2", "rc");
    b.genericDeclarations = "export class G<T> {}\n";
    b.ownFacts = "fb\n";

    auto parsed = parseLibraryDeclarations(printLibraryDeclarations({a}) + printLibraryDeclarations({b}));
    ASSERT_TRUE(parsed.has_value());
    ASSERT_EQ(parsed->size(), 2u);
    EXPECT_EQ((*parsed)[0].suffix, "a_1");
    EXPECT_EQ((*parsed)[0].declarations, a.declarations);
    EXPECT_TRUE((*parsed)[0].genericDeclarations.empty());
    EXPECT_TRUE((*parsed)[0].ownFacts.empty());
    EXPECT_EQ((*parsed)[1].suffix, "b_2");
    EXPECT_TRUE((*parsed)[1].declarations.empty());
    EXPECT_EQ((*parsed)[1].genericDeclarations, b.genericDeclarations);
    EXPECT_EQ((*parsed)[1].ownFacts, "fb\n");
}

// a file that went through an editor or a Windows checkout
TEST(LibraryDeclarationsTest, crlf_and_bom)
{
    auto m = module("m_9", "gc");
    m.declarations = "@dllimport\nfunction f() : void;\n";
    auto text = printLibraryDeclarations({m});
    std::string crlf;
    for (auto c : text)
    {
        if (c == '\n')
        {
            crlf += '\r';
        }

        crlf += c;
    }

    auto parsed = parseLibraryDeclarations("\xEF\xBB\xBF" + crlf);
    ASSERT_TRUE(parsed.has_value());
    ASSERT_EQ(parsed->size(), 1u);
    EXPECT_EQ(parsed->front().suffix, "m_9");
    EXPECT_EQ(parsed->front().declarations, m.declarations);
}

TEST(LibraryDeclarationsTest, not_a_file_of_declarations)
{
    EXPECT_FALSE(parseLibraryDeclarations("declare function foo(): void;\n").has_value());
    EXPECT_FALSE(isLibraryDeclarationsText(""));
    EXPECT_TRUE(isLibraryDeclarationsText(printLibraryDeclarations({module("x_1", "gc")})));
}
