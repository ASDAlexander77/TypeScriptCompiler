// Every skip reason in the spec: the declaration is left out with a `// skipped:` line and a
// warning, generation carries on, and a skipped struct is still Opaque for its pointer users.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::Contains;
using testing::HasSubstr;
using testing::Not;

TEST(Skip, UnionIsSkippedButItsPointerUsersStay)
{
    auto out = generate("union Value { int i; float f; };\nvoid set(union Value *v);\nvoid pass(union Value v);\n");
    EXPECT_THAT(out.text, HasSubstr("// skipped: Value \xE2\x80\x94 union\ntype Value = Opaque;"));
    EXPECT_THAT(out.text, HasSubstr("declare function set(v: Value): void;"));
    EXPECT_THAT(out.text, HasSubstr("// skipped: pass \xE2\x80\x94 parameter 'v': union passed by value"));
    EXPECT_THAT(out.warnings, Contains("skipped Value: union"));
}

TEST(Skip, StructWithABitfieldOrAnUnmappableField)
{
    auto out = text("struct Flags { unsigned a : 1; };\nstruct Wide { long double x; };\n"
                    "struct Grid { int cells[4]; };\n");
    EXPECT_THAT(out, HasSubstr("// skipped: Flags \xE2\x80\x94 bitfield 'a'"));
    EXPECT_THAT(out, HasSubstr("// skipped: Wide \xE2\x80\x94 field 'x': long double"));
    EXPECT_THAT(out, HasSubstr("// skipped: Grid \xE2\x80\x94 field 'cells': array"));
}

TEST(Skip, PackedStruct)
{
    EXPECT_THAT(text("struct __attribute__((packed)) Packed { char c; int i; };\n"),
                HasSubstr("// skipped: Packed \xE2\x80\x94 packed or over-aligned layout"));
    // how real headers usually pack
    EXPECT_THAT(text("#pragma pack(push, 1)\nstruct Wire { char c; int i; };\n#pragma pack(pop)\n"
                     "struct Natural { char c; int i; };\n"),
                testing::AllOf(HasSubstr("// skipped: Wire \xE2\x80\x94 packed or over-aligned layout"),
                               HasSubstr("type Natural = [c: s8, i: s32];")));
}

TEST(Skip, LongDouble)
{
    EXPECT_THAT(text("long double f(long double x);\n"),
                HasSubstr("// skipped: f \xE2\x80\x94 parameter 'x': long double"));
}

TEST(Skip, StructByValue)
{
    auto out = text("struct Point { int x; int y; };\nstruct Point make(int x);\nint sum(struct Point p);\n");
    EXPECT_THAT(out, HasSubstr("type Point = [x: s32, y: s32];"));
    EXPECT_THAT(out, HasSubstr("// skipped: make \xE2\x80\x94 result: struct passed by value"));
    EXPECT_THAT(out, HasSubstr("// skipped: sum \xE2\x80\x94 parameter 'p': struct passed by value"));
}

TEST(Skip, FunctionLikeAndNonLiteralMacros)
{
    auto out = text("#define MAX(a, b) ((a) > (b) ? (a) : (b))\n#define FLAGS (1 << 3)\n#define GUARD_H\n");
    EXPECT_THAT(out, HasSubstr("// skipped: MAX \xE2\x80\x94 function-like macro"));
    EXPECT_THAT(out, HasSubstr("// skipped: FLAGS \xE2\x80\x94 macro is not a literal"));
    EXPECT_THAT(out, Not(HasSubstr("GUARD_H")));
}

TEST(Skip, StaticInlineFunction)
{
    auto out = text("static inline int twice(int x) { return x * 2; }\nstatic int hidden(void);\n");
    EXPECT_THAT(out, HasSubstr("// skipped: twice \xE2\x80\x94 static inline function: no symbol to link against"));
    EXPECT_THAT(out, HasSubstr("// skipped: hidden \xE2\x80\x94 static function: no symbol to link against"));
}

TEST(Skip, OtherTypes)
{
    auto out = text("_Complex double f(void);\nint g();\n");
    EXPECT_THAT(out, HasSubstr("// skipped: f \xE2\x80\x94 result: unsupported type '_Complex double'"));
    EXPECT_THAT(out, HasSubstr("// skipped: g \xE2\x80\x94 function without a prototype"));
}

TEST(Skip, ASkipDoesNotStopGeneration)
{
    auto out = text("union U { int i; };\nint before(void);\nvoid broken(union U u);\nint after(void);\n");
    EXPECT_THAT(out, HasSubstr("declare function before(): s32;"));
    EXPECT_THAT(out, HasSubstr("declare function after(): s32;"));
}

// a macro body is never checked by clang unless used, so a malformed number skips the macro,
// never the whole header
TEST(Skip, MalformedNumericMacroIsSkippedNotAnError)
{
    auto out = text("#define VERSION 1.2.3\n#define OCT 08\nint f(void);\n");
    EXPECT_THAT(out, HasSubstr("// skipped: VERSION \xE2\x80\x94 macro is not a literal"));
    EXPECT_THAT(out, HasSubstr("// skipped: OCT \xE2\x80\x94 macro is not a literal"));
    EXPECT_THAT(out, HasSubstr("declare function f(): s32;"));
}

// tslang calls a declared function with the C convention; a callee that cleans its own stack
// (__stdcall on 32-bit Windows) would corrupt it
TEST(Skip, NonDefaultCallingConventionIsSkipped)
{
    auto out = generate("int __stdcall f(int x);\nint __cdecl g(int x);\n", {}, "i686-pc-windows-msvc").text;
    EXPECT_THAT(out, HasSubstr("// skipped: f \xE2\x80\x94 calling convention stdcall"));
    EXPECT_THAT(out, HasSubstr("declare function g(x: s32): s32;"));
}
