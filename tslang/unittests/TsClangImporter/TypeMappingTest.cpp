// One test per row of the spec's type-mapping table.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::HasSubstr;
using testing::Not;

TEST(TypeMapping, IntegersHaveTheTargetsExactWidth)
{
    auto header = "#include <stdint.h>\n"
                  "int f(signed char a, unsigned short b, long c, unsigned long long d, int8_t e, uint32_t g);\n";

    EXPECT_THAT(text(header, windowsTarget),
                HasSubstr("declare function f(a: s8, b: u16, c: s32, d: u64, e: s8, g: u32): s32;"));
    // C long is 32-bit on Windows and 64-bit on Linux: never `long` in the output
    EXPECT_THAT(text(header, linuxTarget),
                HasSubstr("declare function f(a: s8, b: u16, c: s64, d: u64, e: s8, g: u32): s32;"));
}

TEST(TypeMapping, StructFieldsHaveTheTargetsWidths)
{
    auto header = "struct Sized { long a; char b; };\n";
    EXPECT_THAT(text(header, windowsTarget), HasSubstr("type Sized = [a: s32, b: s8];"));
    EXPECT_THAT(text(header, linuxTarget), HasSubstr("type Sized = [a: s64, b: s8];"));
}

TEST(TypeMapping, SizeTypesAreIndex)
{
    EXPECT_THAT(text("#include <stddef.h>\n#include <stdint.h>\n"
                     "size_t f(ptrdiff_t a, intptr_t b, uintptr_t c);\n"),
                HasSubstr("declare function f(a: index, b: index, c: index): index;"));
}

TEST(TypeMapping, FloatsAndBool)
{
    EXPECT_THAT(text("#include <stdbool.h>\nbool f(float a, double b, _Bool c);\n"),
                HasSubstr("declare function f(a: f32, b: f64, c: boolean): boolean;"));
}

TEST(TypeMapping, CharPointersAreStrings)
{
    EXPECT_THAT(text("const char *f(char *a, const char *b, char **c);\n"),
                HasSubstr("declare function f(a: string, b: string, c: Reference<string>): string;"));
}

TEST(TypeMapping, PointerToScalarOrStructIsReference)
{
    auto out = text("struct Point { int x; int y; };\nvoid f(int *a, struct Point *p, double *d);\n");
    EXPECT_THAT(out, HasSubstr("type Point = [x: s32, y: s32];"));
    EXPECT_THAT(out, HasSubstr("declare function f(a: Reference<s32>, p: Reference<Point>, d: Reference<f64>): void;"));
}

TEST(TypeMapping, VoidPointerAndIncompleteStructAreOpaque)
{
    auto out = text("typedef struct Counter Counter;\nvoid *f(void *a, Counter *c, struct Other *o);\n");
    EXPECT_THAT(out, HasSubstr("type Counter = Opaque;"));
    EXPECT_THAT(out, HasSubstr("type Other = Opaque;"));
    EXPECT_THAT(out, HasSubstr("declare function f(a: Opaque, c: Counter, o: Other): Opaque;"));
}

TEST(TypeMapping, PointerToCxxClassIsOpaque)
{
    auto out = generate("class Engine { public: virtual ~Engine(); int power; };\n"
                        "extern \"C\" int engine_power(Engine *e);\n",
                        {}, windowsTarget, {}, "c++")
                   .text;
    EXPECT_THAT(out, HasSubstr("type Engine = Opaque;"));
    EXPECT_THAT(out, HasSubstr("declare function engine_power(e: Engine): s32;"));
}

// A class in a namespace or a template specialization has no file-scope TS declaration to name
// (found in the default library: regexp_* take a std::cmatch *), so a pointer to one is plain
// Opaque, with the C++ type in a comment - never a name the output does not declare.
TEST(TypeMapping, PointerToANamespacedOrTemplateClassIsPlainOpaque)
{
    auto out = generate("namespace ns { class Engine { public: virtual ~Engine(); }; }\n"
                        "template <typename T> struct Box { T value; };\n"
                        "typedef Box<char> CharBox;\n"
                        "extern \"C\" int use(ns::Engine *e, Box<int> *b, CharBox *c);\n",
                        {}, windowsTarget, {}, "c++")
                   .text;
    EXPECT_THAT(out, HasSubstr("declare function use(e: Opaque /* ns::Engine */, b: Opaque /* Box<int> */, "
                               "c: Opaque /* CharBox */): s32;"));
    EXPECT_THAT(out, Not(HasSubstr("type Engine")));
    EXPECT_THAT(out, Not(HasSubstr("type Box")));
}

TEST(TypeMapping, FunctionPointerIsOpaqueWithItsSignature)
{
    auto out = text("int apply(int (*fn)(int, const char *), int v);\ntypedef void (*callback_t)(void *);\n"
                    "void on(callback_t cb);\n");
    EXPECT_THAT(out, HasSubstr("declare function apply(fn: Opaque /* (p0: s32, p1: string) => s32 */, v: s32): s32;"));
    EXPECT_THAT(out, HasSubstr("type callback_t = Opaque; // (p0: Opaque) => void"));
    EXPECT_THAT(out, HasSubstr("declare function on(cb: callback_t): void;"));
    EXPECT_THAT(out, HasSubstr("pass a function without captures, as `fn as Opaque`"));
}

TEST(TypeMapping, CompleteStructIsANamedTuple)
{
    EXPECT_THAT(text("#include <stdint.h>\ntypedef struct { int8_t a; int32_t b; int64_t c; } Mixed;\n"
                     "struct Node { int value; struct Node *next; };\n"),
                testing::AllOf(HasSubstr("type Mixed = [a: s8, b: s32, c: s64];"),
                               HasSubstr("type Node = [value: s32, next: Opaque /* Reference<Node>: a type cannot refer to itself */];")));
}

// tslang cannot declare a type that refers to itself (the compiler dies on one), so every pointer
// field that leads back to its own struct - directly, through another struct, or through a typedef
// - is Opaque
TEST(TypeMapping, TypeCyclesBecomeOpaque)
{
    auto out = text("struct A { struct B *b; int x; };\nstruct B { struct A *a; };\n"
                    "typedef struct List *ListPtr;\nstruct List { ListPtr next; int v; };\n"
                    "void walk(struct A *a, ListPtr l);\n");
    EXPECT_THAT(out, HasSubstr("type A = [b: Opaque /* Reference<B>: a type cannot refer to itself */, x: s32];"));
    EXPECT_THAT(out, HasSubstr("type B = [a: Opaque /* Reference<A>: a type cannot refer to itself */];"));
    EXPECT_THAT(out, HasSubstr("type List = [next: Opaque /* Reference<List>: a type cannot refer to itself */, v: s32];"));
    EXPECT_THAT(out, HasSubstr("type ListPtr = Reference<List>;"));
    EXPECT_THAT(out, HasSubstr("declare function walk(a: Reference<A>, l: ListPtr): void;"));
    EXPECT_THAT(out, Not(HasSubstr("fn as Opaque")));
}

TEST(TypeMapping, CEnumIsATsEnum)
{
    auto out = text("enum Color { RED, GREEN = 5, BLUE };\nint f(enum Color c);\n");
    EXPECT_THAT(out, HasSubstr("enum Color { RED = 0, GREEN = 5, BLUE = 6 }"));
    EXPECT_THAT(out, HasSubstr("declare function f(c: Color): s32;"));
}

TEST(TypeMapping, FixedNonIntEnumIsATypeAndConstants)
{
    auto out = generate("#include <stdint.h>\n"
                        "enum Mode : uint8_t { Off = 0, On = 1 };\n"
                        "enum class Level : int16_t { Low = -1, High = 2 };\n"
                        "extern \"C\" void set(Mode m, Level l);\n",
                        {}, windowsTarget, {}, "c++")
                   .text;
    EXPECT_THAT(out, HasSubstr("type Mode = u8;\nconst Off = 0;\nconst On = 1;\n"));
    EXPECT_THAT(out, HasSubstr("type Level = s16;\nconst Level_Low = -1;\nconst Level_High = 2;\n"));
    EXPECT_THAT(out, HasSubstr("declare function set(m: Mode, l: Level): void;"));
}

TEST(TypeMapping, VarargsKeepTheFixedParameters)
{
    EXPECT_THAT(text("int print_all(const char *format, ...);\n"),
                HasSubstr("@varargs declare function print_all(format: string): s32;"));
}

TEST(TypeMapping, LiteralMacrosAreConstants)
{
    auto out = text("#define ANSWER 42\n#define NEGATIVE (-7)\n#define MASK 0xFFu\n#define RATIO 1.5\n"
                    "#define WHOLE 2.0\n#define NAME \"fix\\\"ture\"\n");
    EXPECT_THAT(out, HasSubstr("const ANSWER = 42;"));
    EXPECT_THAT(out, HasSubstr("const NEGATIVE = -7;"));
    EXPECT_THAT(out, HasSubstr("const MASK = 255;"));
    EXPECT_THAT(out, HasSubstr("const RATIO = 1.5;"));
    EXPECT_THAT(out, HasSubstr("const WHOLE = 2.0;"));
    EXPECT_THAT(out, HasSubstr("const NAME = \"fix\\\"ture\";"));
}

TEST(TypeMapping, MacrosFollowUndefAndRedefinition)
{
    auto out = text("#define GONE 1\n#undef GONE\n#define LEVEL 1\n#undef LEVEL\n#define LEVEL 2\n");
    EXPECT_THAT(out, Not(HasSubstr("GONE")));
    EXPECT_THAT(out, HasSubstr("const LEVEL = 2;"));
    EXPECT_THAT(out, Not(HasSubstr("const LEVEL = 1;")));
}

TEST(TypeMapping, TypedefIsATypeAliasUnlessItNamesTheStruct)
{
    auto out = text("typedef struct Point { int x; } Point;\ntypedef struct Point Pt;\ntypedef unsigned int handle_t;\n"
                    "void f(Pt *p, handle_t h);\n");
    EXPECT_THAT(out, HasSubstr("type Point = [x: s32];"));
    EXPECT_THAT(out, Not(HasSubstr("type Point = Point")));
    EXPECT_THAT(out, HasSubstr("type Pt = Point;"));
    EXPECT_THAT(out, HasSubstr("type handle_t = u32;"));
    EXPECT_THAT(out, HasSubstr("declare function f(p: Reference<Point>, h: handle_t): void;"));
}

TEST(TypeMapping, ParameterNames)
{
    EXPECT_THAT(text("void f(int, int delete, int function, int value);\n"),
                HasSubstr("declare function f(p0: s32, delete_: s32, function_: s32, value: s32): void;"));
}

// tslang ignores a `type` alias whose name is one of its own types - `typedef int boolean` (libjpeg)
// would bind an int result as boolean's one bit - so such a typedef is looked through and never
// emitted, and a struct of such a name is renamed
TEST(TypeMapping, BuiltinTypeNamesAreNeverRedeclared)
{
    auto out = text("typedef int boolean;\nboolean is_ok(double n);\nstruct string { int len; };\n"
                    "void use(struct string *s);\n");
    EXPECT_THAT(out, Not(HasSubstr("type boolean")));
    EXPECT_THAT(out, HasSubstr("declare function is_ok(n: f64): s32;"));
    EXPECT_THAT(out, HasSubstr("type string_ = [len: s32];"));
    EXPECT_THAT(out, HasSubstr("declare function use(s: Reference<string_>): void;"));
}

// an enum that is declared but never defined has no enumerators to emit: its values are integers
TEST(TypeMapping, EnumWithoutDefinitionIsItsInteger)
{
    auto out = generate("enum Mode : short;\nextern \"C\" void set_mode(Mode m);\n", {}, windowsTarget, {}, "c++").text;
    EXPECT_THAT(out, HasSubstr("declare function set_mode(m: s16): void;"));
    EXPECT_THAT(out, Not(HasSubstr("Mode")));
}
