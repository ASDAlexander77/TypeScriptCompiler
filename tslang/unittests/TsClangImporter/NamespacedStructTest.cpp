// A C++ struct in a namespace or a class is declared under its own name, a tuple when its fields
// are where a tuple's are (constructors, methods and member initializers don't move them), and a
// pointer parameter to it is nullable where a C++ reference isn't.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::AllOf;
using testing::Contains;
using testing::HasSubstr;
using testing::Not;

namespace
{

const char *const gfx = "namespace gfx {\n"
                        "struct Color {\n"
                        "    float r, g, b, a;\n"
                        "    Color() : r(0), g(0), b(0), a(0) {}\n"
                        "    Color(float value);\n"
                        "    bool operator==(const Color &other) const;\n"
                        "};\n"
                        "struct Desc { unsigned width = 1; bool cpuAccess = false; Color clear; };\n"
                        "class Device {\n"
                        "public:\n"
                        "    virtual ~Device();\n"
                        "    virtual void clear(const Color &color, Desc *desc) = 0;\n"
                        "};\n"
                        "}\n"
                        "extern \"C\" gfx::Device *make_device();\n";

std::string cpp(const std::string &header, tsbindgen::PrintOptions options = {})
{
    return generate(header, std::move(options), windowsTarget, {}, "c++").text;
}

} // namespace

TEST(NamespacedStruct, IsATuple)
{
    auto out = cpp(gfx);
    EXPECT_THAT(out, HasSubstr("type Color = [r: f32, g: f32, b: f32, a: f32];"));
    EXPECT_THAT(out, HasSubstr("type Desc = [width: u32, cpuAccess: boolean, clear: Color];"));
    EXPECT_THAT(out, HasSubstr("    @slot(1) clear(color: Reference<Color>, desc: Reference<Desc> | null): void;\n"));
}

// --filter picks a struct by its name or its qualified name, and what it uses comes along
TEST(NamespacedStruct, FilterByNameOrQualifiedName)
{
    tsbindgen::PrintOptions qualified;
    qualified.filters = {"gfx::Desc"};
    auto out = cpp(gfx, qualified);
    EXPECT_THAT(out, AllOf(HasSubstr("type Desc = "), HasSubstr("type Color = ")));
    EXPECT_THAT(out, Not(HasSubstr("class Device")));

    tsbindgen::PrintOptions plain;
    plain.filters = {"Color"};
    EXPECT_THAT(cpp(gfx, plain), HasSubstr("type Color = "));
}

// The first of two structs of one name keeps it; the next is named with its namespaces, less the
// --namespace the bindings are wrapped in
TEST(NamespacedStruct, SameNameInTwoNamespaces)
{
    tsbindgen::PrintOptions options;
    options.namespaceName = "outer";
    auto generated = generate("namespace outer {\n"
                              "struct Desc { int x; };\n"
                              "namespace rt { struct Desc { double y; }; }\n"
                              "}\n"
                              "extern \"C\" void use(outer::Desc *a, outer::rt::Desc *b);\n",
                              options, windowsTarget, {}, "c++");
    EXPECT_THAT(generated.text, HasSubstr("    export type Desc = [x: s32];"));
    EXPECT_THAT(generated.text, HasSubstr("    export type rt_Desc = [y: f64];"));
    EXPECT_THAT(generated.text, HasSubstr("use(a: Reference<Desc> | null, b: Reference<rt_Desc> | null): void;"));
    EXPECT_THAT(generated.warnings, Contains("named outer::rt::Desc 'rt_Desc': 'Desc' is taken by outer::Desc"));
}

TEST(NamespacedStruct, NestedInAClass)
{
    auto out = cpp("namespace n { struct Outer { struct Inner { int v; }; Inner inner; int w; }; }\n"
                   "extern \"C\" int use(n::Outer::Inner *i, const n::Outer *o);\n");
    EXPECT_THAT(out, HasSubstr("type Inner = [v: s32];"));
    EXPECT_THAT(out, HasSubstr("type Outer = [inner: Inner, w: s32];"));
    EXPECT_THAT(out, HasSubstr("declare function use(i: Reference<Inner> | null, o: Reference<Outer> | null): s32;"));
}

// A derived struct, and one holding a template's specialization (std::string, a container), are
// named but Opaque: their layouts aren't a tuple's, or not one tsbindgen can spell
TEST(NamespacedStruct, WhatStaysOpaque)
{
    auto out = cpp("template <typename T> struct Vec { T *data; };\n"
                   "namespace n {\n"
                   "struct Base { int a; };\n"
                   "struct Derived : Base { int b; };\n"
                   "struct Holder { Vec<int> values; };\n"
                   "}\n"
                   "extern \"C\" void use(n::Derived *d, n::Holder *h);\n");
    EXPECT_THAT(out, HasSubstr("// skipped: Derived \xE2\x80\x94 derived C++ class\ntype Derived = Opaque;"));
    EXPECT_THAT(out, HasSubstr("// skipped: Holder \xE2\x80\x94 field 'values': C++ class 'struct Vec<int>'\ntype Holder = Opaque;"));
    EXPECT_THAT(out, HasSubstr("declare function use(d: Derived | null, h: Holder | null): void;"));
    EXPECT_THAT(out, Not(HasSubstr("type Vec")));
}

// A struct passed by value is still left out: each ABI passes one its own way
TEST(NamespacedStruct, ByValueStaysSkipped)
{
    auto out = cpp(std::string(gfx) + "extern \"C\" void fill(gfx::Color color);\n");
    EXPECT_THAT(out, HasSubstr("// skipped: fill \xE2\x80\x94 parameter 'color': struct passed by value"));
}
