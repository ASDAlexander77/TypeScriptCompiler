// A polymorphic C++ class is a @cpp class: its virtual methods @slot(N), with the slot clang's
// vtable layout gives for the target's ABI, its non-virtual ones @cppname with their C++
// declarations (@linkname with the target's mangled symbol when @cppname can't spell one). The slot
// numbers below are the compilers' own: MSVC's from
// `cl /d1reportSingleClassLayoutCircle`, the Itanium ABI's from clang's -fdump-vtable-layouts.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::AllOf;
using testing::Contains;
using testing::HasSubstr;
using testing::Not;

namespace
{

const char *const shapes = "namespace geo {\n"
                           "class Shape {\n"
                           "public:\n"
                           "    virtual ~Shape();\n"
                           "    virtual int area() const = 0;\n"
                           "    virtual void scale(int by) = 0;\n"
                           "    virtual void scale(float by) = 0;\n"
                           "    virtual void scale(double by) = 0;\n"
                           "protected:\n"
                           "    virtual void hidden() = 0;\n"
                           "public:\n"
                           "    virtual int last() = 0;\n"
                           "};\n"
                           "class Circle : public Shape {\n"
                           "public:\n"
                           "    int area() const override;\n"
                           "    virtual bool contains(const Shape &other) const;\n"
                           "    void scale(float by) override;\n"
                           "};\n"
                           "}\n";

std::string cpp(const std::string &header, const std::string &target = windowsTarget)
{
    return generate(header, {}, target, {}, "c++").text;
}

} // namespace

// MSVC: one destructor slot, and overloads grouped in reverse declaration order, under one name
TEST(CppClass, MsvcSlots)
{
    auto out = cpp(shapes);
    EXPECT_THAT(out, HasSubstr("@cpp\nclass Shape { // geo::Shape\n"
                               "    @slot(1) area(): s32;\n"
                               "    @slot(4) scale(by: s32): void;\n"
                               "    @slot(3) scale(by: f32): void;\n"
                               "    @slot(2) scale(by: f64): void;\n"
                               "    @slot(6) last(): s32;\n"
                               "}\n"));
    // the destructor (0) and the protected hidden() (5) are slots left out; overrides aren't
    // declared again: a @cpp class's override keeps its base's slot by name
    EXPECT_THAT(out, HasSubstr("@cpp\nclass Circle extends Shape { // geo::Circle\n"
                               "    @slot(7) contains(other: Shape): boolean;\n"
                               "}\n"));
}

// Itanium: two destructor slots, overloads in declaration order
TEST(CppClass, ItaniumSlots)
{
    auto out = cpp(shapes, linuxTarget);
    EXPECT_THAT(out, HasSubstr("    @slot(2) area(): s32;\n"
                               "    @slot(3) scale(by: s32): void;\n"
                               "    @slot(4) scale(by: f32): void;\n"
                               "    @slot(5) scale(by: f64): void;\n"
                               "    @slot(7) last(): s32;\n"));
    EXPECT_THAT(out, HasSubstr("    @slot(8) contains(other: Shape): boolean;\n"));
}

// one binding for every target: @cppname mangles the declaration for the one compiled for
TEST(CppClass, NonVirtualMethodsAreBoundByTheirDeclarations)
{
    auto header = "namespace geo { class Shape { public: virtual ~Shape(); int id(int x); int inlined() { return 1; }\n"
                  "static int count(); int operator+(int x); }; }\n";
    auto windows = generate(header, {}, windowsTarget, {}, "c++");
    EXPECT_THAT(windows.text, HasSubstr("    @cppname(\"int geo::Shape::id(int)\") id(x: s32): s32;\n"));
    EXPECT_THAT(windows.text, HasSubstr("    // skipped: inlined \xE2\x80\x94 inline method: no symbol to link against\n"));
    EXPECT_THAT(windows.warnings, Contains("skipped geo::Shape::inlined: inline method: no symbol to link against"));
    // static members and operators aren't methods a @cpp class can have
    EXPECT_THAT(windows.text, AllOf(Not(HasSubstr("count")), Not(HasSubstr("operator"))));

    EXPECT_THAT(cpp(header, linuxTarget), HasSubstr("    @cppname(\"int geo::Shape::id(int)\") id(x: s32): s32;\n"));
}

// the types @cppname spells: builtins, <stdint.h> typedefs as written (another typedef as what it
// names), records by their qualified names and tags, enums, cv-qualified pointers and references,
// and a const method; the same on every target
TEST(CppClass, DeclarationsSpellEveryTypeCppnameMangles)
{
    auto header = "#include <stdint.h>\n#include <stddef.h>\n"
                  "namespace ns { struct Pt { int x; }; class Obj; union U { int i; float f; }; enum E { A };\n"
                  "enum class Scoped : uint8_t { B }; typedef int myint; typedef struct Pt PtAlias;\n"
                  "class Widget { public: virtual ~Widget();\n"
                  "    int64_t wide(uint64_t a, int32_t b, size_t c, ptrdiff_t d, uint8_t e);\n"
                  "    const char *name(char *const *argv, const volatile int *v) const;\n"
                  "    void kinds(Pt *p, Obj *o, U &u, E e, Scoped s, const PtAlias *alias, myint m);\n"
                  "    void words(unsigned long long a, long b, unsigned short c, signed char d, wchar_t e, bool f,\n"
                  "               double g, float h, char16_t i);\n"
                  "    Widget *self(Widget &&other); }; }\n";
    for (auto target : {windowsTarget, linuxTarget, "aarch64-apple-darwin", "aarch64-linux-gnu"})
    {
        auto out = generate(header, {}, target, {}, "c++");
        EXPECT_THAT(out.text, HasSubstr("    @cppname(\"int64_t ns::Widget::wide(uint64_t, int32_t, size_t, ptrdiff_t, "
                                        "uint8_t)\") wide("))
            << target;
        EXPECT_THAT(out.text, HasSubstr("    @cppname(\"const char* ns::Widget::name(char* const*, const volatile "
                                        "int*) const\") name("))
            << target;
        EXPECT_THAT(out.text, HasSubstr("    @cppname(\"void ns::Widget::kinds(ns::Pt*, class ns::Obj*, union ns::U&, "
                                        "enum ns::E, enum ns::Scoped, const ns::Pt*, int)\") kinds("))
            << target;
        EXPECT_THAT(out.text, HasSubstr("    @cppname(\"void ns::Widget::words(unsigned long long, long, unsigned "
                                        "short, signed char, wchar_t, bool, double, float, char16_t)\") words("))
            << target;
        EXPECT_THAT(out.text, HasSubstr("    @cppname(\"class ns::Widget* ns::Widget::self(class ns::Widget&&)\") self("))
            << target;
        EXPECT_THAT(out.text, Not(HasSubstr("@linkname"))) << target;
    }
}

// what @cppname can't spell stays bound by this target's symbol: a function pointer, a
// ref-qualified method, a template's specialization
TEST(CppClass, DeclarationsCppnameCantSpellKeepTheSymbol)
{
    auto header = "template <typename T> struct Box { T v; };\n"
                  "class Widget { public: virtual ~Widget(); int call(int (*fn)(int)); int ref() &;\n"
                  "    int boxed(Box<int> *box); int plain(int x); };\n";
    auto windows = cpp(header);
    EXPECT_THAT(windows, HasSubstr("    @linkname(\"?call@Widget@@QEAAHP6AHH@Z@Z\") call("));
    EXPECT_THAT(windows, HasSubstr("    @linkname(\"?ref@Widget@@QEGAAHXZ\") ref(): s32;\n"));
    EXPECT_THAT(windows, HasSubstr("    @linkname(\"?boxed@Widget@@QEAAHPEAU?$Box@H@@@Z\") boxed("));
    EXPECT_THAT(windows, HasSubstr("    @cppname(\"int Widget::plain(int)\") plain(x: s32): s32;\n"));

    auto linux = cpp(header, linuxTarget);
    EXPECT_THAT(linux, HasSubstr("    @linkname(\"_ZN6Widget4callEPFiiE\") call("));
    EXPECT_THAT(linux, HasSubstr("    @linkname(\"_ZNR6Widget3refEv\") ref(): s32;\n"));
    EXPECT_THAT(linux, HasSubstr("    @linkname(\"_ZN6Widget5boxedEP3BoxIiE\") boxed("));
}

// a declaration @cppname spells but mangles to another symbol than clang's stays bound by clang's:
// a header's own `int64_t` of `long long` where the target's is `long` (64-bit Linux)
TEST(CppClass, DeclarationsThatMangleOtherwiseKeepTheSymbol)
{
    auto header = "typedef long long int64_t;\n"
                  "class Widget { public: virtual ~Widget(); int64_t wide(int64_t x); };\n";
    EXPECT_THAT(cpp(header, linuxTarget), HasSubstr("    @linkname(\"_ZN6Widget4wideEx\") wide(x: int64_t): int64_t;\n"));
    // where `int64_t` is `long long`, the declaration is the target's own
    EXPECT_THAT(cpp(header), HasSubstr("    @cppname(\"int64_t Widget::wide(int64_t)\") wide(x: int64_t): int64_t;\n"));
}

// A record returned by value is a tuple with its layout; one C++ can't pass in registers (a
// destructor, a copy constructor) is @cppnontrivial, which decides where the Itanium ABI puts it
TEST(CppClass, RecordsReturnedByValue)
{
    auto header = "struct Ptr { void *p; Ptr(const Ptr &o); ~Ptr(); };\n"
                  "struct Obj { void *p; };\n"
                  "struct Two { unsigned long long a; void *b; };\n"
                  "struct Three { int a; int b; int c; };\n"
                  "class Maker { public: virtual ~Maker(); virtual Ptr make(int x) = 0; virtual Obj obj() = 0;\n"
                  "virtual Two two() = 0; virtual Three three() = 0; Obj plain(); };\n";
    auto windows = cpp(header);
    EXPECT_THAT(windows, HasSubstr("    @cppnontrivial\n    @slot(1) make(x: s32): [p: Opaque | null];\n"));
    EXPECT_THAT(windows, HasSubstr("    @slot(2) obj(): [p: Opaque | null];\n"));
    EXPECT_THAT(windows, HasSubstr("    @slot(3) two(): [a: u64, b: Opaque | null];\n"));
    // MSVC returns any record through a hidden pointer
    EXPECT_THAT(windows, HasSubstr("    @slot(4) three(): [a: s32, b: s32, c: s32];\n"));
    EXPECT_THAT(windows, HasSubstr("// skipped: plain \xE2\x80\x94 a non-virtual method returning a record by value "
                                   "isn't supported yet"));

    auto linux = cpp(header, linuxTarget);
    EXPECT_THAT(linux, HasSubstr("    @cppnontrivial\n    @slot(2) make(x: s32): [p: Opaque | null];\n"));
    EXPECT_THAT(linux, HasSubstr("    @slot(4) two(): [a: u64, b: Opaque | null];\n"));
    // the Itanium ABI returns a trivial record of up to 16 bytes in registers, which tslang reads
    // back as the record
    EXPECT_THAT(linux, HasSubstr("    @slot(5) three(): [a: s32, b: s32, c: s32];\n"));
}

// The Itanium ABI returns a trivial record of more than 16 bytes through a pointer before `this`,
// as one with a destructor (@cppnontrivial); one of floating-point fields comes back in SSE
// registers, which tslang doesn't do
TEST(CppClass, RecordsReturnedInMemoryOrFloatRegisters)
{
    auto header = "struct Big { unsigned long long a, b, c; };\n"
                  "struct Floats { float x; float y; };\n"
                  "class Maker { public: virtual ~Maker(); virtual Big big() = 0; virtual Floats floats() = 0; };\n";
    auto linux = cpp(header, linuxTarget);
    EXPECT_THAT(linux, HasSubstr("    @cppnontrivial\n    @slot(2) big(): [a: u64, b: u64, c: u64];\n"));
    EXPECT_THAT(linux, HasSubstr("// skipped: floats \xE2\x80\x94 result: trivial record with floating-point fields "
                                 "returned in registers"));
    // MSVC: both through the pointer after `this`
    auto windows = cpp(header);
    EXPECT_THAT(windows, HasSubstr("    @slot(1) big(): [a: u64, b: u64, c: u64];\n"));
    EXPECT_THAT(windows, HasSubstr("    @slot(2) floats(): [x: f32, y: f32];\n"));
}

// A record a virtual method takes by value is its own type, which tslang passes as the ABI does;
// one C++ can't pass in registers, a union, and (Itanium) one of floating-point fields are skipped,
// and so is a non-virtual method taking one
TEST(CppClass, RecordsPassedByValue)
{
    auto header = "struct Range { unsigned int first, count; unsigned int base, size; };\n"
                  "struct Owned { void *p; ~Owned(); };\n"
                  "union Either { int i; float f; };\n"
                  "struct Floats { float x; float y; };\n"
                  "class Taker { public: virtual ~Taker(); virtual void range(Range r, int tail) = 0;\n"
                  "virtual void owned(Owned o) = 0; virtual void either(Either e) = 0; virtual void floats(Floats f) = 0;\n"
                  "void plain(Range r); };\n";
    for (auto target : {std::string(windowsTarget), std::string(linuxTarget)})
    {
        auto out = cpp(header, target);
        EXPECT_THAT(out, HasSubstr("type Range = [first: u32, count: u32, base: u32, size: u32];"));
        EXPECT_THAT(out, HasSubstr(") range(r: Range, tail: s32): void;\n"));
        EXPECT_THAT(out, HasSubstr("// skipped: owned \xE2\x80\x94 parameter 'o': struct with a destructor or a copy "
                                   "constructor passed by value"));
        EXPECT_THAT(out, HasSubstr("// skipped: either \xE2\x80\x94 parameter 'e': union passed by value"));
        EXPECT_THAT(out, HasSubstr("// skipped: plain \xE2\x80\x94 parameter 'r': a non-virtual method taking a record "
                                   "by value isn't supported yet"));
    }
    EXPECT_THAT(cpp(header), HasSubstr(") floats(f: Floats): void;\n"));
    EXPECT_THAT(cpp(header, linuxTarget), HasSubstr("// skipped: floats \xE2\x80\x94 parameter 'f': struct with "
                                                    "floating-point fields passed by value"));
}

// A pointer or a reference to a @cpp class's object is the class; a C++ reference to anything
// else is a pointer; an enum or a typedef of a namespace is its integer type
TEST(CppClass, ParameterTypes)
{
    auto out = cpp("struct Point { int x; int y; };\n"
                   "namespace geo { enum class Kind : unsigned char { Circle, Square }; typedef unsigned int Layer;\n"
                   "class Canvas { public: virtual ~Canvas(); virtual void draw(Canvas &other, const Canvas *again,\n"
                   "const Point &at, Kind kind, bool fill, Layer layer) = 0; }; }\n"
                   "extern \"C\" geo::Canvas *make_canvas(int width);\n");
    EXPECT_THAT(out, HasSubstr("type Point = [x: s32, y: s32];"));
    EXPECT_THAT(out, HasSubstr("    @slot(1) draw(other: Canvas, again: Canvas | null, at: Reference<Point>, kind: u8, "
                               "fill: boolean, layer: u32): void;\n"));
    EXPECT_THAT(out, Not(HasSubstr("Layer")));
    EXPECT_THAT(out, HasSubstr("declare function make_canvas(width: s32): Canvas;"));
}

// Fields follow the vtable pointer and the base's fields, as a @cpp class lays them out
TEST(CppClass, Fields)
{
    auto out = cpp("class Base { public: virtual ~Base(); int a; };\n"
                   "class Derived : public Base { public: virtual void f(); double b; };\n");
    EXPECT_THAT(out, HasSubstr("class Base { // Base\n    a: s32;\n}\n"));
    EXPECT_THAT(out, HasSubstr("class Derived extends Base { // Derived\n    b: f64;\n    @slot(1) f(): void;\n}\n"));

    // a @cpp class lays a derived class's fields out after its base's storage, padded to its
    // alignment, as MSVC does; the Itanium ABI puts them in the base's tail padding, so there they
    // aren't declared
    auto padded = "class Base { public: virtual ~Base(); int a; };\n"
                  "class Derived : public Base { public: int b; };\n";
    EXPECT_THAT(cpp(padded), HasSubstr("class Derived extends Base { // Derived\n    b: s32;\n}\n"));
    EXPECT_THAT(cpp(padded, linuxTarget),
                HasSubstr("class Derived extends Base { // Derived\n"
                          "    // fields not declared: field 'b' isn't where a tuple would put it\n}\n"));
}

// What a @cpp class can't be stays Opaque, as before
TEST(CppClass, ClassesThatArentBound)
{
    auto out = cpp("class A { public: virtual ~A(); };\nclass B { public: virtual ~B(); };\n"
                   "class Both : public A, public B { public: virtual void f(); };\n"
                   "class Virtual : public virtual A { public: virtual void g(); };\n"
                   "template <typename T> class Box { public: virtual T get(); };\n"
                   "extern \"C\" void use(Both *both, Virtual *v, Box<int> *box);\n");
    EXPECT_THAT(out, HasSubstr("// skipped: Both \xE2\x80\x94 C++ class"));
    EXPECT_THAT(out, HasSubstr("// skipped: Virtual \xE2\x80\x94 C++ class"));
    EXPECT_THAT(out, HasSubstr("declare function use(both: Both | null, v: Virtual | null, box: Opaque | null /* Box<int> */): void;"));
    EXPECT_THAT(out, AllOf(HasSubstr("class A "), HasSubstr("class B ")));
}

// --filter picks a class by its name or its qualified name, and brings its base along
TEST(CppClass, FilterByQualifiedName)
{
    tsbindgen::PrintOptions options;
    options.filters = {"geo::Circle"};
    auto out = generate(shapes, options, windowsTarget, {}, "c++").text;
    EXPECT_THAT(out, AllOf(HasSubstr("class Shape "), HasSubstr("class Circle extends Shape ")));
}
