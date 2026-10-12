// How tsbindgen spells a struct's layout as a tuple: arrays, bitfields, unions, bases, and the
// specializations of templates a field holds (std::string, a static_vector, a RefCountPtr), with
// the shapes C++ libraries give them. Hermetic: the "library" types are written here, as MSVC's
// and libc++'s are shaped.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::HasSubstr;
using testing::Not;

namespace
{

std::string cpp(const std::string &header, const std::string &target = windowsTarget)
{
    return generate(header, {}, target, {}, "c++").text;
}

} // namespace

// `T x[N]` is a tuple of its N elements
TEST(Layout, ArrayIsATupleOfItsElements)
{
    auto out = text("struct Item { int id; };\nstruct Grid { int cells[3]; struct Item items[2]; float m[2][2]; };\n");
    EXPECT_THAT(out, HasSubstr("type Grid = [cells: [_0: s32, _1: s32, _2: s32], items: [_0: Item, _1: Item], "
                               "m: [_0: [_0: f32, _1: f32], _1: [_0: f32, _1: f32]]];"));
}

// A run of bitfields is raw words over its bytes, named after the first
TEST(Layout, BitfieldRunIsRawWords)
{
    auto out = text("struct Flags { unsigned a : 1; unsigned b : 7; int x; unsigned short c : 3; };\n");
    EXPECT_THAT(out, HasSubstr("type Flags = [a: [_0: u32], x: s32, c: [_0: u32]];"));
}

// A union is its first member when that has the union's size and alignment, raw words otherwise;
// an anonymous one is named after its first member, an anonymous struct's fields are the struct's
TEST(Layout, Unions)
{
    auto out = text("#include <stdint.h>\n"
                    "struct Str { union { char buf[16]; char *ptr; } bx; uint64_t size; uint64_t cap; };\n"
                    "struct Obj { union { void *pointer; uint64_t integer; }; };\n"
                    "struct Pt { struct { int x; int y; }; int z; };\n");
    EXPECT_THAT(out, HasSubstr("type Str = [bx: [_0: u64, _1: u64], size: u64, cap: u64];"));
    EXPECT_THAT(out, HasSubstr("type Obj = [pointer: Opaque | null];"));
    EXPECT_THAT(out, HasSubstr("type Pt = [x: s32, y: s32, z: s32];"));
}

// A specialization a field holds is declared under a name of its arguments; its bases' fields come
// first, an empty base has none. A pointer to one alone stays Opaque.
TEST(Layout, SpecializationHeldByValue)
{
    const char *header = "#include <stddef.h>\n"
                         "namespace lib {\n"
                         "template <typename T, size_t N> struct array { T elems[N]; };\n"
                         "template <typename T, size_t N> struct static_vector : private array<T, N> {\n"
                         "    size_t current_size; size_t size() const { return current_size; } };\n"
                         "template <typename T> struct Ref { T *ptr_; Ref(); ~Ref(); };\n"
                         "template <typename T> struct Other { T value; };\n"
                         "}\n"
                         "namespace app {\n"
                         "class Item { public: virtual ~Item(); virtual int id() = 0; };\n"
                         "struct State { lib::Ref<Item> owner; lib::static_vector<Item *, 2> items; };\n"
                         "}\n"
                         "extern \"C\" void use(const app::State *state, lib::Other<int> *other);\n";
    for (auto *target : {windowsTarget, linuxTarget})
    {
        auto out = cpp(header, target);
        EXPECT_THAT(out, HasSubstr("type State = [owner: Ref_Item, items: static_vector_Item_2];")) << target;
        EXPECT_THAT(out, HasSubstr("type Ref_Item = [ptr_: Item | null];")) << target;
        EXPECT_THAT(out, HasSubstr("type static_vector_Item_2 = [elems: [_0: Item | null, _1: Item | null], current_size: index];"))
            << target;
        EXPECT_THAT(out, HasSubstr("use(state: Reference<State> | null, other: Opaque | null /* lib::Other<int> */)")) << target;
        EXPECT_THAT(out, Not(HasSubstr("type Other"))) << target;
    }
}

// An std::string shaped as MSVC's: an empty allocator base (no storage), its value in a member
// whose union holds a char buffer or a pointer
TEST(Layout, MsvcShapedString)
{
    auto out = cpp("#include <stddef.h>\n"
                   "namespace stdx {\n"
                   "template <typename T> struct allocator {};\n"
                   "template <typename A, typename V> struct compressed_pair : private A { V value; };\n"
                   "template <typename C> struct string_val { union bxty { C buf[16]; C *ptr; } bx; size_t size; size_t res; };\n"
                   "template <typename C> class basic_string { compressed_pair<allocator<C>, string_val<C>> pair; };\n"
                   "}\n"
                   "struct Desc { unsigned width; stdx::basic_string<char> name; bool flag; };\n");
    EXPECT_THAT(out, HasSubstr("type Desc = [width: u32, name: basic_string_char, flag: boolean];"));
    EXPECT_THAT(out, HasSubstr("type basic_string_char = [pair: compressed_pair_allocator_char_string_val_char];"));
    EXPECT_THAT(out, HasSubstr("type compressed_pair_allocator_char_string_val_char = [value: string_val_char];"));
    EXPECT_THAT(out, HasSubstr("type string_val_char = [bx: [_0: u64, _1: u64], size: index, res: index];"));
}

// An std::string shaped as libc++'s: its representation a union of structs with bitfields (raw
// words), its allocator a [[no_unique_address]] member with no storage
TEST(Layout, LibcxxShapedString)
{
    auto out = cpp("#include <stddef.h>\n"
                   "namespace stdx {\n"
                   "template <typename T> struct allocator {};\n"
                   "template <typename C> class basic_string {\n"
                   "    struct longs { size_t is_long : 1; size_t cap : 63; size_t size; C *data; };\n"
                   "    struct shorts { unsigned char is_long : 1; unsigned char size : 7; C data[23]; };\n"
                   "    union rep { shorts s; longs l; };\n"
                   "    rep rep_;\n"
                   "    [[no_unique_address]] allocator<C> alloc_;\n"
                   "};\n"
                   "}\n"
                   "struct Desc { unsigned width; stdx::basic_string<char> name; };\n",
                   linuxTarget);
    EXPECT_THAT(out, HasSubstr("type basic_string_char = [rep_: [_0: u64, _1: u64, _2: u64]];"));
    EXPECT_THAT(out, HasSubstr("type Desc = [width: u32, name: basic_string_char];"));
}

// MSVC puts a derived struct's fields after its base's tail padding (padded here), the Itanium ABI
// in it when the base isn't a POD
TEST(Layout, BaseTailPadding)
{
    const char *header = "#include <stdint.h>\n"
                         "struct Base { Base(); int64_t x; char c; };\n"
                         "struct Derived : Base { int32_t y; };\n"
                         "extern \"C\" void use(Derived *d);\n";
    EXPECT_THAT(cpp(header, windowsTarget),
                HasSubstr("type Derived = [x: s64, c: s8, _pad: [_0: u8, _1: u8, _2: u8, _3: u8, _4: u8, _5: u8, _6: u8], y: s32];"));
    EXPECT_THAT(cpp(header, linuxTarget), HasSubstr("type Derived = [x: s64, c: s8, y: s32];"));
}

// A base's field and the struct's own of one name: the later gets `_2`
TEST(Layout, InheritedNamesStayDistinct)
{
    auto out = cpp("struct A { int x; };\nstruct B : A { int x; };\nextern \"C\" void use(B *b);\n");
    EXPECT_THAT(out, HasSubstr("type B = [x: s32, x_2: s32];"));
}
