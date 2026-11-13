# C and C++ bindings with tsbindgen

`tsbindgen` reads a C header (or `.c` file) with clang and prints the matching tslang declarations.

```text
tsbindgen <input.h|.c|.cpp> [options] [-- extra clang args]

  -o out.ts          write to a file (default: stdout)
  -I dir, -D name=v  include directories and macros, as for clang
  --filter <glob>    emit the declarations whose C name matches (repeatable)
  --namespace N      wrap the output in `namespace N`
  --strip-prefix P   remove P from the TS names (needs --namespace)
  --target triple    target triple (default: the host)
```

Given `simple.c`:

```c
#include <stdint.h>
#define MAX_ITEMS 16
typedef struct { int32_t x; int32_t y; } Point;
int add(int a, int b) { return a + b; }
static int helper(int a) { return a; }
double dist(const Point *p);
```

`tsbindgen simple.c -o simple.ts` writes:

```typescript
const MAX_ITEMS = 16;
type Point = [x: s32, y: s32];
declare function add(a: s32, b: s32): s32;
// skipped: helper — static function: no symbol to link against
declare function dist(p: Reference<Point> | null): f64;
```

A pointer parameter or struct field is nullable (`| null`), as C lets it be null; a C++ reference (`const Point &`) and a result aren't.

A C function taking a callback is declared `@cdecl`: its function-typed parameters are then C function pointers, given a function or an arrow that captures nothing (a closure or a bound method is a compile error):

```typescript
@cdecl declare function qsort(base: Opaque, count: u64, size: u64, compare: (a: Opaque, b: Opaque) => s32): void;
```

A struct is a named tuple with its fields where C puts them; tslang lays a nested tuple out in place, as C does a struct member:

- **an array** `T x[N]` is a tuple of its elements, `x: [_0: T, _1: T, ...]` (tslang has no fixed-size array type; one of more than 1024 elements, or of unknown size, is skipped);
- **a union** is its first member when that alone has the union's size and alignment (`union { void *p; uint64_t n; }` is `p: Opaque | null`), raw words otherwise (`union { char buf[16]; char *ptr; }` is `[_0: u64, _1: u64]`); an anonymous one is named after its first member, an anonymous struct's fields are the struct's own;
- **a run of bitfields** is raw words over the bytes it uses, named after its first bitfield (`unsigned a : 1; unsigned b : 7;` is `a: [_0: u32]`): set and read the bits yourself;
- a struct with an `alignas` alignment beyond its fields' (`DECLSPEC_ALIGN(16)`), or packed, is skipped.

Include the result with `/// <reference path="simple.ts" />`, not `import`. Name it `.ts`, not `.d.ts`: a `.d.ts` file is read as declarations only, so its constants would have no value.

## When nothing is emitted

- **Declarations from `#include`d headers are emitted only with `--filter`.** Without it, only what the input file itself declares is printed; a file that only includes headers prints nothing but the warning `nothing to emit`. Pick what you need by name:

  ```bat
  tsbindgen inc.c --filter puts --filter printf
  ```

  ```typescript
  declare function puts(_Buffer: string | null): s32;
  @varargs declare function printf(_Format: string | null): s32;
  ```

- **`static` and `static inline` functions are skipped**: there is no symbol to link against.
- **Function-like macros and macros that are not literals are skipped.**

## "function without a prototype"

```c
extern int test_func();
```

gives `// skipped: test_func — function without a prototype`. Before C23, empty parentheses do not mean "no arguments": they leave the arguments unspecified, and clang parses C17 by default. Either write `(void)`:

```c
extern int test_func(void);
```

or, for a header you cannot change, parse it as C23 (everything after `--` goes to clang):

```bat
tsbindgen test.c -o test.ts -- -std=c23
```

Both give:

```typescript
declare function test_func(): s32;
```

## C++ headers

`.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh` and `.hxx` files are parsed as C++; a `.h` file is parsed as C, so a C++ header named `.h` needs `-x c++` after `--`:

```bat
tsbindgen mylib.h -o mylib.ts -- -x c++
```

Given `mylib.h`:

```cpp
#include <cstdint>
namespace lib { class Widget { public: int size() const; }; int helper(int); }
struct Point { int32_t x; int32_t y; };
enum class Mode : int { A = 1, B = 2 };
#ifdef __cplusplus
extern "C" {
#endif
int add(int a, int b);
double dist(const Point *p);
#ifdef __cplusplus
}
#endif
int cxx_only(int a);
```

it writes:

```typescript
// skipped: Widget — empty struct
type Widget = Opaque;
type Point = [x: s32, y: s32];
enum Mode { A = 1, B = 2 }
declare function add(a: s32, b: s32): s32;
declare function dist(p: Reference<Point> | null): f64;
```

Structs, enums and `extern "C"` functions are emitted, and polymorphic classes (below). A struct or a class without virtual methods, in any namespace or class, is a tuple when its fields are where a tuple's are: constructors, methods and member initializers don't move them, so `struct Color { float r, g, b, a; Color(); }` is `type Color = [r: f32, g: f32, b: f32, a: f32];`. It is declared under its own name, and `--filter` matches that or the qualified name (`--filter 'nvrhi::*'`); of two of one name in different namespaces, the next is named with its namespaces (`rt_Desc` for `nvrhi::rt::Desc` under `--namespace nvrhi`). A derived struct has its bases' fields first, as C++ lays them out (an empty base has none; MSVC's tail padding after a base is a `_pad` field). A field holding a template's specialization by value is laid out as the library has it, and written once for the template as a generic tuple, its type arguments the parameters and its integer arguments part of the name: `std::vector<Item>` is `vector<Item>`, with `type vector<T> = [_Mypair: [_Myval2: [_Myfirst: Reference<T> | null, ...]]]` under MSVC (the specializations its pointers are in, `_Compressed_pair` and `_Vector_val`, inlined) and `[__begin_: Reference<T> | null, ...]` under libc++; `nvrhi::static_vector<IBindingSet *, 8>` is `static_vector8<IBindingSet | null>` (its private `std::array`'s elements, then `current_size`); `RefCountPtr<IShader>` is `RefCountPtr<IShader>`, with `type RefCountPtr<T> = [ptr_: T | null]`. A specialization whose layout doesn't show its type argument, or that is laid out unlike its template's others (an explicit specialization), is declared on its own, named by its arguments: `std::string` is `basic_string_char` (MSVC's `[_Mypair: ...]` down to its `_Bx` buffer words, size and capacity; libc++'s `[__rep_: [_0: u64, _1: u64, _2: u64]]`). Their member names are the C++ library's own, which differ between MSVC's and libc++; a pointer to a specialization alone stays `Opaque`. One that isn't a tuple is `type Name = Opaque;`, with the reason in a `// skipped:` line: a polymorphic class, a virtual base, a packed or over-aligned layout, or a field tsbindgen can't spell (`long double`, a huge array). Namespace functions (`lib::helper`) and functions with C++ linkage (`cxx_only`) are left out, without a warning. To bind those, expose them through `extern "C"` wrapper functions (with an opaque handle for each class) and run tsbindgen on the header that declares them.

### C++ classes

A polymorphic class (one with virtual methods), in any namespace, becomes a [`@cpp` class](cpp-classes.md): tslang calls its virtual methods through the object's vtable and its non-virtual methods by their C++ declarations (`@cppname`, which tslang mangles for the target it compiles for), with no wrapper. clang computes the slots for the target's ABI (`--target`), so generate the bindings once per ABI: MSVC and the Itanium ABI (Linux, Android) number the slots differently. A non-virtual method `@cppname` can't spell (a function pointer or a template in its signature) is bound by the target's mangled symbol, `@linkname`, instead.

```cpp
namespace geo {
struct Handle { void *p; Handle(const Handle &); ~Handle(); };
class Shape {
public:
    virtual ~Shape();
    virtual int area() const = 0;
    virtual void scale(int by) = 0;
    virtual void scale(double by) = 0;
    virtual Handle handle() = 0;
    int id(int x);
};
class Square : public Shape { public: int area() const override; virtual bool bigger(const Shape &other) const; int side; };
}
extern "C" geo::Square *make_square(int side);
```

`tsbindgen shapes.hpp --target x86_64-pc-windows-msvc` writes:

```typescript
type Handle = [p: Opaque | null];
@cpp
class Shape { // geo::Shape
    @slot(1) area(): s32;
    @slot(3) scale(by: s32): void;
    @slot(2) scale(by: f64): void;
    @cppnontrivial
    @slot(4) handle(): [p: Opaque | null];
    @cppname("int geo::Shape::id(int)") id(x: s32): s32;
}
@cpp
class Square extends Shape { // geo::Square
    side: s32;
    @slot(5) bigger(other: Shape): boolean;
}
declare function make_square(side: s32): Square;
```

- **Slots** are clang's: the destructor (slot 0 under MSVC, 0 and 1 under the Itanium ABI) and methods that aren't public are slots left out. A method overriding a base's is not declared again: the `@cpp` class's override keeps the base's slot.
- **Overloads** keep the C++ name, each with its own slot (or symbol), in whatever order the ABI puts the slots: a call picks the overload by its arguments, as C++ does (`shape.scale(2)`, `shape.scale(0.5)`). A method named as a field gets `_2`, `_3`, ...
- **A pointer or a reference** to an object of a bound class is the class (`| null` for a pointer); a reference to a struct is a `Reference` to its tuple.
- **A record returned by value** is a tuple with its layout, `@cppnontrivial` when it has a destructor or a copy constructor, or (Itanium) is more than 16 bytes; under the Itanium ABI one with floating-point fields is skipped.
- **A record passed by value** to a virtual method is its own type (a tuple), which tslang passes as the C++ ABI does ([cpp-classes.md](cpp-classes.md), "Structs passed by value"); one that isn't trivially copyable, a union, and (Itanium) one with floating-point fields are skipped, and so is a non-virtual method taking one.
- **Fields** are declared when they are where a `@cpp` class puts them (after the vtable pointer and the base's storage).
- **Left out, with a `// skipped:` line:** `inline` methods (no symbol to link to), struct parameters passed by value, and a non-virtual method returning a struct. Static methods and operators are left out silently.
- **Not bound at all:** templates, classes with several bases or a virtual one, and 32-bit targets. A pointer to such a class stays `Opaque`.

Compile the bindings with `-mm=gc` or `-mm=none`: a `@cpp` class's objects are C++'s.
