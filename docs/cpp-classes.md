# `@cpp` classes — calling C++ methods directly

A class decorated with `@cpp` is laid over objects made by C++. tslang calls its virtual methods
through the C++ object's own vtable, and its non-virtual methods by their symbols, so a C++
interface such as NVRHI's `nvrhi::IDevice` can be used from TypeScript without a flat
`extern "C"` wrapper per method:

```ts
@cpp
class NvResource {
    @slot(0) _destructor(): void;
    @slot(1) AddRef(): u32;
    @slot(2) Release(): u32;
    @slot(3) GetRefCount(): u32;
}

@cpp
class NvDevice extends NvResource {
    @slot(39) getGraphicsAPI(): u8;    // slots 4..38 aren't declared: nothing calls them
}

const device = <NvDevice>Donut_GetNvrhiDevice(app);   // an Opaque from C/C++
print(device.getGraphicsAPI());
```

(`@cpp` was called `@cppvtable` before `@slot` existed, and numbered the methods by their order.
The old name is an error now.)

## How it maps

A tslang class already matches a C++ single-inheritance object on 64-bit targets in the
following ways:

- the vtable pointer is the object's first word;
- a method call loads slot N of that table and calls it as `fn(this, args...)`, with no hidden
  parameters, in the target's C calling convention (on x64 Windows that is the MSVC member
  function convention);
- a base class's slots come first, and a base object is the prefix of a derived one.

What `@cpp` changes is what is virtual, and where. In an ordinary class every method is virtual,
and the vtable has `.instanceOf` in slot 0, `.new`, a slot for each implemented interface, and
slots for its static fields (`.rtti`, `.size`). A `@cpp` class has none of these, and a method is
virtual only as in C++:

| Method | What it is |
| --- | --- |
| `@slot(N) name(...): T;` | a C++ virtual method: slot N of the vtable, counted from 0 with the base classes' slots |
| `name(...): T;` overriding a base class's `@slot` method | the same slot, as in C++ (`@slot` may be repeated, with the same number) |
| `@linkname("symbol") name(...): T;` | a C++ non-virtual method, called by its (mangled) symbol as `symbol(this, args...)` |
| `@cppname("T name(...)") name(...): T;` | the same, by its C++ declaration: the compiler mangles it for the target (see below) |
| `name(...): T { ... }` with no `@slot`, overriding nothing | a method of TypeScript's own, called directly; not in the vtable |

The slots may be given in any order, and a slot no method names is skipped: declare only the
methods you call. Two methods in one slot are an error.

The class gets no vtable, constructor, RTTI or allocation of its own. It is a declaration: its C++
methods have no bodies, and an `abstract` class with `abstract` methods (each with its `@slot`)
reads the same. (A class whose virtual methods have bodies is the other way round: see
"Implementing C++ methods".)

Fields declared on the class are the C++ object's fields after the vtable pointer, laid out like
a tuple's.

## Rules

The compiler rejects anything that would add a slot, run code of its own, or ask the object what
it is. A wrong slot calls a different C++ method with no error at all, which is why these are
errors and not warnings.

| Not allowed | Why |
| --- | --- |
| a method without a body and without `@slot`, `@linkname` or `@cppname`, overriding nothing | it's neither a slot nor a symbol: a forgotten `@slot` |
| an `abstract` method without `@slot` | it is virtual, so it needs its slot |
| `@slot` on a method of an ordinary class | an ordinary class's vtable is tslang's own |
| a constructor or a field initializer, in a class none of whose virtual methods has a body | the objects and their code are C++'s (see "Implementing C++ methods") |
| static members, `get`/`set` accessors, `accessor` fields | they would take vtable slots |
| `implements` | an implemented interface takes a slot |
| `extends` an ordinary class, or an ordinary class extending a `@cpp` one | the ordinary class's slots (`.instanceOf`, `.rtti`, ...) would shift the numbering |
| `instanceof`, a cast to or from `any`, a cast to an interface | each asks the object through slot 0, which is a C++ method |
| a 32-bit x86 MSVC target | C++ member functions are `__thiscall` there (`this` in ECX) |
| a virtual method returning a tuple, on a target other than MSVC and Itanium x86-64 / AArch64 | each ABI returns records its own way (see below) |
| on Itanium, a virtual method returning a trivial tuple with floating-point fields, unmarked | SysV / AAPCS64 return it in SSE registers / as an HFA, which isn't done (see below) |
| on Itanium, a call of a method returning a trivial tuple of more than 16 bytes, unmarked | it is returned in memory: mark the method `@cppnontrivial` (see below) |
| on Itanium, a call passing a tuple with floating-point fields by value | SysV / AAPCS64 pass it in SSE registers / as an HFA, which isn't done (see below) |
| a non-virtual C++ method returning or taking a tuple | not supported yet: its symbol would have to be declared as each ABI passes records |
| a method implementing a C++ method (with a body) taking a tuple | not supported yet: its body would have to read it as the ABI passes it |

A generic or static class can't be `@cpp` either.

## Memory models

A `@cpp` class works under every memory model. Its reference is a C++ pointer, and no model
counts it:

- under `-mm=rc` and `-mm=own`, an ordinary object has its count in the word before it. A C++
  object has no such word: the word before it belongs to the C++ heap, or to whatever is next to
  it. So copying a `@cpp` reference into a variable, a field, an array or an argument doesn't
  retain it, and dropping one doesn't release it. A field, array element or union member of a
  `@cpp` type is skipped when what holds it is destroyed;
- the object is C++'s to free, the way the C++ library frees it (`Release()`, `delete`), as under
  `gc` and `none`.

## Writing a binding

`tsbindgen` writes the binding from the C++ header, with the slots and the `@cppnontrivial` marks
clang computes for the target, and each non-virtual method's `@cppname` declaration (its
`@linkname` symbol when `@cppname` can't spell it or mangles it otherwise) (README, "C++ classes"):
`tsbindgen nvrhi.hpp --target x86_64-pc-windows-msvc --filter 'nvrhi::I*' -- -std=c++17`.
Writing one by hand:

- **Slot numbers.** Take them from the compiler, not by counting: MSVC prints a class's vtable
  with `cl /c /Zs /d1reportSingleClassLayout<Name> file.cpp`; clang (Itanium) with
  `clang++ -S -emit-llvm -o /dev/null -Xclang -fdump-vtable-layouts file.cpp`, for a class whose
  vtable the file emits (it defines the first virtual method that isn't inline), where slot N is
  entry N + 2 (entries 0 and 1 are the offset to top and the RTTI). The numbers differ between
  the two ABIs (the destructor, below), so a binding for both is two files, picked by the target
  (`<name>.<os>.ts`).
- **The destructor** is one slot under MSVC (the scalar deleting destructor) and two under the
  Itanium ABI. Don't call it: release objects the way the C++ library does (`Release()`).
- **Overloads.** Declare each C++ overload under the C++ name, with its own `@slot` (or
  `@linkname` for a non-virtual one). A call picks the overload from its arguments, at compile
  time as C++ does: an argument of the parameter's own type fits best, then an integer for an
  integer or a float for a float, or a class derived from the parameter's class, then any other
  conversion. A class is a C++ pointer, so `T` and `T | null` fit a `T | null` parameter alike. A
  call that fits none, or fits two
  as well, is an error that lists the overloads. MSVC groups overloads in the vtable, in reverse
  declaration order: take the numbers from the layout report (tsbindgen does).

  ```ts
  @slot(8) over(x: s32): s32;
  @slot(7) over(x: f64): s32;
  @slot(6) over(a: s32, b: s32): s32;   // methods.over(5), methods.over(2.5), methods.over(6, 7)
  ```

  A derived class overriding one overload names its slot: by name alone it would be ambiguous.
- **Non-virtual methods** are bound by their mangled symbol: `@linkname("?scaled@Methods@@QEAAHH@Z")`
  under MSVC, `@linkname("_ZN7Methods6scaledEi")` under the Itanium ABI. The symbol has to be
  exported by the C++ library (`__declspec(dllexport)` on the method or the class).

  `@cppname` takes the C++ declaration instead and mangles it for the target: MSVC's scheme on
  `windows-msvc`, the Itanium one (g++, clang) everywhere else, MinGW included. That way one class
  works on every OS:

  ```ts
  @cppname("int scaled(int)")      scaled(x: s32): s32;   // ?scaled@Methods@@QEAAHH@Z, _ZN7Methods6scaledEi
  @cppname("int scaled(double)")   scaled(x: f64): s32;
  @cppname("int get() const")      get(): s32;            // a const method
  @cppname("size_t length(const char* text)") length(text: string): u64;
  ```

  An unqualified name is a member of the class. A qualified one is taken as written
  (`ns::Methods::scaled`). On a `declare function`, `@cppname("int ns::sum(int, int)")` binds a
  free function. The types are C++'s builtin ones; the `<stdint.h>`/`<stddef.h>` typedefs as the
  target defines them (`int64_t` is `long` on 64-bit Linux and `long long` on Windows and macOS, and
  each mangles differently); and records by qualified name (`ns::Pt*`, `const Pair&`), all with
  `const`, `*`, `&`, `&&`. A bare record name is a `struct`: write `class Obj`, `union U`
  or `enum E` when it isn't, because MSVC encodes which one it is. Names aren't looked up, so write
  a record in another namespace with its full name. Templates, operators, constructors,
  destructors, function pointers and arrays aren't supported: use `@linkname` for those.
- **Struct returned by value.** A C++ virtual method that returns a class or struct by value (for
  example `nvrhi::RefCountPtr<T>` or `nvrhi::Object`) returns a record. Declare its return type
  as a tuple with the record's layout:

  ```ts
  type RefCountPtr = [pointer: Opaque | null];          // nvrhi::RefCountPtr<T>
  @cppnontrivial                                        // it has a destructor
  @slot(17) createBuffer(desc: Ref<BufferDesc>): RefCountPtr;   // const handle = device.createBuffer(Ref(desc))
  ```

  MSVC returns a record from an instance method through a hidden pointer, whatever its size. The
  pointer is passed after `this`, and the method returns it as well. A call to such a method
  passes a local for the result and reads it back, so the call's value is the record. This is
  clang's `MicrosoftCXXABI::classifyReturnType`: "MSVC always returns structs indirectly from C++
  instance methods". A method returning a class (`getLayout(): IBindingLayout`, a C++ pointer) or
  a number still returns it in a register.

  The Itanium ABI (g++, clang) decides by the record, and tslang does it on x86-64 and AArch64
  (Linux, Android):

  - a record with a destructor or a copy constructor - `RefCountPtr<T>` - comes back through a
    hidden pointer passed *before* `this`, as the call's `sret` argument (in RDI on x86-64, X8 on
    AArch64). A tuple can't say the C++ type has them, so mark the method `@cppnontrivial`. A
    trivial record of more than 16 bytes is returned the same way, so mark that one too (a call
    of it unmarked is an error). MSVC ignores the mark;
  - a trivial record of up to 16 bytes of integers and pointers - `nvrhi::Object`, a struct of
    three `bool`s, one of three `uint32_t`s - comes back in RAX:RDX / X0:X1, packed as the record
    is in memory: the call reads the two registers back as the record. No mark;
  - one with floating-point fields comes back in SSE registers (SysV) or as an HFA (AAPCS64),
    which isn't done: such a method is an error, as is any method returning a tuple on another
    Itanium target (32-bit x86, say).

  The record isn't destroyed: a `RefCountPtr` the method returns holds a reference, which the
  caller gives back with `Release()`. A method stored in a variable and called later takes the
  hidden pointer as an argument; call it directly.
- **Structs passed by value.** A parameter of a virtual method that is a trivially copyable
  struct by value (`nvrhi::TextureSubresourceSet subresources`, `nvrhi::Object texture`) is a
  tuple with its layout, passed as the C++ ABI passes the struct:

  ```ts
  type TextureSubresourceSet = [baseMipLevel: u32, numMipLevels: u32, baseArraySlice: u32, numArraySlices: u32];
  @slot(9) clearTextureFloat(t: ITexture | null, subresources: TextureSubresourceSet, clearColor: Ref<Color>): void;
  // commandList.clearTextureFloat(texture, [0, 1, 0, 1], Ref(color))
  ```

  The call copies the tuple into a slot of its own (at least 16 bytes, zeroed), then:
  - MSVC x64 passes a struct of 1, 2, 4 or 8 bytes in a register, as an integer of its size, and
    any other by a pointer to the copy (which the callee may change: the caller's tuple doesn't);
  - SysV x86-64 passes one of up to 16 bytes of integers and pointers in one or two 64-bit
    registers, when that many of the six remain (else all of it on the stack), and a bigger one on
    the stack (`byval`);
  - AAPCS64 passes one of up to 16 bytes in one or two consecutive 64-bit registers (`[N x i64]`,
    all of it in registers or all on the stack), and a bigger one by a pointer to the copy.

  A struct with a destructor or a copy constructor (which MSVC destroys in the callee, and the
  Itanium ABI passes by a pointer to a copy the caller destroys) can't be a tuple parameter: take
  it as a `Ref<T>` in a wrapper of your own, or leave the method out. Nor can a struct with
  floating-point fields under the Itanium ABI (SSE registers, an HFA's), nor any struct taken by a
  non-virtual method or by one with a body.
- **Narrow parameters.** A `bool` is a `boolean`, an `uint8_t` (or an `enum class : uint8_t`) a
  `u8`, an `int16_t` an `s16`: a call of a C++ method extends an argument narrower than 32 bits as
  a C++ caller does (`zeroext`, `signext` for a signed type), which clang's code relies on.
- **Structs by reference** (`const BufferDesc&`) are a `Ref<T>` to a named tuple with the C++
  layout. That layout depends on the C++ build: `std::string`, for example, is 32 bytes with the
  release CRT and 40 with the debug CRT.
- **Casting.** `<NvDevice>opaqueValue` is a plain pointer cast with no check: the object must
  really be one.
- **Methods of your own.** A method with a body and no `@slot` is TypeScript's: a helper over the
  C++ methods, compiled as an ordinary function taking `this`.

## Implementing C++ methods

A `@cpp` class whose virtual methods have bodies implements C++ methods: C++ calls them through
the object's vtable. An example is a callback interface such as `nvrhi::IMessageCallback`. Declare
the C++ interface as above, and extend it:

```ts
@cpp
class IMessageCallback {               // nvrhi::IMessageCallback (MSVC)
    @slot(0) _destructor(): void;
    @slot(1) message(severity: u8, text: string): void;
}

@cpp
class MessageLog extends IMessageCallback {
    errors = 0;

    message(severity: u8, text: string): void {   // overrides slot 1
        if (severity >= 2) {
            this.errors++;
        }
    }
}

const log = new MessageLog();          // pass <Opaque>log to C++
```

- **The vtable** is the C++ layout and nothing else: the slots, with no `.instanceOf`, interface
  or static slots. `new` points the object's first word at it. A method with a body overrides the
  base's slot of the same name. A slot left to a method of the C++ interface (a method with no
  body, such as the destructor), or one no method names, is null, as a pure virtual's would be:
  C++ calling it crashes. A new virtual method takes the slot its `@slot` gives it.
- **The object** is allocated by tslang: the vtable pointer, the C++ interface's fields, then the
  class's own fields, laid out like a tuple's. It may have a constructor and field initializers.
- **Calls from C++** come in as C++ calls them: `this` first, then the arguments. A narrow result
  (`boolean`, `u8`, ...) is extended to 32 bits, as C++ callers expect. Declare by-value struct
  parameters the way the ABI passes them (see "Writing a binding").
- **Lifetime.** C++ keeps a raw pointer that the collector can't see. Hold the object in tslang,
  for example in a global, for as long as C++ may call it. Under `-mm=rc` and `-mm=own` its
  references aren't counted (see "Memory models"), so tslang never frees it: an object made with
  `new` lives until the program ends, and so does what its fields hold. A `string` argument is C++'s buffer:
  copy it to keep it past the call. (The other way, a `string` a C++ method returns to
  TypeScript is copied by the call; declare the method to return `Opaque` for the pointer.)
- **Not yet supported:**
  - returning a struct (an error);
  - exporting the class (an error: an importer would get the methods but not the vtable);
  - type information: C++ must not `dynamic_cast` or `typeid` the object;
  - throwing from a method C++ calls.

## Tests

- `tslang/test/tester/tests/00cppvtable_class.ts` checks the slots, fields, `extends` and
  `abstract` against a vtable made by hand, and `tests/00cpp_slot.ts` slots out of order, skipped
  slots and a method of TypeScript's own. Both are registered for AOT and JIT.
- `tslang/test/tester/cppvtable/*.ts` holds one compile error per rule above.
- `test-jit-cppvtable-abi` checks against code from a real C++ compiler. `cppvtable/abi_host.cpp` is built as a shared library by the host's compiler (MSVC, or g++/clang). `abi_msvc.ts` or `abi_itanium.ts` then:
  - calls its object's virtual methods, including struct returns and narrow arguments, and a
    non-virtual one by its mangled symbol;
  - implements its `Callback` interface (`Counter`), which C++ calls back, with a method of
    TypeScript's own beside the overrides.
