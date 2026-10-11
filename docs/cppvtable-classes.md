# `@cppvtable` classes — calling C++ virtual methods directly

A class decorated with `@cppvtable` is laid over objects made by C++. tslang calls the methods
through the C++ object's own vtable, so a C++ interface such as NVRHI's `nvrhi::IDevice` can be
used from TypeScript without a flat `extern "C"` wrapper per method:

```ts
@cppvtable
class NvResource {
    _destructor(): void;   // slot 0: the C++ virtual destructor, declared only to keep the count
    AddRef(): u32;         // slot 1
    Release(): u32;        // slot 2
    GetRefCount(): u32;    // slot 3
}

@cppvtable
class NvDevice extends NvResource {
    // ... slots 4..38 ...
    getGraphicsAPI(): u8;  // slot 39
}

const device = <NvDevice>Donut_GetNvrhiDevice(app);   // an Opaque from C/C++
print(device.getGraphicsAPI());
```

## How it maps

A tslang class already matches a C++ single-inheritance object on 64-bit targets in the
following ways:

- the vtable pointer is the object's first word;
- a method call loads slot N of that table and calls it as `fn(this, args...)`, with no hidden
  parameters, in the target's C calling convention (on x64 Windows that is the MSVC member
  function convention);
- a base class's slots come first, and a base object is the prefix of a derived one.

What `@cppvtable` changes is what the class adds to the vtable. An ordinary class has
`.instanceOf` in slot 0, `.new`, a slot for each implemented interface, and slots for its static
fields (`.rtti`, `.size`). A `@cppvtable` class has none of these. Its methods are the vtable's
slots from 0, in declaration order, with a base class's slots first.

The class gets no vtable, constructor, RTTI or allocation of its own. It is a declaration: its
methods have no bodies, and an `abstract` class with `abstract` methods reads the same.

Fields declared on the class are the C++ object's fields after the vtable pointer, laid out like
a tuple's.

## Rules

The compiler rejects anything that would add a slot, run code of its own, or ask the object what
it is. A slot number off by one calls a different C++ method with no error at all, which is why
these are errors and not warnings.

| Not allowed | Why |
| --- | --- |
| a method with a body, a constructor, a field initializer | the objects and their code are C++'s |
| static members, `get`/`set` accessors, `accessor` fields | they would take vtable slots, and accessor slots don't follow declaration order |
| `implements` | an implemented interface takes a slot |
| `extends` an ordinary class, or an ordinary class extending a `@cppvtable` one | the ordinary class's slots (`.instanceOf`, `.rtti`, ...) would shift the numbering |
| `instanceof`, a cast to or from `any`, a cast to an interface | each asks the object through slot 0, which is a C++ method |
| `-mm=rc` / `-mm=own` | retains and releases write a count in the word before an object, inside the C++ object |
| a 32-bit x86 MSVC target | C++ member functions are `__thiscall` there (`this` in ECX) |
| a method returning a tuple, on a target other than MSVC and Itanium x86-64 / AArch64 | each ABI returns records its own way (see below) |
| on Itanium, a method returning a trivial tuple other than one or two 8-byte integer or pointer fields | SysV / AAPCS64 pack it into registers by its fields, which a tuple isn't (see below) |

A generic or static class can't be `@cppvtable` either.

## Writing a binding

- **Slot numbers.** Take them from the compiler, not by counting: MSVC prints a class's vtable
  with `cl /c /Zs /d1reportSingleClassLayout<Name> file.cpp`; clang (Itanium) with
  `clang++ -S -emit-llvm -o /dev/null -Xclang -fdump-vtable-layouts file.cpp`, for a class whose
  vtable the file emits (it defines the first virtual method that isn't inline), where slot N is
  entry N + 2 (entries 0 and 1 are the offset to top and the RTTI). Declare unused slots with
  placeholder names (`_slot17(): void;`) so the methods you call keep their numbers.
- **The destructor** is one slot under MSVC (the scalar deleting destructor) and two under the
  Itanium ABI. Don't call it: release objects the way the C++ library does (`Release()`).
- **Overloads.** tslang methods are identified by name, so every C++ overload needs its own
  TypeScript name. MSVC groups overloads in the vtable, and not always in declaration order:
  check the slot numbers in the layout report.
- **Struct returned by value.** A C++ method that returns a class or struct by value (for
  example `nvrhi::RefCountPtr<T>` or `nvrhi::Object`) returns a record. Declare its return type
  as a tuple with the record's layout:

  ```ts
  type RefCountPtr = [pointer: Opaque | null];          // nvrhi::RefCountPtr<T>
  @cppnontrivial                                        // it has a destructor
  createBuffer(desc: Ref<BufferDesc>): RefCountPtr;     // const handle = device.createBuffer(Ref(desc))
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
    AArch64). A tuple can't say the C++ type has them, so mark the method `@cppnontrivial`. (A
    trivial record of more than 16 bytes is returned the same way, so `@cppnontrivial` is right
    for it too.) MSVC ignores the mark;
  - a trivial record of one or two 8-byte integer or pointer fields - `nvrhi::Object` - comes
    back in RAX:RDX / X0:X1, as tslang returns such a tuple: no mark;
  - any other trivial record of up to 16 bytes is packed into registers by the SysV / AAPCS64
    classification, which isn't done: such a method is an error, as is any method returning a
    tuple on another Itanium target (32-bit x86, say).

  The record isn't destroyed: a `RefCountPtr` the method returns holds a reference, which the
  caller gives back with `Release()`. A method stored in a variable and called later takes the
  hidden pointer as an argument; call it directly. tslang pre-alpha94 and earlier need the hidden
  pointer declared as a parameter: `createBuffer(ret: Ref<Opaque | null>, desc: Ref<BufferDesc>): Opaque`.
- **Narrow parameters.** A `bool` is a `boolean`, an `uint8_t` (or an `enum class : uint8_t`) a
  `u8`, an `int16_t` an `s16`: a call of a `@cppvtable` method extends an argument narrower than
  32 bits as a C++ caller does (`zeroext`, `signext` for a signed type), which clang's code relies on.
- **Structs by reference** (`const BufferDesc&`) are a `Ref<T>` to a named tuple with the C++
  layout. That layout depends on the C++ build: `std::string`, for example, is 32 bytes with the
  release CRT and 40 with the debug CRT.
- **Casting.** `<NvDevice>opaqueValue` is a plain pointer cast with no check: the object must
  really be one.

## Tests

- `tslang/test/tester/tests/00cppvtable_class.ts` checks the slots, fields, `extends` and
  `abstract` against a vtable made by hand. It is registered for AOT and JIT.
- `tslang/test/tester/cppvtable/*.ts` holds one compile error per rule above.
