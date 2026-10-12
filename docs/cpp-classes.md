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
| a method without a body and without `@slot` or `@linkname`, overriding nothing | it's neither a slot nor a symbol: a forgotten `@slot` |
| an `abstract` method without `@slot` | it is virtual, so it needs its slot |
| `@slot` on a method of an ordinary class | an ordinary class's vtable is tslang's own |
| a constructor or a field initializer, in a class none of whose virtual methods has a body | the objects and their code are C++'s (see "Implementing C++ methods") |
| static members, `get`/`set` accessors, `accessor` fields | they would take vtable slots |
| `implements` | an implemented interface takes a slot |
| `extends` an ordinary class, or an ordinary class extending a `@cpp` one | the ordinary class's slots (`.instanceOf`, `.rtti`, ...) would shift the numbering |
| `instanceof`, a cast to or from `any`, a cast to an interface | each asks the object through slot 0, which is a C++ method |
| `-mm=rc` / `-mm=own` | retains and releases write a count in the word before an object, inside the C++ object |
| a 32-bit x86 MSVC target | C++ member functions are `__thiscall` there (`this` in ECX) |
| a virtual method returning a tuple, on a target other than MSVC and Itanium x86-64 / AArch64 | each ABI returns records its own way (see below) |
| on Itanium, a virtual method returning a trivial tuple other than one or two 8-byte integer or pointer fields | SysV / AAPCS64 pack it into registers by its fields, which a tuple isn't (see below) |
| a non-virtual C++ method returning a tuple | not supported yet: its symbol would have to be declared with the hidden result pointer |

A generic or static class can't be `@cpp` either.

## Writing a binding

- **Slot numbers.** Take them from the compiler, not by counting: MSVC prints a class's vtable
  with `cl /c /Zs /d1reportSingleClassLayout<Name> file.cpp`; clang (Itanium) with
  `clang++ -S -emit-llvm -o /dev/null -Xclang -fdump-vtable-layouts file.cpp`, for a class whose
  vtable the file emits (it defines the first virtual method that isn't inline), where slot N is
  entry N + 2 (entries 0 and 1 are the offset to top and the RTTI). The numbers differ between
  the two ABIs (the destructor, below), so a binding for both is two files, picked by the target
  (`<name>.<os>.ts`).
- **The destructor** is one slot under MSVC (the scalar deleting destructor) and two under the
  Itanium ABI. Don't call it: release objects the way the C++ library does (`Release()`).
- **Overloads.** tslang methods are identified by name, so every C++ overload needs its own
  TypeScript name, with its own `@slot`. MSVC groups overloads in the vtable, and not always in
  declaration order: take the numbers from the layout report.
- **Non-virtual methods** are bound by their mangled symbol: `@linkname("?scaled@Methods@@QEAAHH@Z")`
  under MSVC, `@linkname("_ZN7Methods6scaledEi")` under the Itanium ABI. The symbol has to be
  exported by the C++ library (`__declspec(dllexport)` on the method or the class).
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
  hidden pointer as an argument; call it directly.
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
  for example in a global, for as long as C++ may call it. A `string` argument is C++'s buffer:
  copy it to keep it past the call.
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
