# tslang issues found while working on native_ts_graphics

Collected 2026-10-12 on `tslang v0.0-pre-alpha96`, updated the same day against
`tslang v0.0-pre-alpha97` (I:\tslang\tslang.exe, Windows x64, JIT run: `tslang file.ts`).
Every repro below is a complete file that was run. Source project:
https://github.com/ASDAlexander77/native_ts_graphics

## Status

| # | Issue | pre-alpha96 | pre-alpha97 |
|---|---|---|---|
| 1 | A derived class without a constructor does not forward its arguments to the base constructor | bug | **fixed** (TypeScriptCompiler#557) |
| 2 | The error for (1) pointed at line 1, column 1 of the base class | bug | **moot**: nothing synthesized fails any more; an explicit bad `super()` is reported at the call |
| 3 | `u64` `<<` shifted by the count modulo 32 | bug | **fixed** |
| 4 | `Ref<string>` field with a `null` initializer rejected | question | **answered**: declare it `Ref<string> \| null` |
| 5 | `@cppvtable` (now `@cpp`) overload naming (C++ overloads need distinct TypeScript names) | open | **fixed**: `@slot(N)` (#561), slots from tsbindgen (#562), and overloads under one name, picked by the arguments (this PR) |
| 6 | C `string` return values keep the raw `char*` (no copy) | open | **fixed**: copied at the call (this PR; `Opaque` is the raw pointer) |
| 7 | `Opaque` compared with `Opaque \| null` crashed the compiler | - | **fixed** by #536 (re-checked with the original drawScene shape) |
| 8 | A ternary of number literals in arithmetic truncated to the other operand's integer type | - | **fixed** by #560 (still wrong on pre-alpha97) |
| 9-12 | Bool const arrays, `Math.max`, `f32` array literal, `let x = 0` | - | fixed (#537, DefaultLib#31, works, #544 warning) |
| 13 | A generic function's `T` isn't inferred from an argument of a generic tuple alias (`Vec<T>`) | - | **fixed** by #572 (pre-alpha100: open) |
| 14 | A literal-union ternary (`x ? 1 : 0`) as an integer tuple field fails in codegen (`llvm.sext` of a union, no location) | - | **fixed** by #568 (pre-alpha100: open) |
| 15 | `u64 * <integer literal>` is computed in 32 bits (`2^32 * 2 == 0`, `d *= 256` wraps) | - | **fixed** by #570 (pre-alpha100: open) |
| 16 | `u64 %` goes through f64 (wrong above 2^53); `u64 /` has no integer division | - | `%` **fixed** by #571; `/` stays a number, as in JavaScript |
| 17 | Storing through `Ref<u8>` indexing (`bytes[i] = x`) is rejected: "saving to constant object" | - | **answered**: `Deref(bytes[i]) = x` stores; the error says so since #573 |
| 18 | `ns.f()` resolves to a local variable named `f` declared earlier in the function: "not a function to call" | - | **fixed** by #569 (pre-alpha100: open) |
| 19 | A @cpp overload call rejects a class argument for a `T \| null` or base-class parameter; a call whose argument the first overload can't take fails to compile | - | **fixed** (this PR; pre-alpha101: open) |

Verified downstream: native_ts_graphics dropped its 71 forwarding constructors (PR #49) and rebuilt
everything with pre-alpha97 (all objects deleted first): Windows build clean, CTest 61/61, Android
x86_64 build of 105 APKs clean, emulator smoke test 64/65 (the one failure, `oit_linked_lists`, was a
process that vanished without a crash dump after loading a 3624 x 3624 texture and passed when rerun
alone; no sign of a compiler cause).

---

## Fixed in pre-alpha97

### 1. Implicit constructor of a derived class did not forward its arguments

TypeScript gives a derived class with no constructor an implicit
`constructor(...args) { super(...args); }`. tslang generated `super()` with no arguments:

```ts
class Base {
    constructor(protected a: int, protected b: int = 0) {
    }
    sum(): int { return this.a + this.b; }
}
class Derived extends Base {
    extra = 1;                       // no constructor
}
function main(): void {
    const d = new Derived(2, 3);     // TypeScript: fine, prints 5
    console.log(d.sum());
}
```

pre-alpha96: `error: Expected 1-2 arguments, but got 0.` at `class Base {`, then `failed statement`.
pre-alpha97: prints `5`.

### 2. Diagnostic location

Was reported at the base class (1:1 of the file in the project's case), not at the derived class
whose implicit constructor was wrong. With (1) fixed the case no longer occurs; a wrong explicit call
is reported correctly on pre-alpha97:

```ts
class Derived extends Base {
    constructor() {
        super();                    // badsuper.ts:7:9: error: Expected 1-2 arguments, but got 0.
    }
}
```

The general point (code synthesized by the compiler should report the location of the construct it was
synthesized for) may still be worth keeping in mind for other synthesized members.

### 3. `u64` shift count

```ts
function main(): void {
    let a: u64 = 1;
    let b = a << 40;
    console.log(b);        // pre-alpha96: 256, pre-alpha97: 1099511627776
    let c: u64 = 1;
    c = c << 33;
    console.log(c);        // pre-alpha96: 2,   pre-alpha97: 8589934592
}
```

Not checked on pre-alpha97: `>>`, `>>>`, `i64`, `<<=`. Worth a regression test if the fix did not cover them.

### 4. `Ref<string>` field initialized with `null`

`protected argv: Ref<string> = null;` gives `can't cast from 'null' to 'Reference<string>' in 'strict null
mode'`, which is the intended strictness. The nullable spelling works (pre-alpha97):

```ts
class Holder {
    argv: Ref<string> | null = null;
}
```

---

## Open

### 5. `@cppvtable`: overload naming

From the notes on TypeScriptCompiler PR #550 (the `@cppvtable` decorator): C++ overloads of an abstract
interface method need distinct TypeScript names, and the MSVC slot order of overloads must be checked by
hand (`cl /d1reportSingleClassLayout<Class>`). A way to name the C++ overload (or a tool that prints the
slot map) would remove that manual step. See `docs/cppvtable-classes.md`. Not re-checked.

**Fixed**: `@cppvtable` is now `@cpp`, and each virtual method names its slot with `@slot(N)` (#561), which
tsbindgen takes from clang (#562), MSVC's reversed overload order included. C++ overloads are now declared
under the C++ name, each with its own `@slot` or `@linkname`, and a call picks one by its arguments as C++
does (`methods.over(5)`, `methods.over(2.5)`); no fit, or two equal fits, is an error listing the
overloads. tsbindgen no longer writes `_2`, `_3` (docs/cpp-classes.md).

### 6. C `string` return values are not copied

A C/C++ function declared as returning `string` hands tslang its raw `char*`, with no copy and no
lifetime tracking. A C helper that reused one `std::string` buffer for several calls returned garbage
to TypeScript later; the project now returns pointers to storage that lives as long as the owning object.
Decide whether this is by design; if so, the C interop docs should say the string must outlive its use,
otherwise copying on return (or an explicit borrowed annotation) would be safer. Not re-checked.

**Fixed**: worse than stale text - under `-mm=rc` the count was kept in the 8 bytes before the
pointer, so storing `strchr(base, 'k')` rewrote two characters of `base`. A `string` returned by a
top-level `declare function` (defined nowhere in the program) or a `@cpp` method is now copied at the
call; declare the result `Opaque` for the pointer itself (docs/c-cpp-header-import.md).

### 13. `T` not inferred from a generic tuple alias argument

Found on pre-alpha100, writing `stdVectorSize<T>(items: vector<T>)` for native_ts_graphics' NVRHI
bindings (`vector<T>` a generic tuple alias, as tsbindgen now writes C++ template specializations).
A call with a `Vec<Item>` argument doesn't infer `T`; giving it explicitly works:

```ts
type Vec<T> = [first: Ref<T> | null, last: Ref<T> | null];
type Item = [id: s32, size: s32];

function count<T>(v: Vec<T>): int {
    return <int>((<index><Opaque>v.last - <index><Opaque>v.first) / sizeof<T>());
}

function main() {
    let items: Item[] = [[1, 2], [3, 4]];
    const v: Vec<Item> = [Ref(items[0]), <Ref<Item>><Opaque>(<index><Opaque>Ref(items[0]) + 2 * sizeof<Item>())];
    print(count<Item>(v));  // 2
    print(count(v));        // error: not all types could be inferred
}
```

TypeScript infers `T = Item` here (the alias is expanded and matched structurally). native_ts_graphics
passes the type (`nvrhi.stdVectorSize<nvrhi.BindingSetItem>(...)`) until it is.

**Fixed by #572**: inference had no case for `Ref<T>` (a plain `f<T>(r: Ref<T>)` failed too). The same
PR stops `T | null` against `Item | null` from binding `T = null`: unions were paired by member position.

### 14. A literal-union ternary as an integer tuple field fails in codegen

Found on pre-alpha100 writing native_ts_graphics' `nvrhi.graphicsPipelineDesc()`: a
`static_vector`'s count given as `layout ? 1 : 0`. The ternary is `0 | 1` (a union of literals,
as in #8), and storing it into an `index` field of a tuple literal emits a sign extension of the
union itself, which the LLVM dialect rejects. The error has no source location:

```ts
type Vec = [items: [_0: Opaque | null], count: index];

function make(item: Opaque | null): Vec {
    return [[item], item ? 1 : 0];
}

function main() {
    print(make(null).count);   // expected 0
}
```

```
<unknown>:0: error: 'llvm.sext' op operand #0 must be signless integer or LLVM dialect-compatible
vector of signless integer, but got '!ts.union<!ts.literal<0 : si32, si32>, !ts.literal<1 : si32, si32>>'
```

#560 unwraps such a union for binary operators (`unwrapForBinaryOp`); the cast to a tuple field's
type needs the same. Workaround: `let count: index = 0; if (item) { count = 1; }`.

**Fixed by #568**: any cast of such a union (a variable, a parameter, a tuple field) goes through its
literals' type first.

### 15. `u64 * <integer literal>` is computed in 32 bits

Found on pre-alpha100: native_ts_graphics' `stdString()` built a `std::string`'s words with
`shift *= 256`, which wrapped to 0 after four bytes, so every debug name NVRHI got was cut to its
first four characters (and an entry point `main_vs` reached Vulkan as `main`):

```ts
function main() {
    let a: u64 = 4294967296;   // 2^32
    print(a * 2);              // 0, expected 8589934592
    let d: u64 = 16777216;     // 2^24
    d *= 256;
    print(d);                  // 0, expected 4294967296
    const k: u64 = 256;
    let e: u64 = 16777216;
    print(e * k);              // 4294967296: with a u64 operand it is right
}
```

The literal's type (s32) seems to win over the `u64` operand, so the multiply is done in 32 bits.
Workaround: a typed constant (`const BYTE: u64 = 256; shift *= BYTE`).

**Fixed by #570**: the numeric promotion order had no unsigned types, so only the literal's s32 matched.
`u64 op s32` is u64 now, and so is `u64 op index`; `u32 op s32` is still s32.

### 16. `u64 %` goes through f64; `u64 /` has no integer division

Found on pre-alpha100, reading the bytes of a 64-bit word:

```ts
function main() {
    const BYTE: u64 = 256;
    let w: u64 = 32499774568096109;   // the bytes of "main_vs\0", little-endian
    print(w % BYTE);                  // 108, expected 109 ('m')
    print(w / BYTE);                  // 1.26952e+014
    let small: u64 = 1000;
    print(small / 3);                 // 333.333
}
```

`/` giving a float follows JavaScript, but `%` on two `u64`s should be exact; it is done in
double precision, so it is wrong once the value passes 2^53. And with `/` always a float, a `u64`
has no exact integer division at all. Workaround: read the bytes through a pointer
(`<Ref<u8>><Opaque>Ref(copy)`), which needs no arithmetic.

**`%` fixed by #571**: with a 64-bit integer operand, `%` is done in that integer type (exact, and an
unsigned operand takes the unsigned remainder); `x % 0` traps there instead of giving NaN. `/` is
unchanged and still gives a number. `w >> 8` (exact since #556) with `w % 256` reads the bytes.

### 17. Storing through `Ref<u8>` indexing is rejected

Found on pre-alpha100, trying to write a `std::string`'s characters into a tuple's bytes. Reading
through the pointer works (`Deref(bytes[i])`); storing doesn't, `const` or `let`:

```ts
type Str = [buf: [_0: u64, _1: u64], size: index];

function main() {
    let value: Str = [[0, 0], 0];
    let bytes = <Ref<u8>><Opaque>Ref(value);
    bytes[0] = 65;             // error: saving to constant object
    print(value.buf._0);       // expected 65
}
```

Is there a way to store through a `Ref<T>` element (or is it by design)? If there is, the error
could name it.

**Answered**: `bytes[i]` is the element's reference for a store as for a read, so
`Deref(bytes[i]) = 65` stores (prints 65 above). Since #573 the error says so:
`indexing a 'Ref<T>' gives a reference to the element; store through it with 'Deref(p[i]) = value'`.

### 18. A namespace member call resolves to a same-named local

Found on pre-alpha100 in native_ts_graphics' `nvrhi_direct.ts`: a local `const textureDesc =
Deref(texture.getDesc())`, then `nvrhi.textureDesc(...)` further down the same function. The
qualified name `nvrhi.textureDesc` is resolved to the local, which isn't callable:

```ts
namespace ns {
    export function value(): int {
        return 42;
    }
}

function main() {
    const value = ns.value();   // 42
    print(value);
    print(ns.value());          // error: not a function to call
}
```

```
i18.ts:10:11: error: not a function to call
```

`ns.value` names the namespace's member whatever is in scope; the local should only be found by
the bare name `value`. Workaround: name the local differently.

**Fixed by #569**: the member is looked up in the namespace first. A name the namespace doesn't have
still falls back to the general lookup, so `ns.missing` can still find a local `missing`.

### 19. A @cpp overload call rejects a class for `T | null`, and checks the first overload's parameters

Found on pre-alpha101 in native_ts_graphics' `core/nvrhi.ts`, after tsbindgen started writing
NVRHI's overloads under one name: `device.createGraphicsPipeline(Ref(desc), framebuffer)` with
`framebuffer: IFramebuffer` fits neither overload, `(Ref<GraphicsPipelineDesc>, Ref<FramebufferInfo>)`
nor `(Ref<GraphicsPipelineDesc>, IFramebuffer | null)`:

```ts
@cpp
class Shape {
    @slot(1) area(): s32;
}

@cpp
class Square extends Shape {
    @slot(2) side(): s32;
}

@cpp
class Canvas {
    @slot(1) draw(x: s32): s32;
    @slot(2) draw(shape: Shape | null): s32;
}

// canvas: Canvas, shape: Shape, square: Square
canvas.draw(shape);    // error: invalid cast from Shape to s32
```

```
error: no overload of 'createGraphicsPipeline' takes these arguments (Reference<...>, nvrhi.IFramebuffer); it has:
  createGraphicsPipeline(Reference<...>, Reference<...>)
  createGraphicsPipeline(Reference<...>, nvrhi.IFramebuffer | null)
```

Two causes:
- an argument fit a parameter only by its own type, a number conversion or `canCastFromTo` (tuples
  and interfaces): a `Shape` didn't fit `Shape | null`, nor a `Square` a `Shape`;
- the overload was picked in the full pass only. The partial resolve before it (finding the
  function's types) called the first overload with the arguments, and an argument it can't be cast
  to (`Shape` to `s32` above) failed the function. The NVRHI call gave the "no overload" error
  instead because a class does cast to a `Ref<T>`.

Workaround: pass the argument through a local of the parameter's type
(`const arg: IFramebuffer | null = framebuffer;`).

**Fixed** (this PR): a class fits its own type and a `T | null` holding it (3, the same C++
pointer) and a base class, alone or in a union (2, a derived-to-base conversion). The overload is
picked in a partial resolve too; there a call that fits none (or two) keeps the first overload
without an error, and the full pass reports it.

---

## 8. A union of literals in arithmetic (found live on pre-alpha97)

`c ? 0.5 : 1.0` has the type `0.5 | 1`, a union of literals. A binary operator cast that union to
the other operand's type, so with an integer on the other side the fraction was truncated:

```ts
function main(): void {
    let c = true;
    const width = 100;                     // s32
    console.log(width * (c ? 0.5 : 1.0));  // pre-alpha97: 0, expected 50
    const h = c ? 0.5 : 1.0;
    console.log(width * h);                // pre-alpha97: 0
    const left = c ? 448.0 : 96.0;
    let m = left * 2;
    m = m / 3;
    console.log(m);                        // pre-alpha97: 298, expected 298.667
}
```

MLIR: `ts.Cast (!ts.union<!ts.literal<0.5>, !ts.literal<1.0>>) -> si32`, then an s32 multiply.
Fixed by #560: `unwrapForBinaryOp` unwraps a union whose members are
all literals of one type to that type, as it already did for a single literal. Regression test
`tslang/test/tester/tests/00literal_union_arith.ts`.

---

## Older reports (history)

Seen in real code on tslang versions up to pre-alpha93/94. TypeScriptCompiler's own findings
document (removed by #558, in the history) records #536, #537, #544 and DefaultLib#31 as their fixes;
the ternary one (8 above) was the only one still live.

| # | Symptom (as first seen) | Minimal repro tried | Result |
|---|---|---|---|
| 7 | Silent compiler crash, exit 0xC00000FD (stack overflow, no message), on `opaque != nullableOpaque` inside a method (`examples/async_compute_bloom.ts` `drawScene`, 2026-10-08) | class `Foo`; `p != this.cur` with `cur: Foo \| null` | fixed by #536; the original shape (`Opaque` ternary vs an `Opaque \| null` local in a loop) compiles and runs correctly on pre-alpha97 |
| 8 | Ternary of numeric literal consts inside arithmetic evaluated wrong, silently (`width * (split ? 0.5 : 1.0)` gave 0) | consts `HALF`/`FULL`, `width * (split ? HALF : FULL) + ((held ? 1.0 : 0.0) - (split ? 1.0 : 0.0))` | correct there because the consts were folded; with a runtime condition it was still wrong: see 8 above |
| 9 | namespace-level `const X = [false, true, ...]` read `false` for every element in `if (X[i])` | same shape, 4 elements, loop | correct |
| 10 | `Math.max` of all-negative numbers returned 2.2e-308 (seeded with `Number.MIN_VALUE`) | `Math.max(-1.0, -2.0)` | `-1`: fixed |
| 11 | array literal mixing an `f32` element and number literals, passed as `number[]`, failed to cast | `[f, 0, 2]` assigned to `number[]` | works |
| 12 | `let x = 0;` later assigned a fraction silently kept an `s32` | `let x = 0; x = 2.5;` | now a clear warning ("a 'number' is stored in a 's32'..."): fixed |
