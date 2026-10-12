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
| 5 | `@cppvtable` overload naming (C++ overloads need distinct TypeScript names) | open | **addressed** on branch `cppvtable-slot`: `@cpp` + `@slot(N)` (docs/cpp-classes.md) |
| 6 | C `string` return values keep the raw `char*` (no copy) | open | open, not re-checked |
| 7 | `Opaque` compared with `Opaque \| null` crashed the compiler | - | **fixed** by #536 (re-checked with the original drawScene shape) |
| 8 | A ternary of number literals in arithmetic truncated to the other operand's integer type | - | **fixed** by #560 (still wrong on pre-alpha97) |
| 9-12 | Bool const arrays, `Math.max`, `f32` array literal, `let x = 0` | - | fixed (#537, DefaultLib#31, works, #544 warning) |

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

### 6. C `string` return values are not copied

A C/C++ function declared as returning `string` hands tslang its raw `char*`, with no copy and no
lifetime tracking. A C helper that reused one `std::string` buffer for several calls returned garbage
to TypeScript later; the project now returns pointers to storage that lives as long as the owning object.
Decide whether this is by design; if so, the C interop docs should say the string must outlive its use,
otherwise copying on return (or an explicit borrowed annotation) would be safer. Not re-checked.

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
