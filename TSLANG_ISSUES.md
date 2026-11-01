# tslang issues found while working on native_ts_graphics

Issues hit while porting https://github.com/ASDAlexander77/native_ts_graphics to tslang. Every repro
should be a complete file that was run (`tslang file.ts`, JIT, Windows x64), with the tslang version.

## Open

Items 1-21 (collected 2026-10-12 to pre-alpha111) are fixed or answered; their write-ups are in this
file's git history. The last one, 21 (a C `string` result not copied when its `.d.ts` was first
loaded by an imported module), was fixed by #761 and checked with pre-alpha112: `strerror(2)` and
`strerror(13)` called from the main module and from an imported one give four different texts.

### 22. `T | null` of a `@value` type whose only field is a class or interface: compiler crash

tslang v0.0-pre-alpha112 (Windows x64). Found 2026-10-31. A regression from #760 (pre-alpha111
compiled all of these).

#760 stores `T | null` of a `@value` type T without a tag when T has a class or interface field
(`MLIRTypeHelper::getValueNullNiche`): null is T with that pointer zeroed. When that field is T's
**only** field, the lowering fails:

- comparing such a value with null (`== null`, `=== null`, `if (v)`) crashes tslang (0xC0000005,
  exit 139), under every memory model (gc and rc tried), JIT and `--emit=obj`/`--emit=llvm` alike;
- passing a `@value` class value to a `T | null` parameter fails with
  `error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast`.

Add a second field (a number, before or after the reference) and both compile and print the right
results. `--emit=mlir` succeeds in every case; it is the lowering to LLVM that fails.
`test/tester/value/value_missing_niche.ts` passes with pre-alpha112: its `Hit` and `Mark` have two
fields each.

Repro 1, `one_null.ts` - `tslang --emit=jit one_null.ts`:
```ts
class H { x = 1; }

@value
interface One {
    readonly h: H;
}

function f(d: number): One | null { return d < 0 ? null : { h: new H() }; }

function main() {
    print(f(-1) == null, f(1) == null);
}
```
Expected `true false`. Actual: "PLEASE submit a bug report ... Exception Code: 0xC0000005", stack
all in tslang.exe. With `readonly n: number;` added to `One` (and `n: 1` in the literal) it prints
`true false`. The same crash with a `@value class` with a constructor, and with a top-level
`let p: One | null = null; print(p == null);`.

Repro 2, `one_param.ts` - `tslang --emit=jit one_param.ts`:
```ts
class H { x = 1; }

@value
class One {
    readonly h: H;
    constructor(h: H) { this.h = h; }
}

function take(o: One | null): int { return 1; }

function main() {
    print(take(new One(new H())));
}
```
Expected `1`. Actual:
```
error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
tslang: error: failed to emit LLVM IR
```
With a second field (`readonly n: number;`, set in the constructor) it prints `1`. The same
program with a `@value interface` and an object literal (`const o: One = { h: new H() }; take(o)`)
compiles and prints `1`.

MLIR (repro 1, `--emit=mlir`): `f` returns
`!ts.optional<!ts.union<!ts.null, !ts.value<@One, !ts.tuple<{"h",!ts.class<@H, ...>}>>>>`, null made
as `ts.Cast (!ts.null) -> !ts.union<!ts.null, !ts.value<@One, ...>>`, the value as
`ts.Cast (!ts.value<@One, ...>) -> !ts.union<...>`: the niche (untagged) form, as for two fields.

Suspect: the lowering of a one-field `@value` type next to the niche code. `getValueNullNiche` gives
the path `{0}` for the one field; `valueNullCompare` (LowerToLLVM/UnaryBinLogicalOrHelper.h)
`LLVM::ExtractValueOp`s that path from `asLLVMUnion(...)`, and the null/value casts into the union
(CastLogicHelper.h, near lines 743 and 1313) build it the same way. If a one-field tuple's LLVM type
(or the union's converted type for it) isn't the `{ptr}` struct those expect - a bare `ptr`, or a
different struct than the value's - the extract/insert has no element 0 (the crash) and the type
conversion leaves an `unrealized_conversion_cast` (repro 2).

How native_ts_graphics hits it: core/donut.ts (generated) wraps every Donut object as a `@value`
class whose only field is `readonly handle: XHandle` (an `@cpp declare class`), and takes them as
`X | null` parameters (`App.createToneMappingPass(..., previousToneMappingPass: ToneMappingPass |
null)` does `previousToneMappingPass ? (previousToneMappingPass as ToneMappingPass).handle : null`).
With pre-alpha112, compiling core/donut.ts crashes tslang (exit 139, desktop and Android), and
examples/feature_demo.ts:942 (the `createToneMappingPass` call) fails with repro 2's error, so no
example builds. No workaround in native_ts_graphics yet: it can't build with pre-alpha112 until this
is fixed (a dummy second field in every wrapper would change the generated classes' layout for
nothing).

### 23. `h && cond` with `h` a `@cpp` class `| null`: "instanceof can't be used with a @cpp class"

tslang v0.0-pre-alpha113 (Windows x64). Found 2026-10-31.

A `@cpp` handle that may be null tests fine alone (`if (h)`, `if (!h)`, and `if (!a || !b)` of two
of them), but not as the operand of `&&` with a boolean: the condition fails to compile with an
`instanceof` error the source never wrote.

Repro, `repro.ts` - `tslang --emit=mlir repro.ts` (any emit; `-mm=gc`):
```ts
@cpp declare class Handle { private readonly __handle: index; }
declare function make(): Handle | null;
declare function use(h: Handle): void;

function main() {
    const h = make();
    const n = 3;
    if (h && n == 3) {
        use(h);
    }
}
```
Expected: compiles (the condition is h != null and n == 3). Actual:
```
repro.ts:8:5: error: instanceof can't be used with a @cpp class: its objects are C++ objects, with no type information
repro.ts:8:5: error: can't instantiate function. '___cast' not all generic types can be identified
```
`if (h) { if (n == 3) { use(h); } }` compiles and runs.

Suspect: `h && n == 3` is typed as a union (`Handle | null | boolean`), and making it a boolean goes
through `MLIRGenImpl::castFromUnion` (MLIRGenCast.cpp, near line 3104), which generates the generic
`___cast<T, U>` with an `instanceof` test for each class member of the union; the `instanceof`
lowering (MLIRGenImpl.h, near line 6610) rejects a `@cpp` class there. For a `@cpp` member (a plain
pointer with no vtable of tslang's) the union's truthiness needs only the null test, not
`instanceof`.

How native_ts_graphics hits it: examples/basic_triangle_nri.ts (NRI handles, `@cpp declare class`es
in core/nri/nri_interop.d.ts) had `if (readbackBuffer && frameIndex == TEST_FRAMES - 1)`.
Workaround there: nested `if`s.
