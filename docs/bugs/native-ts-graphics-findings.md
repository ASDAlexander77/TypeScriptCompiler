# tslang issues found while porting graphics samples (native_ts_graphics)

These issues came up while porting about 100 D3D12 and Vulkan samples to TypeScript in
[native_ts_graphics](https://github.com/ASDAlexander77/native_ts_graphics), from October 2026.
Each one has a minimal repro beside this file (`ntg_*.ts`), written in the style of the other
files here (`print` / `assert`, top-level code).

**Verified on:** `tslang version v0.0-pre-alpha92` (LLVM 22.1.8), the `I:\tslang\tslang.exe` build
of 2026-10-06, on Windows 11, 2026-10-11. Run each repro with:

```
tslang --emit=jit --jit-cache=false docs/bugs/ntg_<name>.ts
```

Every repro below failed on that build: a failed assertion (exit code 3), a compile error, or a
compiler crash. Issue 7 is newer: it comes from pre-alpha93's JIT cache, so its repro runs *without*
`--jit-cache=false` (see there).

**Status (2026-10-11):** each issue has a fix with a regression test in the compiler's suite
(`tslang/test/tester`); the "Fix" column links it. Issue 3 is deliberate (integer literals are
`s32`), so its fix is a warning rather than a change of type, and `ntg_let_zero_is_integer.ts`
keeps failing by design. The repros stay here as the record of what was reported.

| # | Issue | Kind | Repro | Fix |
|---|---|---|---|---|
| 1 | Comparing `Opaque` with `Opaque \| null` crashes the compiler | Compiler crash, no message | `ntg_opaque_nullable_compare.ts` | [#536](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/536) |
| 2 | Boolean array elements read back wrong | Wrong value, silent | `ntg_boolean_array_elements.ts` | [#537](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/537) |
| 3 | `let s = 0` is an integer, so `s += 1.5` truncates | Wrong value, silent | `ntg_let_zero_is_integer.ts` | warning: [#544](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/544) |
| 4 | Escapes not processed in templates with substitutions | Wrong value, silent | `ntg_template_literal_escape.ts` | [#539](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/539) |
| 5 | `Math.max` of negative numbers returns 2.2e-308 | Wrong value, silent (default lib) | `ntg_math_max_negative.ts` | [DefaultLib#31](https://github.com/ASDAlexander77/TypeScriptCompilerDefaultLib/pull/31), and [#535](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/535) for why its test passed |
| 6 | `boolean && (Class \| null)` as a condition fails to compile | Compile error | `ntg_and_nullable_class.ts` | [#540](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/540), and [#541](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/541) for its misplaced first error |
| 7 | The JIT cache crashes a run that compiles two or more modules (pre-alpha93) | Compiler crash, no message | `ntg_jit_cache_import.ts` | [#549](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/549) |

The silent ones (2 to 5) cost the most time: the programs compile and run, and only the output
is wrong.

---

## 1. Comparing `Opaque` with `Opaque | null` crashes the compiler

```ts
class State {
    current: Opaque | null;
    ...
    set(next: Opaque): boolean {
        if (next != this.current) { ... }
```

**Observed:** tslang exits with `0xC00000FD` (stack overflow) and prints nothing: no diagnostic,
no line number. Under Ninja the build step just shows `FAILED`.

**Expected:** compiles (it's an identity comparison of two handles), or at least a diagnostic.

**Impact and workaround:** this happened in a renderer that tracked the currently bound pipeline
to skip redundant binds. Finding it took a bisection, stubbing method bodies one at a time. The
workaround is to track such state with an integer (e.g. a pipeline index, -1 for none). From
bash the exit code shows as 127; run tslang from cmd or PowerShell to see `0xC00000FD`. (127 is
how Git Bash reports that crash, not a missing DLL.)

**Fixed by #536:** the lowering of `!=` for a nullable left operand and a tag-less union on the
right called itself again with the same operand types, until the compiler's stack ran out.

## 2. Boolean array elements read back wrong

```ts
const FLAGS = [false, true, false, true];
let n = 0;
for (let i = 0; i < FLAGS.length; i++) {
    if (FLAGS[i]) { n++; }
}
print(n);              // 10 (expected 2)
const b: boolean = FLAGS[1];
print(b);              // false (expected true)
```

**Observed:** `if (FLAGS[i])` is never taken for any element, a counted loop gives a nonsense
count (10 for 4 elements), and printing the elements one by one inside a namespace function gave
them inverted (`true false true false`). The array's `length` is right. This happens both at the
top level and inside namespaces.

**Expected:** 2, true.

**Impact and workaround:** one port's table of per-axis rotation flags was silently skipped, so
bounding boxes weren't rotated. The workaround is number arrays compared with `!= 0`. It looks
like an element-type or width mismatch (i1 vs i8?) in boolean array loads.

**Fixed by #537:** the guess was right. The literal's constant data was an LLVM vector
`<4 x i1>`, which packs the elements into bits, while elements are read one byte apart. Such data
is now an array, one byte per element.

## 3. `let s = 0` is an integer, so adding a fraction truncates

```ts
let s = 0;
s += 1.5;
print(s);              // 1 (expected 1.5)

function sum(values: number[]): number {
    let a = 0;
    for (...) a += values[i];
    return a;          // sum([1.5, 0, 2]) gives 3, expected 3.5
}
```

**Observed:** the variable's type is inferred from the integer literal as an integer, so every
later `+=` / `= s + x` truncates. `let u: number = 0` works.

**Expected (TypeScript semantics):** `s` is a `number` (a double). This may be a deliberate
design choice for performance. If so, it's worth documenting prominently, because ordinary
TypeScript code (accumulators, running totals) changes meaning silently. A warning when a
fractional value is assigned to an integer-inferred variable would catch it.

**Workaround:** write `0.0`, or annotate `: number`.

**Status:** by design: integer literals are `s32`, and that stays. #544 adds the suggested
warning: storing a `number` into a variable whose type was inferred from an integer literal says
that any fraction is dropped, and suggests declaring it `number`.

## 4. Escapes not processed in template literals with substitutions

```ts
const n = 1;
const s = `${n}\n${n}`;
print(s.length);       // 4: "1", "\", "n", "1" (expected 3)
```

**Observed:** the characters around a substitution keep their escape sequences literally. A
template without substitutions (`` `a\nb` ``) and a string literal (`"a\nb"`) are fine.

**Expected:** 3 (a newline character).

**Impact and workaround:** two-line ImGui labels (`` `Timestamps Period:\n ${period} ns` ``)
showed a literal `\n` on screen. Split such texts into separate strings, or concatenate a
`"\n"` string literal.

**Fixed by #539:** the text around a substitution was taken as written instead of with its
escapes processed. Tagged templates and template literal types had the same mistake.

## 5. `Math.max` of negative numbers returns 2.2e-308

```ts
print(Math.max(-3, -5));   // 2.22507e-308 (expected -3)
```

**Observed:** the default library's `Math.max` (TypeScriptCompilerDefaultLib `src/lib.ts`, line
1552: `let m = Number.MIN_VALUE;`) starts its running maximum at `Number.MIN_VALUE`, the smallest
*positive* double, rather than `-Infinity`. So any all-negative input returns 2.2e-308. The
library's own `tests/math.ts` (line 107, `isEq(Math.max(-1, -7), -1)`) checks exactly this case.
`Math.min` of positive numbers is fine. Also, `Number.MIN_VALUE` there is 2.22507385855e-308 (the
smallest *normal* double); in JavaScript it is 5e-324 (the smallest subnormal), as the comment
beside it notes.

**Expected:** -3. Start from `-Infinity`, or from the first argument (and `Math.max()` with no
arguments is `-Infinity`).

**Impact and workaround:** a camera's pitch clamp broke. Clamp with explicit comparisons.

**Fixed by DefaultLib#31:** `Math.max` starts from `-Infinity` and `Math.min` from `+Infinity`,
and a NaN argument gives NaN. `tests/math.ts` didn't catch it because its `isEq(x: any, y: any)`
uses `!==`, which answered false for a `number` and an integer boxed in `any`s. #535 fixes that,
and with it the test reports this bug. `Number.MIN_VALUE` is unchanged.

## 6. `boolean && (Class | null)` as a condition fails to compile

```ts
function use(enabled: boolean, handle: Handle | null): number {
    if (enabled && handle) {
        return handle.value;
    }
    return 0;
}
```

**Observed:**

```
error: can't instantiate function. '___cast' not all generic types can be identified
    if (enabled && handle) {
error: failed statement
```

**Expected:** compiles, and `handle` is narrowed to `Handle` inside the block.

**Workaround:** nest the conditions: `if (enabled) { if (handle) { ... } }`.

**Fixed by #540:** not specific to `&&`. Any truth test of a union of null and two or more other
types failed, and `enabled && handle` is a `boolean | Handle | null`. The generated union cast
returned null for the null member, which can't be a boolean. The repro's first error, reported
at line 1 inside the comment, was a separate bug: errors in that generated code took its offsets
as positions in the user's file (#541).

## 7. The JIT cache crashes a run that compiles two or more modules

Found on `tslang version v0.0-pre-alpha93` (the `I:\tslang\tslang.exe` build of 2026-10-11), the
first build with the JIT cache on by default. The repro is two files:

```ts
// ntg_jit_cache_import_lib.ts
export function twice(x: number): number {
    return x * 2;
}

// ntg_jit_cache_import.ts
import { twice } from "./ntg_jit_cache_import_lib";
print(twice(21));   // crashes before running (expected 42)
```

Run it **without** `--jit-cache=false`, from an empty cache (delete `docs/bugs/__jit` first):

```
tslang --emit=jit docs/bugs/ntg_jit_cache_import.ts
```

**Observed:** an access violation (`Exception Code: 0xC0000005`) inside `tslang.exe` before the
program starts, with the usual "PLEASE submit a bug report" stack dump (only
`__jit_debug_register_code` frames in a release build). The cache folder then holds the importing
file's object only. Run it again and it works: the second run loads that object from the cache and
compiles only the imported file. In general, a run that has to compile two or more modules crashes
after writing the first, so each crashing run caches one more module; a program of N modules runs
on the N-th attempt and from then on, until a source changes. With `--jit-cache=false` (one module
per run) every run works.

**Expected:** 42, done, on the first run.

**Impact and workaround:** every native_ts_graphics example imports `core/input_pass.ts`, which
imports `core/donut.ts`, so every example crashed under the JIT twice before running.
`run_jit.bat` passes `--jit-cache=false`.

**Fixed by #549:** the cache compiles each module in an MLIRContext of its own, and the built-in
types (`i32`, `f32`, `Opaque`, ...) were kept in a `static` map, made once, in the first module's
context. Every later module in the run got types from a context destroyed by then. The map is
now per module. The default library's declarations name such types, which is why every program
importing a module crashed; without it (the suite runs with `--no-default-lib`) a module only
crashed if it named one itself. Regression test: `test-jit-cache-embedded-types`, which runs from
an empty cache every time.

---

## Seen earlier but not reproduced in minimal form

These came up in larger programs during the port series, but small repros of them pass on the
build above. Either they were fixed since, or they need more context than the obvious minimal
form. They're listed in case they ring a bell:

- **Literal-union ternaries in arithmetic:** `width * (split ? 0.5 : 1.0)` evaluated to 0, and
  `(held ? 1.0 : 0.0) - (...)` broke a camera's movement. Assigning the ternary to a `: number`
  local first fixed it. Separately, `const left = c ? 448.0 : 96.0` got a literal-union type that
  wouldn't mix with other numbers in arithmetic or ternaries without a `: number` annotation.
  Minimal versions, with `number` and with `f32` operands, now compute the right values.
- **An array literal mixing `f32` values and number literals, passed as `number[]`:** failed with
  `invalid cast from [f32, 0, ...]`. It now compiles and passes the right values.
- **Methods passed as callbacks** (`pass.setRenderCallback(this.onRender)`) seemed to resolve only
  for members declared earlier in the class, so the examples declare `init()` (which registers the
  callbacks) last. A minimal class calling `invoke(this.onRender)` with `onRender` declared after
  `init` works.

## Interop behaviors worth documenting

These aren't necessarily bugs, but each cost time because the behavior was silent:

- **`string` returns from declared C functions keep the raw `char*` without copying.** A C++
  helper that reused one `std::string` buffer for several calls handed TypeScript strings that
  changed under it (garbage after the next call). The C side must return pointers to storage that
  outlives the TypeScript string.
- **`number` parameters of declared C functions are doubles.** A C++ function declared with a
  `float` parameter reads garbage, with no error (rotations came out as identity). The C side must
  take `double`.

## The other repros in this folder

Reviewed alongside the issues above, on the same build:

| File | Status |
|---|---|
| `testFallThrough.ts` | A `default` clause without `break` jumped out of the switch instead of falling through. Fixed by [#538](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/538). |
| `generic_in_generic.ts` | Nested untyped arrows with a same-named parameter, capturing a variable, failed to instantiate: their implicit type parameters had the same name. Fixed by [#542](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/542). |
| `array_of_ifaces_cast.ts` | The error it expects (an array passed for an interface) was never reported; such a cast compiled to garbage. Now an error, by [#543](https://github.com/ASDAlexander77/TypeScriptCompiler/pull/543). The file still stops earlier, at errors TypeScript reports too (`return null` under strict null checks, `IToken[]` to `IStateToken[]`). |
| `bug14.ts` | Circular `typeof` / `ReturnType` aliases. TypeScript reports these as circular too; tslang reports errors and doesn't crash. Not a bug. |
| `bug21.ts` | `g: new (x: this) => x` isn't valid TypeScript (`x` isn't a type). Not a bug. |
