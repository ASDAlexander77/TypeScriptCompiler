# tslang issues found while working on native_ts_graphics

Issues hit while porting https://github.com/ASDAlexander77/native_ts_graphics to tslang. Every repro
should be a complete file that was run (`tslang file.ts`, JIT, Windows x64), with the tslang version.

## Open

Items 1-23 (collected 2026-10-12 to pre-alpha113) are fixed or answered; their write-ups are in this
file's git history. The last two: 22 (`T | null` of a `@value` type whose only field is a class or
interface crashed the compiler, a regression from #760) was fixed by #763; 23 (`h && cond` with `h`
a `@cpp` class `| null` failed with "instanceof can't be used with a @cpp class") was fixed by #765,
regression test `tslang/test/tester/tests/00cpp_union_truth.ts`.

### 26. `Ref(x)` of a `number` (f64) is accepted for a `Ref<f32>` parameter, passing a `double*` as `float*`

tslang v0.0-pre-alpha115, Windows x64. Repro (compiled with `tslang -emit=llvm r.ts -o r.ll`, no
diagnostic):

```ts
declare function Fill(enable: int, dst: Ref<f32>): int;
class A {
    sensed: number[] = [0.0, 0.0, 0.0];
    run() {
        Fill(1, Ref(this.sensed[0]));
        print(this.sensed[0]);
    }
}
function main() { const a = new A(); a.run(); }
```

IR: `%14 = getelementptr double, ptr %12, i32 0` then `call i32 @Fill(i32 1, ptr %14)`: the double
array's pointer goes to the `float*` parameter unchanged. A native function writing 3 floats fills
the first double with the x and y floats' bits (the double's sign and exponent come from y) and half
of the second. Expected: a type error (`Ref<number>` isn't a `Ref<f32>`). The `Ref<f32>` declaration
is meant to type-check this; with `sensed: f32[]` the IR is `getelementptr float`, as expected.

Where it bit: native_ts_graphics' powder_sandbox_nri, tilt gravity, `Nri_GetGravity(1,
Ref(this.sensed[0]))` with `sensed: number[]`. On a phone the gravity pointed right when held upright,
left upside down, and stayed put when held sideways. Workaround: declare the array `f32[]` (branch
powder-tilt-gravity-f32).

### 25. `cond ? [1] : []` (an empty array literal against a non-empty one) crashes when iterated

tslang v0.0-pre-alpha113, JIT, Windows x64. Repro (run):

```ts
class S {
    scene: int = 4;

    check(): int {
        const expected = this.scene == 4 ? [1] : [];
        let sum = 0;
        for (const k of expected) {
            sum += k;
        }
        return sum;
    }
}

function main() {
    console.log(`${new S().check()}`);
}
```

It crashes (`Exception Code: 0xC0000005`) when the non-empty branch is taken. With `this.scene == 3`
(the empty branch taken) it prints 0; `cond ? [1] : [2, 3]` prints 1 and `cond ? [1, 2] : [3, 4]`
prints 3, so it's the empty literal against a non-empty one. `--emit=llvm`: the ternary's result is
`alloca <{ ptr, <{ [1 x i64] }> }>`, a tagged union (a type descriptor, `@td_..._array`, then the
array), each branch storing `<{ ptr, ptr }>` into it, rather than one array type (TypeScript gives
the expression `number[]`); `for...of` over that union is where it fails.

Where it bites: native_ts_graphics' powder_sandbox_nri, the sound events a test scene must have,
`this.scene == 2 ? [0, 2] : (this.scene == 4 ? [1] : (this.scene == 5 ? [2, 3] : []))`, segfaulted
in scene 4 right after logging them. Workaround: `const expected: int[] = [];` and `push` the values
in `if` branches.

### 24. An array literal holding a `const X = 1 << n` widens to 64-bit elements

tslang v0.0-pre-alpha113, JIT and `--emit=llvm`, Windows x64. Repro (run):

```ts
const CAP = 1 << 18;

class Grid {
    w: int = 290;
    size(): int {
        const a = [CAP, this.w];
        return a.length;
    }
}

function main() {
    console.log(`${new Grid().size()}`);
}
```

It runs (prints 2), but `a` is an array of `i64`: the IR has `alloca <{ i64, [2 x i64] }>`, although
`@CAP = private constant i32 262144` (CAP itself is stored as i32, loaded with `load i32, ptr @CAP`)
and `w` is `int`. Without CAP the same literal is `[2 x i32]`: `[0x53445750, this.w]`,
`[1, this.w]` and `[this.w, this.w]` are all i32; `[CAP, this.w]`, `[1, this.w, CAP]` and
`[0x53445750, 1, CAP]` are i64. So the shift's result type seems to widen to a 64-bit integer when
the array's element type is inferred (TypeScript gives `<<` a 32-bit result), while the constant's
storage stays i32.

Where it bites: `Ref(header[0])` passed to a C function taking `const int*` (native_ts_graphics'
powder_sandbox_nri save header `[SAVE_MAGIC, SAVE_VERSION, this.gridW, this.gridH, CAPACITY]` with
`CAPACITY = 1 << 18`) handed it 8-byte elements, so the file's header came out as magic, 0,
version, 0, ... Workaround: annotate the array (`const header: int[] = [...]`), which gives `[5 x i32]`.
