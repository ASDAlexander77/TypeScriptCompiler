// What an `await` gives back used to travel through `!async.value<T>`, and MLIR's async-to-LLVM
// conversion - which runs before this compiler's own types are lowered - converts that payload
// with an LLVM type converter that knows nothing about the TypeScript dialect. So awaiting
// anything whose type belongs to this dialect failed with "failed to legalize operation
// 'async.runtime.load'": `number`, `string`, `boolean`, a class, an array. Only the payloads that
// happened to be builtin MLIR types - `i32`, `i64`, `f32` - ever compiled, which is why every
// async test in this suite was written to return one of those.
//
// The result now travels through a slot in the awaiting function and the async value carries only
// a token, so there is no payload left to convert. See docs/reference-counting-evaluation.md
// section 9.56.
//
// Arguments were never the problem, despite how it first looked: `await twice(3.0)` failed and
// `await withDefault()` compiled because of what they RETURN, not what they take. Both directions
// are covered here.

class Point {
    x: number;
    y: number;

    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
}

async function aNumber(n: number): number {
    return n * 2.0;
}

async function aString(s: string): string {
    return s + "!";
}

async function aBoolean(n: number): boolean {
    return n > 0.0;
}

async function aClass(x: number, y: number): Point {
    return new Point(x, y);
}

async function anArray(n: number): number[] {
    return [n, n + 1.0, n + 2.0];
}

async function anI32(n: i32): i32 {
    return n + n;
}

async function nothing(): void {
    // A void async function yields a token and no value at all - the case that always worked,
    // kept so the token-only path stays covered from both sides.
}

// Awaiting inside an async function, rather than from a plain one: this is a coroutine resumed
// from inside another coroutine, and it failed for the same reason (its result is a `number`).
async function awaitsInsideAsync(n: number): number {
    const half = await aNumber(n);

    return half + 1.0;
}

// The result has to survive the await, not just be produced by it. Under reference counting the
// value is built inside the awaited region, whose own temporaries are released as that region
// ends - so the slot has to own what it holds, or this reads freed memory. It read correctly at
// -O0 and garbage at -O3 before the slot took ownership, which is only ever a question of what
// reused the block first.
function churn(): number {
    let total = 0.0;
    for (let i = 0; i < 64; i++) {
        let filler = new Point(999.0, 999.0);
        total = total + filler.x * 0.0;
    }

    return total;
}

function classResultOutlivesTheAwait(): number {
    const p = await aClass(3.0, 4.0);
    churn();

    return p.x + p.y;
}

function stringResultOutlivesTheAwait(): number {
    const s = await aString("ab");
    churn();

    return s.length;
}

function arrayResultOutlivesTheAwait(): number {
    const a = await anArray(1.0);
    churn();

    return a[0] + a[1] + a[2];
}

function main() {
    assert(await aNumber(2.0) == 4.0, "a `number` result");
    assert(await aString("ab") == "ab!", "a `string` result");
    assert(await aBoolean(1.0), "a `boolean` result");
    assert(await anI32(3) == 6, "an `i32` result still works");
    assert(await aClass(1.0, 2.0).y == 2.0, "a class result");
    assert(await anArray(1.0).length == 3, "an array result");
    await nothing();

    assert(await awaitsInsideAsync(2.0) == 5.0, "awaiting inside an async function");

    assert(classResultOutlivesTheAwait() == 7.0, "a class result survives the await");
    assert(stringResultOutlivesTheAwait() == 3, "a string result survives the await");
    assert(arrayResultOutlivesTheAwait() == 6.0, "an array result survives the await");

    print("done.");
}
