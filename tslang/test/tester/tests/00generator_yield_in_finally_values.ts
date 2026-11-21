// A yield in a finally block of a generator, with values that live on the heap: a string or an object returned
// through it (kept for after the finally), and return() while the generator is paused in a yield* (it finishes
// through the finally, as in JavaScript - the delegate is left where it is). The results are node's for the same
// JavaScript. Not under -mm=own: a value that owns memory returned through such a finally is an error there yet
// (diagnostic-location/generator_return_through_yielding_finally_own.ts), and so is a yield* of a Generator<T>.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return: (value: T) => IterateResult<T>;
    throw: (e: any) => IterateResult<T>;
}
class Item { constructor(public n: number) {} }

function* strRet(n: number) { try { yield "1"; return "r" + n; } finally { yield "2"; } }
function* objRet(n: number) { try { yield new Item(1); return new Item(n); } finally { yield new Item(2); } }
function* inner(): Generator<number> { try { yield 1; yield 2; } finally { } }
function* outer(): Generator<number> { try { yield* inner(); } finally { yield 9; } }

function main() {
    for (let round = 0; round < 200; round++) {
        let s = "";
        for (const it = strRet(3); ;) { const r = it.next(); s += (r.done ? "D" : "") + r.value + " "; if (r.done) break; }
        assert(s == "1 2 Dr3 ", "a string returned through it");

        let o = "";
        for (const it = objRet(4); ;) { const r = it.next(); o += (r.done ? "D" : "") + r.value.n + " "; if (r.done) break; }
        assert(o == "1 2 D4 ", "an object returned through it");
    }

    const it = outer();
    it.next();
    const a = it.return(7);
    const b = it.next();
    assert(a.value == 9 && !a.done && b.value == 7 && b.done, "return() in a yield*");
    print("done.");
}
