// -mm=own: a generator method keeps its `this` unless it says it borrows it, so `yield* this.range()` would give the
// object to a second owner (TSLANG 217, 159: with `@borrows("this")` on both methods it compiles). An error that says so.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
}

class R {
    n: number = 3;
    *range(): Generator<number> {
        for (let i = 0; i < this.n; i++) yield i;
    }
    *all(): Generator<number> {
        yield 100;
        yield* this.range();
    }
}

function main() {
    let s: number = 0;
    for (const v of new R().all()) s += v;
    print(s);
}
