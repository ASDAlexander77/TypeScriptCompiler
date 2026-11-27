// -mm=own: a generator method of a class the module keeps to itself borrows its `this` (TSLANG 217, user ruling
// 2026-11-27), so the generator may not outlive the object it is made on: returned from the function that made the
// object, it would. An error.
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
}

function make(): Generator<number> {
    return new R().range();
}

function main() {
    let s = 0;
    for (const v of make()) s += v;
    print(s);
}
