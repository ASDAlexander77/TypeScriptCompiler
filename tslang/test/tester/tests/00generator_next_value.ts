// next(value) of a generator typed Generator<T, TReturn, TNext>: the value is what the yield it resumes
// evaluates to, a TNext (its default when next() is given none). A generator whose next() takes no value is
// a Generator too, and a for...of calls next() with none.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
}

function* echo(): Generator<string, void, string> {
    let last = "start";
    while (true) {
        const got = yield last;
        last = "got " + got;
    }
}

function* counter(): Generator<number> {
    for (let i = 0; i < 3; i++) yield i;
}

function main() {
    const e = echo();
    print(e.next().value);
    print(e.next("a").value);
    print(e.next("b").value);

    for (const v of counter()) print(v);

    const c = counter();
    print(c.next(123).value);

    print("done.");
}

