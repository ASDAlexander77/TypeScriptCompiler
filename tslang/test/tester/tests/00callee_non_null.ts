// TSLANG 187: `f!(x)` of an optional function calls the function, and throws a TypeError where it is missing, as
// JavaScript calling undefined does (the call was skipped, its result a default). Under -mm=own the result of such a
// call can be kept (`r = g.throw!(e)`): it is the call's, not one of the call and a default.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    throw?: (e: any) => IterateResult<T>;
}

interface I {
    f?: () => number;
}

function* inner(k: number): Generator<string> {
    try {
        yield "a" + k;
        yield "b" + k;
    } catch (e) {
    }
    yield "c" + k;
}

function main() {
    const a: I = { f: () => 5 };
    assert(a.f!() == 5, "present");

    const b: I = {};
    let caught = false;
    try {
        b.f!();
    } catch (e) {
        caught = true;
    }
    assert(caught, "missing: a TypeError");

    let g: (() => string) | undefined = () => "s" + 1;
    assert(g!() == "s1", "an optional function value");

    const it = inner(1);
    let r = it.next();
    assert(r.value == "a1", "a1");
    r = it.throw!("x");
    assert(r.value == "c1", "kept");

    print("done.");
}
