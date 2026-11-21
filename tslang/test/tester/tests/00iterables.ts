// An array or a string given where an iterator interface is wanted (`Iterable<T>`: an object with `next`, as the
// default library declares it) was "type 'number[]' can't be cast to interface 'Iterable<number>': it has none of
// the interface's members". A generator over its elements is given; an `s32[]` given to an `Iterable<number>` is
// converted first. (-mm=own: a generator typed as an interface other than the library's is still refused.)
// as the default library declares them (the tests compile without it)
type IterateResult<T> = { value: T, done: boolean };
interface Iterable<T> {
    next: () => IterateResult<T>;
}
interface IterableIterator<T> {
    next: () => IterateResult<T>;
}
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: () => IterateResult<T>;
}

function sum(v: Iterable<number>) {
    let s = 0.0;
    for (const x of v) {
        s += x;
    }

    return s;
}

function cat(v: Iterable<string>) {
    let s = "";
    for (const x of v) {
        s += x;
    }

    return s;
}

function main() {
    assert(sum([1.5, 2]) == 3.5);
    const a: number[] = [1, 2, 3];
    assert(sum(a) == 6);
    const b = [4, 5];
    assert(sum(b) == 9);
    assert(cat("abc") == "abc");
    const names: string[] = ["x", "y"];
    assert(cat(names) == "xy");
    print("done.");
}
