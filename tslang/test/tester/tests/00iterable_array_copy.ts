// An array given where an iterator interface is wanted (`sum(a)` of `sum(v: Iterable<number>)`): a generator over
// its elements. Under -mm=own it iterates a copy of the array, which it owns (USER DECISION 2026-11-21): one
// borrowing the array would hold the borrow for as long as it lives, which -mm=own can't say - it was "a value takes
// a second reference". The other models iterate the array itself. (-mm=own: a for...of over such an interface of
// strings is still refused, and an array of objects can't be copied there.)
type IterateResult<T> = { value: T, done: boolean };
interface Iterable<T> {
    next: () => IterateResult<T>;
}

function sum(v: Iterable<number>) {
    let s = 0.0;
    for (const x of v) {
        s += x;
    }

    return s;
}

function main() {
    for (let round = 0; round < 1000; round++) {
        assert(sum([1.5, 2]) == 3.5);
        const a: number[] = [1, 2, 3];
        assert(sum(a) == 6 && a.length == 3);
        const b = [4, 5];
        assert(sum(b) == 9);
    }

    print("done.");
}
