interface Iterable<T> {
    next: () => { value: T, done: boolean }
};

type ElementTypeOf<T> = T extends unknown[] ? T[number] : T extends Iterable<infer E> ? E : never 

function* Map<A extends unknown[] | Iterable<unknown>, R>(a: A, f: (i: ElementTypeOf<A>) => R) : Iterable<R> {
    for (const v of a) yield f(v);
}

function main() {
    let count = 0;

    for (const v of [1, 2, 3].Map((i) => { count++; return i + 1; }).Map((j) => { count++; return j + 10; })) {
        print(v);
    }

    assert(count == 6);

    print("done.");
}