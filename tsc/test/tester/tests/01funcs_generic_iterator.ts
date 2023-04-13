interface Iterable<T> {
    next: () => { value: T, done: boolean }
};

type ElementTypeOf<T> = T extends unknown[] ? T[number] : T extends Iterable<infer E> ? E : never

function* Filter<A extends unknown[] | Iterable<unknown>>(a: A, f: (i: ElementTypeOf<A>) => boolean): Iterable<ElementTypeOf<A>> {
    for (const v of a) if (f(v)) yield v;
}

function* Map<A extends unknown[] | Iterable<unknown>, R>(a: A, f: (i: ElementTypeOf<A>) => R): Iterable<R> {
    for (const v of a) yield f(v);
}

function ForEach<A extends unknown[] | Iterable<unknown>>(a: A, f: (i: ElementTypeOf<A>) => void) {
    for (const v of a) f(v);
}

function main() {
    let count = 0;

    [1, 2, 3]
        .Filter((i) => i % 2)
        .Map((i) => { count++; return i + 1; })
        .Map((j) => { count++; return "asd: " + j + 10; })
        .ForEach((e) => print(e));

    assert(count == 4);

    print("done.");
}