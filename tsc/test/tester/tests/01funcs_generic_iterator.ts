interface Iterable<T> {
    next: () => { value: T, done: boolean }
};

type ElementTypeOf<T> = T extends unknown[] ? T[number] : T extends Iterable<infer E> ? E : T; 

function* Map<A extends unknown[] | Iterable<unknown>, R>(a: A, f: (i: ElementTypeOf<A>) => R) {
    for (const v of a) yield f(v);
}

function* g(): Iterable<string> {
    for (let i = 0; i < 3; i++) {
        yield "i = " + i; // string is assignable to string
    }
}

function main() {
    let count = 0;

    for (const v of [1, 2, 3].Map((i) => { count++; return i + 1; })) {
        print(v);
    }

    for (const v of g().Map((i) => { count++; return i + 1; })) {
        print(v);
    }

    assert(count == 6);
    print("done.");
}