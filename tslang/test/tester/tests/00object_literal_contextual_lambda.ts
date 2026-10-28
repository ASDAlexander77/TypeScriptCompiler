// The untyped parameters of an object literal's function members take their types from the literal's
// receiver (docs/not_impl/not_impl28.ts): a typed variable or parameter, or, in a generic call, the
// parameter's type with its type parameters inferred first from the rest of the call - the other arguments,
// and the literal's other members (`read: () => s` gives T for `write: value => ...`).

interface Computed<T> {
    read(): T;
    write(value: T);
}

function foo<T>(x: Computed<T>) {
    x.write(x.read());
}

enum E1 { X }

function f1<T, U>(a: { w: (x: T) => U; r: () => T; }, b: T): U {
    return a.w(b);
}

interface Adder {
    add(x: number): number;
}

function useAdder(a: Adder) {
    return a.add(2);
}

function apply(o: { w: (x: number) => number }) {
    return o.w(3);
}

function main() {
    let s: string = "a";

    foo({
        read: () => s,
        write: value => s = value + "!"
    });
    assert(s == "a!");

    foo({
        write: value => s = value + "?",
        read: () => s
    });
    assert(s == "a!?");

    let v1: number = f1({ w: x => x, r: () => 0 }, 0);
    assert(v1 == 0);
    v1 = f1({ w: x => x + 1, r: () => 0 }, 5);
    assert(v1 == 6);

    let v2: E1 = f1({ w: x => x, r: () => E1.X }, E1.X);
    assert(v2 == E1.X);

    // not generic: the receiver's member types the parameters
    const o: { w: (x: number) => number } = { w: x => x * 2 };
    assert(o.w(4) == 8);
    assert(apply({ w: x => x + 1 }) == 4);
    assert(useAdder({ add: x => x + 10 }) == 12);

    print("done.");
}
