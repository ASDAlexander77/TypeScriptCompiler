// A generic function given as an argument infers its own type parameters from the parameter it is given to
// (`map(xs, identity)`, `apply(identity)`); a mapped type over `keyof T` of a T not known yet keeps its key
// (`{ [K in keyof T]: T[K] }`); `typeof this.x` in a method's signature.
function identity<T>(x: T) { return x; }
function singleton<T>(x: T) { return [x]; }
function apply(f: (x: number) => number) { return f(20); }
function map<T, U>(items: T[], f: (x: T) => U): U[] {
    const r: U[] = [];
    for (const i of items) r.push(f(i));
    return r;
}

function pick<T>(o: { [K in keyof T]: T[K] }, k: keyof T) { return 1; }

class A {
    x = 1;
    add(y: typeof this.x) { return this.x + y; }
}

function main() {
    assert(apply(identity) == 20);

    const xs = [1, 2, 3];
    const m = map(xs, identity);
    assert(m.length == 3 && m[2] == 3);

    const ss = map(xs, singleton);
    assert(ss.length == 3 && ss[1][0] == 2);

    assert(new A().add(2) == 3);

    print("done.");
}
