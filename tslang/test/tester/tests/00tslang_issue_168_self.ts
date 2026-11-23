// TSLANG 148 for a local: a function in a typed local's initializer reads the local. Not under -mm=own: the object
// would hold a closure that holds the object, which own refuses ("'c' is captured by a closure, but -mm=own cannot
// tell who owns its value here"), as it does for `let c: IT; c = { ... }`

interface IT {
    v: number;
    m(): number;
}

interface Named {
    name: string;
    next(): string;
}

function f(i: number) {
    const c: IT = { v: i, m() { return c.v + 1; } };
    return c.m();
}

function main() {
    let t = 0;
    for (let i = 0; i < 100; i++) t += f(i);
    assert(t == 5050);

    let k: IT = { v: 7, m() { return k.v * 2; } };
    assert(k.m() == 14);
    k.v = 8;
    assert(k.m() == 16);

    const n: Named = { name: "n1", next: () => n.name + "!" };
    assert(n.next() == "n1!");

    // a function that declares the name itself reads its own
    const v: (v: number) => number = (v) => v + 1;
    assert(v(1.5) == 2.5);

    print("done.");
}