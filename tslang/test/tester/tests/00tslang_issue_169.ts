// TSLANG 169: an `unknown` owns the box it holds, as `any` does. A value stored into one after its declaration was
// not kept (-mm=rc read it freed, -mm=own leaked the box it replaced); a class made through a construct signature
// with no result type (`new(x: number)`, which returns `unknown`) is boxed into it.

let g: unknown;

function setIt(i: number) {
    g = "value" + i;
}

function churn() {
    let s = "";
    for (let i = 0; i < 1000; i++) s = "zzzzzzzzzzzzzz" + i;
    return s.length;
}

class B {
    constructor(public x: number) {}
}

interface A {
    f: { new(x: number) };
}

function made(i: number): unknown {
    return "m" + i;
}

function main() {
    setIt(42);
    churn();
    let text = "";
    if (typeof g === "string") text = g;
    assert(text == "value42");

    // replaced again and again: each old box is given back
    let total = 0;
    for (let i = 0; i < 10000; i++) {
        setIt(i);
        if (typeof g === "string") total += g.length;
    }
    assert(total == 88890);

    let u: unknown = 1.5;
    for (let i = 0; i < 3; i++) u = "local" + i;
    churn();
    if (typeof u === "string") text = u;
    assert(text == "local2");

    const m = made(7);
    if (typeof m === "string") text = m;
    assert(text == "m7");

    const a: A = { f: B };
    const r = new a.f(3);
    const b = <B>r;
    assert(b.x == 3);
    new a.f(4);

    print("done.");
}