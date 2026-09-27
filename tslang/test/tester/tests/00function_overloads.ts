// Overload signatures: the body-less signatures are types only, and the implementation that
// follows is the one function. Registering a signature as a function crashed the compiler (#343).

function f(a: number): number;
function f(a: number, b: number): number;
function f(a: number, b?: number) {
    return b === undefined ? a : a + b;
}

function g(s: string): string;
function g(n: number): string;
function g(x: string | number): string {
    return typeof x === "string" ? "s" : "n";
}

// signatures that are never called still compile
function unused(a: number): number;
function unused(a: number, b: number): number;
function unused(a: number, b?: number) {
    return a;
}

namespace NS {
    export function h(a: number): number;
    export function h(a: number, b: number): number;
    export function h(a: number, b?: number) {
        return b === undefined ? a : a * b;
    }
}

function outer() {
    function inner(a: number): number;
    function inner(a: number, b: number): number;
    function inner(a: number, b?: number) {
        return b === undefined ? -a : a - b;
    }

    return inner(1) + inner(5, 2);
}

class C {
    k: number;

    constructor();
    constructor(k: number);
    constructor(k?: number) {
        this.k = k === undefined ? 10 : k;
    }

    m(a: number): number;
    m(a: number, b: number): number;
    m(a: number, b?: number) {
        return (b === undefined ? a : a + b) * this.k;
    }

    static s(a: number): number;
    static s(a: number, b: number): number;
    static s(a: number, b?: number) {
        return b === undefined ? a : a - b;
    }
}

abstract class Base {
    abstract get(): number;

    twice() {
        return this.get() * 2;
    }
}

class Impl extends Base {
    get(): number;
    get(scale: number): number;
    get(scale?: number) {
        return scale === undefined ? 3 : 3 * scale;
    }
}

function main() {
    assert(f(1) == 1);
    assert(f(1, 2) == 3);
    assert(g("a") == "s");
    assert(g(1) == "n");
    assert(NS.h(5) == 5);
    assert(NS.h(5, 2) == 10);
    assert(outer() == 2);

    const c = new C();
    const c2 = new C(2);
    assert(c.m(1) == 10);
    assert(c2.m(1, 2) == 6);
    assert(C.s(5) == 5);
    assert(C.s(5, 2) == 3);

    const b: Base = new Impl();
    assert(b.twice() == 6);

    print("done.");
}
