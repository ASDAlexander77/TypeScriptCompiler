// TSLANG 218: a rest parameter of a generic function whose element type names no type parameter (`...a: any[]`)
// was packed twice - `g(1, 3, 4)` saw one argument, an array, and `new c(...a)` of `make<T>(c, ...a: any[])` threw
// "Can't cast from any type" (every model); calls through a function value returning a string, under own, kept
// a result nobody gave back; an integer `number` became text by the slow search for its shortest digits.

function g<T>(x: T, ...a: any[]) {
    return a.length;
}

function gn<T>(x: T, ...a: number[]) {
    let s: number = 0;
    for (const v of a) s += v;
    return s;
}

function gt<T>(x: T, ...a: T[]) {
    return a.length;
}

// the tests compile without the default library: no Array.join
function joined(a: string[], separator: string) {
    let text = "";
    for (let i = 0; i < a.length; i++) text += (i > 0 ? separator : "") + a[i];
    return text;
}

function gs<T>(x: T, ...a: string[]) {
    return joined(a, ",");
}

class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function make<T>(c: new (...args: any[]) => T, ...a: any[]): T {
    return new c(...a);
}

function assert(cond: boolean, msg: string) {
    if (!cond) throw msg;
}

function main() {
    assert(g(1, 3, 4) == 2, "g(1, 3, 4)");
    assert(g(1) == 0, "g(1)");
    assert(g(1, "x") == 1, "g(1, 'x')");
    assert(gn(1, 3, 4.5) == 7.5, "gn");
    assert(gn("a") == 0, "gn('a')");
    assert(gt(1, 3, 4) == 2, "gt");
    assert(gs(1, "p", "q") == "p,q", "gs");

    const arr: any[] = [1, 2, 3];
    assert(g(0, ...arr) == 3, "g(0, ...arr)");

    const p = make(P, 3.5);
    assert(p.x == 3.5, "make(P, 3.5)");

    // calls through a function value, each returning a new string
    let total = 0;
    for (let k = 0; k < 1000; k++) {
        const fs: (() => string)[] = [];
        const s = "a" + k;
        fs.push(() => s);
        const n: number = k;
        fs.push(() => "b" + n);
        total += fs[0]().length + fs[1]().length;
    }

    assert(total == 2 * (10 * 2 + 90 * 3 + 900 * 4), "closures");

    // whole numbers through the quick path, the rest as before
    const texts: string[] = [];
    const values: number[] = [0, -0, 7, -1, 1e15, 1e21, 9007199254740991, 9007199254740992, -9007199254740991, 2 ** 63,
        0.5, 0.1 + 0.2, 1e-7, 4294967295];
    for (const v of values) texts.push("" + v);
    assert(joined(texts, " ") == "0 0 7 -1 1000000000000000 1e+21 9007199254740991 9007199254740992 -9007199254740991 " +
        "9223372036854776000 0.5 0.30000000000000004 1e-7 4294967295", joined(texts, " "));

    print("done.");
}
