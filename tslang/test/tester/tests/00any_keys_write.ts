// `x.name`, and `x[k] = v` (`+=`, `++`) of a value of type `any`: read and written at run time through the
// adapter made for the type of the value when it was boxed (TYPE_DESCR_KEYS, ANY_KEYS_OP_SET). A write is
// a TypeError for a key the type has not got (none is added), a method, an accessor with no setter, a
// tuple's key (the `any` holds a copy), and a value not of a number, string or boolean key's kind.
let gets = 0;

class P {
    constructor(public v: number) {
    }
}

class Q {
    constructor(public w: string) {
    }
}

class C {
    a = 1;
    s = "x";
    b = true;
    p = new P(4);

    get g() { return this.a * 10; }

    get h() { gets++; return this.a; }
    set h(v: number) { this.a = v; }

    m() { return 1; }
}

interface I {
    n: number;
    o?: string;
}

function throws(f: () => void) {
    try {
        f();
    } catch (e) {
        return true;
    }

    return false;
}

function main() {
    const c = new C();
    const x: any = c;

    // reads by name
    assert(x.a == 1 && x.s == "x" && x.g == 10 && x.b, "read by name");
    assert(typeof x.zz == "undefined", "a key it has not got");

    // writes, by name and by key
    x.a = 5;
    x["s"] = "y" + c.a;
    assert(c.a == 5 && c.s == "y5" && x.g == 50, "written");

    // a setter, which a plain write runs without the getter
    x.h = 7;
    x["h"] = 8;
    assert(c.a == 8 && gets == 0, "setter only");
    x.h += 1;
    assert(c.a == 9 && gets == 1, "compound: one read, one write");

    x.a += 2;
    x.a++;
    assert(c.a == 12, "compound and increment");

    const r = (x.s = "z");
    assert(r == "z" && c.s == "z", "an assignment is its value");

    x.p = new P(8);
    assert(c.p.v == 8, "an object");

    // what is refused
    assert(throws(() => { x.g = 1; }), "no setter");
    assert(throws(() => { x.m = 1; }), "a method");
    assert(throws(() => { x.zz = 1; }), "a key it has not got");
    assert(throws(() => { x.a = "hi"; }), "a string for a number");
    assert(throws(() => { x.s = 5; }), "a number for a string");
    assert(throws(() => { x.b = 1; }), "a number for a boolean");
    assert(throws(() => { x.p = 5; }), "a number for an object");
    assert(throws(() => { x.p = new Q("q"); }), "another class");
    assert(c.a == 12 && c.s == "z" && c.b && c.p.v == 8, "unchanged by what was refused");

    // an interface
    const i: I = { n: 1, o: "a" };
    const y: any = i;
    y.o = "set";
    y.n = 9;
    assert(i.n == 9 && i.o == "set" && "o" in y, "an interface");

    // an object made without an optional field has no place for it
    const j: I = { n: 1 };
    assert(throws(() => { j.o = "a"; }), "no place, typed");
    const yj: any = j;
    assert(throws(() => { yj.o = "a"; }), "no place, through any");

    // an array: in range, or one past its end
    const arr = [1, 2, 3];
    const z: any = arr;
    z[1] = 20;
    z[3] = 40;
    assert(arr.length == 4 && arr[1] == 20 && arr[3] == 40 && z.length == 4, "an array");
    assert(throws(() => { z.length = 1; }), "length");
    assert(throws(() => { z[10] = 1; }), "far past its end");
    assert(throws(() => { z[0] = "a"; }), "a string for a number element");

    const strs: any = ["a"];
    strs[1] = "b" + arr.length;
    assert(strs.length == 2 && strs[1] == "b4", "a string array");

    // a tuple and an object literal are values: the `any` holds a copy
    const t: [number, string] = [1, "a"];
    const w: any = t;
    assert(throws(() => { w[0] = 11; }), "a tuple");
    const lit: any = { a: 1 };
    assert(lit.a == 1, "an object literal's key");
    assert(throws(() => { lit.a = 2; }), "an object literal");

    // a string, null
    const s: any = "str";
    assert(s.length == 3 && s[1] == "t", "a string");
    assert(throws(() => { s[0] = "q"; }), "a string can't be changed");
    const nl: any = null;
    assert(throws(() => { nl.a = 1; }), "null");

    // in a loop, with values made at run time
    for (let k = 0; k < 100; k++) {
        x.s = "s" + k;
        x["p"] = new P(k);
        strs[0] = "v" + k;
    }

    assert(c.s == "s99" && c.p.v == 99 && strs[0] == "v99", "in a loop");

    print("done.");
}
