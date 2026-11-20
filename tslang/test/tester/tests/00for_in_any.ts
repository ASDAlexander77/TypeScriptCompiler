// `for (k in x)` and Object.keys(x) with x of type any: its keys adapter lists the keys at run time
// (ANY_KEYS_OP_KEYS) - a class object's public fields (its actual class's, base class first; an optional one
// only while set; no accessor, no method, as JavaScript has them), an object's or a tuple's fields, an
// array's or a string's indices, and none for a number, null or undefined.
class Base {
    b = 1;
    m() {
        return 0;
    }
}

class C extends Base {
    x = 2;
    o?: number;
    get g() {
        return 3;
    }
}

function keysOf(v: any) {
    let s = "";
    for (const k in v) s += k + ",";
    return s;
}

function main() {
    assert(keysOf(new C()) == "b,x,", "a class object's fields, base first");
    const c = new C();
    c.o = 5;
    assert(keysOf(c) == "b,x,o,", "an optional field once set");
    const asBase: Base = new C();
    assert(keysOf(asBase) == "b,x,", "the object's actual class");
    assert(keysOf({ p: 1, q: "a" }) == "p,q,", "an object literal");
    assert(keysOf([10, 20, 30]) == "0,1,2,", "an array's indices");
    assert(keysOf("hey") == "0,1,2,", "a string's indices");
    assert(keysOf(5) == "" && keysOf(null) == "" && keysOf(undefined) == "", "no keys");

    const a: any = { u: 1, w: 2 };
    let sum = 0;
    for (const k in a) sum += <number>a[k];
    assert(sum == 3, "the values through the keys");

    let turns = 0;
    for (const k in a) {
        if (k == "w") continue;
        turns++;
    }
    assert(turns == 1, "continue");

    let last = "";
    for (last in a) {
    }
    assert(last == "w", "a variable declared before the loop");

    const ks = Object.keys(a);
    assert(ks.length == 2 && ks[0] == "u" && ks[1] == "w", "Object.keys of an any");

    print("done.");
}
