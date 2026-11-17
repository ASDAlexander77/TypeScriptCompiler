// `for (const x of t)` over a tuple, and `t[i]` at an index known at run time only: each element, read
// where it is, as the union of the element types. It was "Element access with a non-constant index is
// not supported on this type" (and before #884, "Can't resolve property 'length'").
class A {
    name() {
        return "A";
    }
}

class B {
    name() {
        return "B";
    }
}

function main() {
    // a literal of mixed types
    let numbers = 0;
    let strings = 0;
    let booleans = 0;
    for (const x of [1, "a", true, 2]) {
        if (typeof x == "string") strings++;
        else if (typeof x == "boolean") booleans++;
        else numbers++;
    }

    assert(numbers == 2 && strings == 1 && booleans == 1, "mixed literal");

    // a tuple kept in a local, with a string made at run time
    let t: [number, string] = [3, "b" + 1];
    let text = "";
    for (const x of t) {
        text += x;
    }

    assert(text == "3b1", "local tuple");

    // read at an index, in order and from the end
    let reversed = "";
    for (let i = t.length - 1; i >= 0; i--) {
        reversed += t[i];
    }

    assert(reversed == "b13", "index read");

    // break and continue
    let seen = "";
    for (const x of [1, "a", "b"]) {
        if (x == 1) continue;
        seen += x;
        break;
    }

    assert(seen == "a", "continue and break");

    // classes, and objects kept in a local
    let made = "";
    for (const c of [A, B]) {
        made += new c().name();
    }

    assert(made == "AB", "classes");

    const objects = [new A(), new B()];
    let names = "";
    for (const o of objects) {
        names += o.name();
    }

    assert(names == "AB", "objects");

    print("done.");
}
