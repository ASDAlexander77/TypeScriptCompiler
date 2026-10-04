// A record literal with a run-time field can be assigned to a class type (#487): the instance is
// made from the record's fields, as it is from a constant record's, where the cast used to be an
// "invalid cast" error. A literal assigned to a class is built with the class's field types, so
// `[i, 2]` is the field's `number[]` itself (an `s32[]` copied into one was never freed under rc)
// and `[]` is no `any[]` the field cannot take.
class P {
    x: number;
    items: number[];
    name: string;
}

function take(p: P) {
    return p.items.length + p.name.length;
}

function make(i: number): P {
    return { x: i, items: [i, i + 1, i + 2], name: "n" + i };
}

function main() {
    let i = 7;
    const p: P = { x: 1, items: [i, 2], name: "a" };
    assert(p.items.length == 2 && p.items[0] == 7 && p.x == 1, "const of a run-time record");
    p.items.push(3);
    assert(p.items.length == 3, "its array can be changed");

    const c: P = { x: 2, items: [1, 2], name: "c" };
    assert(c.items.length == 2 && c.items[1] == 2, "a constant record");
    const e: P = { x: i, items: [], name: "e" };
    e.items.push(i);
    assert(e.items.length == 1 && e.items[0] == 7, "an empty array");

    let n: number = 5;
    let r = { x: n, items: [n, 2], name: "b" + n };
    const q: P = r;
    q.items.push(9);
    assert(r.items.length == 3 && q.name == "b5", "a record variable, sharing its array");

    assert(take({ x: 2, items: [i], name: "cc" + i }) == 4, "to a class parameter");
    const m = make(i);
    assert(m.x == 7 && m.items.length == 3 && m.name == "n7", "returned as a class");

    let total = 0;
    for (let k = 0; k < 1000; k++) {
        const z: P = { x: k, items: [k, k], name: "z" + k };
        total += z.items.length + z.name.length;
    }
    assert(total == 5890, "a fresh instance each pass");

    print("done.");
}
