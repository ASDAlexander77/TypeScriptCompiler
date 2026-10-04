// Arrays inside an inferred tuple or record literal can be changed (#479). Such a literal is
// folded to a constant, and its arrays used to be headers in constant global data over the
// literal's constant data: push reallocated a global, pop and `length =` wrote one.
const gst = [1, [6, 7]];
const go = { name: "g", items: ["a", "b"] };
const ga = [6, 7];

class Holder {
    n: number;
    items: number[];
}

function grow(t: [number, number[]]) {
    t[1].push(9);
    return t[1].length;
}

function fresh() {
    const ft = [1, [2]];
    return ft[1];
}

function main() {
    const lt = [1, [6, 7]];
    const x = lt[1];
    const p = x.pop();
    assert(p == 7 && x.length == 1 && x[0] == 6, "pop");
    x.length = 0;
    assert(x.length == 0 && lt[1].length == 0, "length =, seen through the tuple");
    x.push(8);
    assert(lt[1].length == 1 && lt[1][0] == 8, "push, seen through the tuple");

    gst[1].push(8);
    assert(gst[1].length == 3 && gst[1][2] == 8, "module-level tuple");
    go.items.pop();
    assert(go.items.length == 1 && go.items[0] == "a", "module-level record");

    // a global holds a count of its own under rc: a local that reads it and lets go
    // does not free it
    for (let i = 0; i < 3; i++) {
        const items = go.items;
        const inner = gst[1];
        const whole = ga;
        assert(items.length == 1 && inner.length == 3 && whole.length == 2, "a global read through a local, repeatedly");
    }

    const o = { items: [1, 2] };
    o.items.push(3);
    assert(o.items.length == 3, "record literal");

    let n = 5;
    const mo = { inner: { items: [1] }, n: n };
    mo.inner.items.push(2);
    assert(mo.inner.items.length == 2, "record in a record");

    let nt = [1, [2, [3, 4]]];
    nt[1][1].push(5);
    assert(nt[1][1].length == 3, "tuple in a tuple");

    const at = [[1, [2]], [3, [4, 5]]];
    at[1][1].push(6);
    assert(at[1][1].length == 3 && at[0][1].length == 1, "array of tuples");

    assert(grow([1, [2, 3]]) == 3, "a literal passed to a tuple parameter");

    const h: Holder = { n: 1, items: [1, 2] };
    h.items.push(3);
    assert(h.items.length == 3, "a record literal cast to a class");

    const a = fresh();
    a[0] = 9;
    assert(fresh()[0] == 2, "each evaluation makes a new array");

    for (let i = 0; i < 3; i++) {
        const lst = [1, [6, 7]];
        lst[1].push(i);
        assert(lst[1].length == 3, "a new array on each pass");
    }

    print("done.");
}
