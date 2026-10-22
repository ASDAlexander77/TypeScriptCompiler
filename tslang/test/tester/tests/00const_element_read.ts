// A `const` read out of an element keeps what it read alive when the element is given a new value.
// Folded, the const was the read itself and held no reference of its own: under rc the overwrite
// freed the block it still read. Allocations of the same size reuse anything freed before each
// read. A destructuring past the end of an array gives the element type's default (null for a
// reference), not whatever follows the array's data.
class C {
    constructor(public x: number) {}
}

class Junk {
    a = -1;
    b = -2;
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

function element(i: number) {
    const arr: C[] = [new C(0), new C(0)];
    arr[0] = new C(i);
    const c = arr[0];
    arr[0] = new C(9);
    churn();
    return c.x == i && arr[0].x == 9;
}

function destructured(i: number) {
    const arr: C[] = [new C(0), new C(i)];
    const [a, c] = arr;
    arr[1] = new C(9);
    churn();
    return c.x == i && a.x == 0 && arr[1].x == 9;
}

function stringElement(i: number) {
    const words: string[] = ["a", "b"];
    words[0] = "w" + i;
    const w = words[0];
    words[0] = "x" + i;
    churn();
    return w == "w" + i;
}

function pastTheEnd() {
    const pair: C[] = [new C(1), new C(2)];
    const [a, b, c] = pair;
    let [d, e, f] = pair;
    assert(a.x == 1 && b.x == 2 && d.x == 1 && e.x == 2, "in bounds");
    assert(c == null && f == null, "a class element past the end is null");

    const words: string[] = ["a", "b"];
    const [w0, w1, w2] = words;
    assert(w0 == "a" && w1 == "b" && w2 == null, "a string element past the end is null");

    const nums: number[] = [1, 2];
    const [n0, n1, n2] = nums;
    assert(n0 == 1 && n1 == 2 && n2 == 0, "a number element past the end is 0");

    const [k0, k1] = [7];
    assert(k0 == 7 && k1 == 0, "past the end of an array literal");
}

function main() {
    for (let i = 1; i < 4; i++) {
        assert(element(i), "an element's overwrite frees what a const read of it holds");
        assert(destructured(i), "an element's overwrite frees what a const destructured from it holds");
        assert(stringElement(i), "a string element");
    }

    pastTheEnd();
    print("done.");
}
