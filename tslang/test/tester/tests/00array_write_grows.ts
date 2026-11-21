// `a[i] = v` with `i` at or past the end of the array grows it to `i + 1`, as in JavaScript
// (`a[a.length] = v` appends); the slots in between read as the element type's default. Before, the
// length stayed and the write went past the end of the block: under gc into whatever lay there, under
// rc and own a crash.
class Item {
    constructor(public n: number) {}
}

class Holder {
    items: Item[] = [];
}

// through a parameter's array, and a field's of a parameter
function append(src: number[], n: number) {
    for (let i = 0; i < n; i++) src[src.length] = i;
}

function addTo(h: Holder, n: number) {
    h.items[h.items.length] = new Item(n);
}

function main() {
    const a: number[] = [1, 2];
    a[a.length] = 3;
    assert(a.length == 3 && a[2] == 3, "append");

    // a hole: the slots between read as the default
    a[6] = 7;
    assert(a.length == 7 && a[3] == 0 && a[5] == 0 && a[6] == 7, "hole");

    // strings, built at run time, under rc and own owned by the array; one overwritten
    const s: string[] = [];
    for (let i = 0; i < 100; i++) s[i] = "v" + i;
    s[50] = "x" + 50;
    assert(s.length == 100 && s[99] == "v99" && s[50] == "x50", "strings");

    // objects, in a field
    const h = new Holder();
    for (let i = 0; i < 50; i++) h.items[h.items.length] = new Item(i);
    let total: number = 0;
    for (const item of h.items) total += item.n;
    assert(h.items.length == 50 && total == 1225, "objects in a field");

    append(a, 2);
    assert(a.length == 9 && a[8] == 1, "a parameter's array");
    addTo(h, 99);
    assert(h.items.length == 51 && h.items[50].n == 99, "a parameter's field");

    // an array of arrays, each grown
    const m: string[][] = [];
    for (let i = 0; i < 3; i++) {
        m[i] = [];
        for (let j = 0; j < 3; j++) m[i][j] = "c" + i + j;
    }
    assert(m.length == 3 && m[2].length == 3 && m[2][2] == "c22", "nested");

    // within the length: no change
    a[0] = 10;
    assert(a.length == 9 && a[0] == 10, "inside");

    print("done.");
}
