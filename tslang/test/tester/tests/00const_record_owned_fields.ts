// A `const` record literal built at run time keeps, and gives back, what each field holds (#485).
// Folded, nothing counted an array or tuple built in the literal: under rc and own every one
// leaked. The const now has storage, as a `let` has; const-record-release.cmake checks under rc
// that each record's slot is released. Here each kind of field is read after allocations of the
// same size have reused anything freed.
class C {
    v: number[] = [];
}

type Numbers = number[];

function mk(i: number): number[] {
    return [i, 2, 3];
}

function churn() {
    let keep: number[][] = [];
    for (let i = 0; i < 200; i++) keep.push([-i, -i, -i]);
    return keep.length;
}

function arrayField(i: number) {
    const o = { items: [i, 2], n: i };
    o.items.push(3);
    churn();
    return o.items.length == 3 && o.items[0] == i && o.n == i;
}

function stringField(i: number) {
    const o = { s: "a" + i, n: i };
    churn();
    return o.s == "a" + i;
}

function callField(i: number) {
    const o = { items: mk(i), n: i };
    churn();
    return o.items.length == 3 && o.items[0] == i;
}

function newField(i: number) {
    const o = { c: new C(), n: i };
    o.c.v.push(i);
    churn();
    return o.c.v.length == 1 && o.c.v[0] == i;
}

function tupleField(i: number) {
    const o = { t: [i, [i]], n: i };
    churn();
    return o.t[1].length == 1 && o.t[1][0] == i;
}

function newArrayField(i: number) {
    const o = { items: new Numbers(i % 5 + 1), n: i };
    churn();
    return o.items.length == i % 5 + 1;
}

function nestedField(i: number) {
    const o = { inner: { items: [i, 2], k: i }, n: i };
    o.inner.items.push(3);
    churn();
    return o.inner.items.length == 3 && o.inner.items[0] == i;
}

function main() {
    for (let i = 0; i < 100; i++) {
        assert(arrayField(i), "an array built in the record");
        assert(stringField(i), "a string made for the record");
        assert(callField(i), "a call's result");
        assert(newField(i), "a new instance");
        assert(tupleField(i), "a tuple holding an array");
        assert(newArrayField(i), "a new array");
        assert(nestedField(i), "a record in the record");
    }

    print("done.");
}
