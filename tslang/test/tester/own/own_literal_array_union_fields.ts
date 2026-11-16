// -mm=own: an array of object literals whose fields differ in type (`c` a string in one, a number in
// the other) re-makes each as the common record, its `c` a `string | s32`. Literals and numbers seen
// as a union are still data nobody owns, so the array takes them as they are.
function make() {
    let rows = [{ a: "", b: 0, c: "" }, { a: "x", b: 3, c: 0 }];
    return rows;
}

function main() {
    let t = 0;
    for (let i = 0; i < 2000; i++) {
        const rows = make();
        t += rows.length + rows[1].b;
        assert(rows[0].c == "" && rows[1].a == "x", "rows");
    }

    let local = [{ a: "", b: 0, c: "" }, { a: "", b: 3, c: 0 }];
    assert(local.length == 2 && t == 10000, "t");
    print("done.");
}
