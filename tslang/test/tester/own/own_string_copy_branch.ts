// A string stored into a field on one branch only, and used after: the copy is made in the store's
// block. Made at rc's birth retain after the producer instead, it would have no owner on the path
// that skips the store, a leak no run sees; test-own-string-copy-in-store-block reads the IR.
class H {
    text = "";
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function stored(c: boolean) {
    const h = new H();
    const s = "some" + 3;
    if (c) {
        h.text = s;
    }
    print(s);
    return h;
}

function main() {
    const both = stored(true);
    const neither = stored(false);
    assert(churn() == 1000);
    assert(both.text == "some3" && neither.text == "");

    print("done.");
}
