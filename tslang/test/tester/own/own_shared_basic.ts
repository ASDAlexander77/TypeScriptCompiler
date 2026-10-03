// Shared<T> (spec 23): two handles to one block, `.value` read and write, `===` on handles, and a
// payload that is a string. Each handle dropped is followed by reusing freed memory before the
// other is read.
class Node {
    v = 0;
    name = "";
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    let a = new Shared(new Node());
    a.value.name = "first" + 1;
    const b = a;
    assert(a === b, "two handles to one block are equal");
    a.value.v = 5;
    assert(b.value.v == 5, "a write through one handle is seen through the other");

    const c = new Shared(new Node());
    assert(a !== c, "two blocks are not equal");

    a = c;
    assert(churn() == 1000);
    assert(b.value.v == 5 && b.value.name == "first1", "the other handle keeps the block");

    const d = b;
    d.value = new Node();
    d.value.v = 7;
    assert(b.value.v == 7, "the value replaced through one handle is seen through the other");

    const t = new Shared("text" + 2);
    const u = t;
    assert(u.value == "text2");
    t.value = "other" + 3;
    assert(churn() == 1000);
    assert(u.value == "other3");

    print("done.");
}
