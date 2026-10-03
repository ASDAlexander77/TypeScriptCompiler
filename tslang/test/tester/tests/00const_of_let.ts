// `const b = a`, with `a` a `let` (or a parameter) holding a heap value, keeps that value alive
// after `a` is reassigned (#454). Under rc the const was folded into a read of `a`'s slot and took
// no reference, so reassigning `a` freed what `b` still pointed at. Each `b` is read after
// allocations of the same size have reused anything freed.
class Node {
    v = 0;
    name = "";
}

function churn() {
    let keep: Node[] = [];
    for (let i = 0; i < 1000; i++) {
        const n = new Node();
        n.v = -i;
        n.name = "junk" + i;
        keep.push(n);
    }

    return keep.length;
}

function fromParam(p: Node) {
    const kept = p;
    p = new Node();
    assert(churn() == 1000);
    return kept.name;
}

function main() {
    let a = new Node();
    a.name = "first" + 1;
    const b = a;
    a = new Node();
    assert(churn() == 1000);
    assert(b.name == "first1", "a const of a let keeps the first node");

    let s = "text" + 2;
    const t = s;
    s = "other" + 3;
    assert(churn() == 1000);
    assert(t == "text2", "a const of a let keeps the first string");

    let arr = [1, 2, 3];
    const kept = arr;
    arr = [4];
    assert(churn() == 1000);
    assert(kept.length == 3 && kept[2] == 3, "a const of a let keeps the first array");

    const n = new Node();
    n.name = "param" + 4;
    assert(fromParam(n) == "param4", "a const of a parameter keeps its node");

    for (let i = 0; i < 1000; i++) {
        let x = new Node();
        const y = x;
        x = new Node();
        y.v = i;
    }

    print("done.");
}
