// Shared<T> (spec 23) in containers: one node in two arrays, a `Shared<T> | null` field, a closure
// capturing a handle, a handle boxed into `any` and back, and a handle returned from a function.
// Each owner but one is dropped and its memory reused before the last is read.
class Node {
    v = 0;
    tag = "";
}

class Holder {
    s: Shared<Node> | null = null;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function make(v: number) {
    const s = new Shared(new Node());
    s.value.v = v;
    s.value.tag = "t" + v;
    return s;
}

function main() {
    const s = make(3);

    let left: Shared<Node>[] = [s];
    const right: Shared<Node>[] = [s];
    left = [];
    assert(churn() == 1000);
    assert(right[0].value.v == 3 && right[0].value.tag == "t3", "the other array keeps the node");

    const h = new Holder();
    h.s = right[0];
    const read = () => s.value.v;
    assert(read() == 3, "a closure reads through its handle");

    const boxed: any = h.s;
    const back = <Shared<Node>>boxed;
    assert(back === s, "a handle boxed into any comes back the same");
    assert(back.value.tag == "t3");

    h.s = null;
    assert(churn() == 1000);
    assert(read() == 3 && back.value.v == 3, "the closure and the unboxed handle keep the node");

    print("done.");
}
