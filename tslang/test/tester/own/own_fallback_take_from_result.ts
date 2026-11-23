// -mm=own --own-fallback=count: a field taken out of a call's own result that another owner also
// holds. `mk` keeps the box in a global and returns it, which own refuses ('b' is used after its
// value was moved); with the flag the box is counted, and the result has an extra owner. The callers
// take the box's node out of the result (spec 63/64): the take empties the field only when the
// result's release frees the box (ts.TakeOutOfField), so the global still reads the node.
class Node { constructor(public v: number) {} }
class Box { node: Node | null = null; }

let last: Box | null = null;

function mk(v: number) {
    const b = new Box();
    b.node = new Node(v);
    last = b;
    return b;
}

function takeNamed(v: number) {
    const r = mk(v);
    return r.node;
}

function takeTemp(v: number) {
    return mk(v).node;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const n = takeNamed(i);
        sum += n!.v;
        sum += last!.node ? last!.node!.v : -1000000;
        const n2 = takeTemp(i);
        sum += n2!.v;
        sum += last!.node ? last!.node!.v : -1000000;
    }
    print(sum);
    print("done.");
}
