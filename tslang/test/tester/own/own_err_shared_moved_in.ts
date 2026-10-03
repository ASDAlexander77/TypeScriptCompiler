// -mm=own rejects: a class value given to `new Shared(x)` moves into the block, so it cannot be used
// after (spec 23.1, 2.3).
class Node {
    v = 0;
}

function main() {
    const n = new Node();
    const s = new Shared(n);
    print(n.v, s.value.v);
}
