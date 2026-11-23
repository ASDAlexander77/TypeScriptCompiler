// -mm=own --own-fallback=count: a local `new Node` moved into `best`, which a later loop gives other
// nodes before the local is read again. Own refuses it ('v' is used after its value was moved);
// with the flag the node is counted. Its kept retain is an escape: it is never put on the stack
// (StackPromotionPass), which would make it immortal and discard its counts.
class Node {
    constructor(public x: number) {}
}

function pick(n: number) {
    let best: Node = new Node(-1);
    let v = new Node(n); // HEAP_AT=11:13 (CMakeLists.txt) names this `new`: keep it on line 11
    best = v;
    for (let i = 0; i < 3; i++) {
        best = new Node(n + i);
    }

    return v.x + best.x;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        sum += pick(i);
    }

    print(sum);
    print("done.");
}
