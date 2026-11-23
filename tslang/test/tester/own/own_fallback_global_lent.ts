// -mm=own --own-fallback=count: a field of a global's object lent to a call that replaces the
// global. The call may free what it was lent; with the flag the field's node is counted for the
// call (a retain before it, a release after).
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

let current = new Holder();

function replace(n: Node) {
    current = new Holder();
    return n.v + n.s.length;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        current.node = new Node(i, "a" + i);
        sum += replace(current.node!);
    }

    print(sum);
    print("done.");
}
