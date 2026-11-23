// -mm=own --own-fallback=count still rejects: one call is given a counted field read and an array
// literal twice, the callee keeping the array through one parameter and reading the other. The
// field read is counted; the array is not (rc makes a literal at count zero), and its error -
// 'a' is used here after its value was moved - is reported at the same call. An error is a counted
// value's note only when it is about that value, never because of where it is.
class Node {
    constructor(public v: number) {}
}

class Holder {
    node: Node | null = null;
    arr: Node[] | null = null;
}

let current = new Holder();
let keepA = new Holder();

function replace(n: Node, a: Node[], b: Node[]) {
    current = new Holder();
    keepA.arr = a;
    keepA = new Holder();
    return n.v + b.length;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 10; i++) {
        current.node = new Node(i);
        const a = [new Node(i)];
        sum += replace(current.node!, a, a);
    }

    print(sum);
    print("done.");
}
