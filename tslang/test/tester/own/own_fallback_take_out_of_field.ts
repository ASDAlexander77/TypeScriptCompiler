// -mm=own --own-fallback=count: a node taken out of a list (`this.head = this.head.next`) while it
// has a second owner. `link` puts the node in the list and in a holder, which own refuses (a second
// owner); with the flag the node is counted. The pop hands the node's `next` to the list's head
// right before the list lets go of the node (ts.TakeOutOfField). The node has an extra owner, so
// that release does not free it: its `next` stays, and the holder still reads it.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;

    pop() {
        if (this.head) {
            this.head = this.head.next;
        }
    }
}

class Holder {
    node: Node | null = null;
}

function link(l: List, g: Holder, n: Node) {
    l.head = n;
    g.node = n;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const l = new List();
        const g = new Holder();
        const n = new Node(i);
        n.next = new Node(i + 1);
        link(l, g, n);
        l.pop();
        // the list holds the second node, the holder the first, which still links to it
        sum += l.head!.v + g.node!.next!.v;
        l.pop();
        sum += g.node!.next!.v;
    }

    print(sum);
    print("done.");
}
