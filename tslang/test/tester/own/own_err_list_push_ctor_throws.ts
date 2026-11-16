// The head given to a constructor that may throw: a throw caught by the caller would find the field
// given up, where the other models keep the list. Only a call that never throws takes a field this way.
class Node {
    constructor(public v: number, public next: Node | null) {
        if (v < 0) throw "negative";
    }
}

class List {
    head: Node | null = null;
    push(v: number) { this.head = new Node(v, this.head); }
}

function main() {
    const l = new List();
    l.push(1);
    try { l.push(-1); } catch (e) { print("caught"); }
    print(l.head ? l.head.v : 0);
}
