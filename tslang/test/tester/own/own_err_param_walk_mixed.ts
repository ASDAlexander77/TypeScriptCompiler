// A parameter given a new object on one path walks down on the other: it owns what it holds, so the
// caller gives it up, and may not use it after the call.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

function f(n: Node | null, b: boolean) {
    while (n) {
        if (b) n = new Node(9);
        else n = n.next;
        b = false;
    }

    return 1;
}

function main() { const a = new Node(1); a.next = new Node(2); f(a, true); print(a.v); }
