// a `for` condition that is a bare truthiness test of an optional value (`for (let n = head; n; n = n.next)`)
// went to ts.Condition unconverted and failed module verification; `while` and `if` made it a boolean first
class Node {
    constructor(public value: number, public next: Node | undefined) {}
}

class NullNode {
    constructor(public value: number, public next: NullNode | null) {}
}

function sumFor(head: Node | undefined) {
    let sum: number = 0;
    for (let node = head; node; node = node.next) {
        sum += node.value;
    }

    return sum;
}

function sumWhile(head: Node | undefined) {
    let sum: number = 0;
    let node = head;
    while (node) {
        sum += node.value;
        node = node.next;
    }

    return sum;
}

function sumIf(head: Node | undefined) {
    let sum: number = 0;
    if (head) {
        sum += head.value;
    }

    return sum;
}

function sumForNull(head: NullNode | null) {
    let sum: number = 0;
    for (let node = head; node; node = node.next) {
        sum += node.value;
    }

    return sum;
}

function countForString(s: string | undefined) {
    let count = 0;
    for (let t = s; t; t = undefined) {
        count++;
    }

    return count;
}

function main() {
    let head: Node | undefined = new Node(1, new Node(2, undefined));
    let sum: number = 0;
    for (let node = head; node; node = node.next) {
        sum += node.value;
    }

    print(sum);
    assert(sum == 3, "for over an optional class, at the call site");

    assert(sumFor(head) == 3, "for over an optional class");
    assert(sumFor(new Node(5, undefined)) == 5, "for over one node");
    assert(sumFor(undefined) == 0, "for over undefined");

    assert(sumWhile(head) == 3, "while over an optional class");
    assert(sumWhile(undefined) == 0, "while over undefined");

    assert(sumIf(head) == 1, "if on an optional class");
    assert(sumIf(undefined) == 0, "if on undefined");

    assert(sumForNull(new NullNode(4, new NullNode(6, null))) == 10, "for over a nullable class");
    assert(sumForNull(null) == 0, "for over null");

    assert(countForString("a") == 1, "for over an optional string");
    assert(countForString(undefined) == 0, "for over undefined string");

    print("done.");
}
