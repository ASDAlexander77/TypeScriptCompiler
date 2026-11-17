class Node { constructor(public value: number, public next: Node | undefined) {} }
function f() {
    let head: Node = new Node(1, new Node(2, new Node(3, undefined))); let node: Node | undefined = head;
    while (node) { if (node.value == 2) head.next = undefined; print(node.value); node = node.next; }
}
function main() { f(); print("done."); }
