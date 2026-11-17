class Node { constructor(public value: number, public next: Node | undefined) {} }
function f() {
    let head: Node | undefined = new Node(1, new Node(2, undefined)); let node = head;
    while (node) { head = undefined; print(node.value); node = node.next; }
}
function main() { f(); print("done."); }
