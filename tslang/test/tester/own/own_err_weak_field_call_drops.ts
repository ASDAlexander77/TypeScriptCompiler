// own spec 93: a @weak read may not be the receiver of a call that may free objects.
class Node {
    @weak parent: Node | null = null;
    children: Node[] = [];
    v = 1;
    removeLast() { this.children.pop(); }
}
function detach(c: Node) { if (c.parent) c.parent.removeLast(); }
function main() { const p = new Node(); const c = new Node(); c.parent = p; detach(c); print(p.v); }
