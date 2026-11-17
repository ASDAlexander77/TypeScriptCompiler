// own spec 93: a @weak read may not be an argument of a call that may free objects (an override drops).
class Node {
    @weak parent: Node | null = null;
    children: Node[] = [];
    v = 1;
    visit(n: Node) { print(n.v); }
}
class Pruner extends Node {
    visit(n: Node) { n.children.pop(); }
}
function walk(by: Node, c: Node) { if (c.parent) by.visit(c.parent); }
function main() { const p = new Node(); const c = new Node(); c.parent = p; walk(new Pruner(), c); print(p.v); }
