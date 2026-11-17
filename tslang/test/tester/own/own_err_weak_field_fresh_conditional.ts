// own spec 93: either arm of a conditional may be a new object the @weak field would not keep
class Node { @weak parent: Node | null = null; v = 1; }
function main() { const c = new Node(); const q = new Node(); c.parent = c.v == 1 ? new Node() : q; print(c.v); }
