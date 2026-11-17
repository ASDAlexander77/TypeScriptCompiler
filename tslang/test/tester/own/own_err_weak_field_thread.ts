// own spec 93: a class with a @weak field pointing at it is a weak target: what may hold one may not
// be shared with another thread.
class Node { @weak parent: Node | null = null; v = 1; }
class Box { n: Node | null = null; }
function main() { const p = new Node(); const c = new Node(); c.parent = p; const b = new Box(); Shared.share(b); print(c.v, p.v); }
