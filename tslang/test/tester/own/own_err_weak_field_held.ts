// own spec 93: a @weak read held across a call that may free its object is refused (read it again).
class Node { @weak parent: Node | null = null; v = 1; }
class Holder { n: Node | null = null; }
function dropAll(h: Holder) { h.n = null; }
function main() { const h = new Holder(); const c = new Node(); h.n = new Node(); c.parent = h.n; const p = c.parent; dropAll(h); print(p ? p.v : 0); }
