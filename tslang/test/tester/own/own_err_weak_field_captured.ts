// own spec 93: a @weak read borrows its object and cannot be captured.
class Node { @weak parent: Node | null = null; v = 1; }
function later(c: Node) { const p = c.parent; return () => p ? p.v : 0; }
function main() { const p = new Node(); const c = new Node(); c.parent = p; const f = later(c); print(f(), p.v); }
