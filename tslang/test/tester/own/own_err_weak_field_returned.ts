// own spec 93: a @weak read borrows its object and cannot be returned.
class Node { @weak parent: Node | null = null; v = 1; }
function up(c: Node): Node | null { return c.parent; }
function main() { const p = new Node(); const c = new Node(); c.parent = p; const u = up(c); print(u ? u.v : 0, p.v); }
