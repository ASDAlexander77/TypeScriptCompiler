// own spec 93: a @weak read borrows its object and cannot be stored in an owning field.
class Node { @weak parent: Node | null = null; kept: Node | null = null; v = 1; }
function main() { const p = new Node(); const c = new Node(); c.parent = p; c.kept = c.parent; print(p.v); }
