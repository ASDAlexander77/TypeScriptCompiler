// own spec 93: an interface read of a @weak field would skip the check and the count.
class Node { @weak parent: Node | null = null; v = 1; }
interface HasParent { parent: Node | null; }
function main() { const n = new Node(); const h: HasParent = n; print(h.parent === null); }
