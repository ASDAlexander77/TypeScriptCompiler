// own spec 93: a @weak field owns nothing, so a new object written to it would be freed at once.
class Node { @weak parent: Node | null = null; v = 1; }
function main() { const c = new Node(); c.parent = new Node(); print(c.v); }
