// own spec 93: an object type naming a @weak field would read it without the check.
class Node { @weak parent: Node | null = null; v = 1; }
function up(o: { parent: Node | null }) { return o.parent === null; }
function main() { const n = new Node(); print(up(n)); }
