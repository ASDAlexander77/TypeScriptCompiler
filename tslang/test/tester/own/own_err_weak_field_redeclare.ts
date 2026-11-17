// own spec 93: a field over a base class's @weak field is the same slot, so it is @weak as well
class Node { @weak parent: Node | null = null; v = 1; }
class Leaf extends Node { parent: Leaf | null = null; }
function main() { const a = new Leaf(); const b = new Leaf(); b.parent = a; print(b.v); }
