// own spec 93: a field over a base class's field that is not @weak cannot make the one slot weak
class Node { parent: Node | null = null; v = 1; }
class Leaf extends Node { @weak parent: Leaf | null = null; }
function main() { const a = new Leaf(); const b = new Leaf(); b.parent = a; print(b.v); }
