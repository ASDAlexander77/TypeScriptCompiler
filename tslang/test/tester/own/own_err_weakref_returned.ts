// The object a WeakRef refers to is its owner's: a borrow of it cannot be returned.
class Node { v = 1; }
class Holder { w: WeakRef<Node> | null = null; n: Node | null = null; kept: Node | null = null; }
function get(w: WeakRef<Node>) { const p = w.deref(); return p; }
function main() { const n = new Node(); const w = new WeakRef(n); print(get(w) ? 1 : 0); }
