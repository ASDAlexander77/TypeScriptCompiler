// The object a WeakRef refers to is its owner's: a borrow of it cannot be stored.
class Node { v = 1; }
class Holder { w: WeakRef<Node> | null = null; n: Node | null = null; kept: Node | null = null; }
function main() { const h = new Holder(); const n = new Node(); h.w = new WeakRef(n); h.n = n; const p = h.w.deref(); if (p) h.kept = p; }
