// A call that may free a Node ends a borrow of the object a WeakRef refers to.
class Node { v = 1; }
class Holder { w: WeakRef<Node> | null = null; n: Node | null = null; kept: Node | null = null; }
function dropAll(h: Holder) { h.n = null; }
function main() { const h = new Holder(); const n = new Node(); h.w = new WeakRef(n); h.n = n; const p = h.w.deref(); dropAll(h); print(p ? p.v : 0); }
