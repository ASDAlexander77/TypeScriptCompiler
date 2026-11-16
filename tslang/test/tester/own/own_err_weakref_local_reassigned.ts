// The object a WeakRef refers to may be a local's: the local given another value frees it, and ends
// a borrow of it.
class Node { v = 1; }
class Holder { w: WeakRef<Node> | null = null; n: Node | null = null; kept: Node | null = null; }
function main() { let n = new Node(); const w = new WeakRef(n); const p = w.deref(); n = new Node(); print(p ? p.v : 0); }
