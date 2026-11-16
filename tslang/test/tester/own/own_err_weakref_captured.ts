// The object a WeakRef refers to is its owner's: a closure that escapes cannot capture a borrow of it.
class Node { v: number = 1; }
class Holder { w: WeakRef<Node> | null = null; n: Node | null = null; kept: Node | null = null; }
let FNS: (() => number)[] = [];
function main() { const n = new Node(); const w = new WeakRef(n); const p = w.deref(); if (p) FNS.push(() => p.v); }
