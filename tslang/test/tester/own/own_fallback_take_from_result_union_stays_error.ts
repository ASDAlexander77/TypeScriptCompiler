// -mm=own --own-fallback=count: a field of a tagged union (`Node | string`) taken out of a call's own
// result that another owner also holds. Under own nothing can give a tagged union's payload an owner
// (it is reached through a type descriptor, which has no retain routine under own), so the take is
// refused for an object that may be counted, and the program stays an error with the flag.
class Node { constructor(public v: number) {} }
class Box { node: Node | string = "x"; }
let last: Box | null = null;
function mk(v: number) { const b = new Box(); b.node = new Node(v); last = b; return b; }
function takeTemp(v: number) { return mk(v).node; }
function main() {
  let s: number = 0;
  for (let i = 0; i < 1000; i++) {
    const n = takeTemp(i);
    if (n instanceof Node) s += n.v;
    const k = last!.node;
    if (k instanceof Node) s += k.v; else s -= 1000000;
  }
  print(s); print("done.");
}
