// -mm=own --own-fallback=count: the same shape as own_fallback_take_through_id_stays_error with a
// field of one pointer (`Node | null`): the box is counted, the take through the identity function
// leaves the field to the global (ts.TakeOutOfField with an extra owner) and gives the node an owner.
class Node { constructor(public v: number) {} }
class Box { node: Node | null = null; }
let last: Box | null = null;
function id(b: Box) { return b; }
function takeTemp(v: number) {
  const b = new Box(); b.node = new Node(v);
  last = b;
  return id(b).node;
}
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
