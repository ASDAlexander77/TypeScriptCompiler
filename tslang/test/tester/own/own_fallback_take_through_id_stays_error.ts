// -mm=own --own-fallback=count: a box whose field is a tagged union (`Node | string`), kept in a
// global and taken from through an identity function. A counted object can reach any take of a
// field out of it, and nothing can give a tagged union's payload an owner under own (it is reached
// through a type descriptor): so an object that may have such a field is never counted, and the
// program stays an error with the flag (else the take would empty the field under the global).
class Node { constructor(public v: number) {} }
class Box { node: Node | string = "x"; }
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
