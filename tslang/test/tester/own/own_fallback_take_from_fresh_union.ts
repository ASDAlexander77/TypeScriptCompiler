// -mm=own --own-fallback=count: a field of a tagged union (`Node | string`) taken out of a call's own
// result that nothing else holds. Own accepts it, so the flag must too (it only turns errors into
// counts): the result is not counted - a callee never returns a counted object with a field reached
// through a type descriptor (countedGroup) - so the take empties the field as without the flag.
class Node { constructor(public v: number) {} }
class Box { node: Node | string = "x"; }
function mk(v: number) { const b = new Box(); b.node = new Node(v); return b; }
function takeTemp(v: number) { return mk(v).node; }
function main() {
  let s: number = 0;
  for (let i = 0; i < 1000; i++) {
    const n = takeTemp(i);
    if (n instanceof Node) s += n.v;
  }
  print(s); print("done.");
}
