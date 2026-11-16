// -mm=own rejects: `n` and `m` are made outside the loop, and a select inside it would move them
// into `h.node` again on every iteration.
class Node { name = ""; }
class H { node: Node = new Node(); }
function f(h: H, set: boolean) {
    const n = new Node(); n.name = "n";
    const m = new Node(); m.name = "m";
    for (let i = 0; i < 3; i++) {
        h.node = set ? n : m;
    }
}
function main() { const h = new H(); f(h, true); print(h.node.name); }
