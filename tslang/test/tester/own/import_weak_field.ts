// own spec 93 across a module boundary: a class with a @weak field declared in another module is
// read and written as it is there - a WeakRef under own, an ordinary field under gc and none.
import './export_weak_field'

function main() {
    const root = new Node(1);
    const mid = new Node(2);
    const leaf = new Node(3);
    mid.parent = root;
    leaf.parent = mid;
    if (leaf.depth() != 2) throw "depth";
    if (leaf.parent.v != 2) throw "read";

    const a = new Node(10);
    const b = new Node(20);
    a.add(new Node(30));
    const moved = a.children.pop();
    if (moved) b.add(moved);
    if (b.children[0].parent !== b) throw "move";
    print("done.");
}
