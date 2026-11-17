// own spec 93 across a module boundary: a freed read through a class another module declares stops.
import './export_weak_field'

function attach(child: Node) {
    const parent = new Node(1);
    child.parent = parent;
}

function main() {
    const child = new Node(2);
    attach(child);
    print("before");
    print(child.parent.v);
    print("after");
    print("done.");
}
