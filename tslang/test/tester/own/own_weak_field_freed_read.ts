// own spec 93: a read of a @weak field whose object was freed stops the program under own; gc keeps
// the parent alive and none never frees, so they print its value
class Node {
    @weak parent: Node | null = null;
    constructor(public v: number) {}
}

// the parent is freed when this returns; the child keeps its @weak link to it
function attach(child: Node) {
    const parent = new Node(1);
    child.parent = parent;
    if (child.parent.v != 1) throw "attached";
}

function main() {
    const child = new Node(2);
    attach(child);
    print("before");
    print(child.parent.v);
    print("after");
    print("done.");
}
