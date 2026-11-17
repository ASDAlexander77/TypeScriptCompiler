// own spec 93: comparing a @weak field whose object was freed reads the stored pointer and never stops
class Node {
    @weak parent: Node | null = null;
    constructor(public v: number) {}
}

function churn(n: number) {
    let total: number = 0;
    for (let i = 0; i < n; i++) {
        const t = new Node(i);
        total += t.v;
    }
    return total;
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
    churn(1000);
    const p = new Node(3);
    if (child.parent === p) throw "same as a new object";
    if (!(child.parent !== null)) throw "null";
    if (child.parent == null) throw "loose null";
    child.parent = null;
    if (child.parent !== null) throw "cleared";
    churn(1000);
    print("done.");
}
