// A module-level `Node | null` (or `Node | undefined`) initialised from a call keeps the object the
// call made (#459). Under rc the call's reference was released at the end of the initializer, since
// the global's union cast hid it, and the global pointed at freed memory. Each global is read after
// allocations of the same size have reused anything freed.
class Node {
    v = 0;
    name = "";
}

function mk(name: string) {
    const n = new Node();
    n.name = name;
    return n;
}

function churn() {
    let keep: Node[] = [];
    for (let i = 0; i < 1000; i++) {
        const n = new Node();
        n.v = -i;
        n.name = "junk" + i;
        keep.push(n);
    }

    return keep.length;
}

let G: Node | null = mk("made" + 1);
const C: Node | null = mk("const" + 2);
let U: Node | undefined = mk("optional" + 3);

function main() {
    assert(churn() == 1000);
    assert(G !== null && G.name == "made1", "a union global from a call keeps its object");
    assert(C !== null && C.name == "const2", "a union const from a call keeps its object");
    assert(U !== undefined && U.name == "optional3", "an optional global from a call keeps its object");

    G = mk("next" + 5);
    assert(churn() == 1000);
    assert(G !== null && G.name == "next5", "a reassigned union global keeps the new object");

    print("done.");
}
