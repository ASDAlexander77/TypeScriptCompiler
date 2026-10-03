// A module-level variable whose initializer branches: an optional widened into a union with null
// (#462), and a conditional expression. Lowered, the initializer had more than one block, which a
// global's region could not hold: "'ts.Global' op expects region #0 to have 0 or 1 blocks".
class Node {
    v = 0;
    tag = "";
}

function mk(v: number) {
    const n = new Node();
    n.v = v;
    n.tag = "t" + v;
    return n;
}

function mkU(): Node | undefined {
    return mk(7);
}

function none(): Node | undefined {
    return undefined;
}

function on() {
    return true;
}

let g: Node | null | undefined = mkU();
let h: Node | null | undefined = none();
let x = on() ? 1 : 2;
const label = on() ? "yes" : "no";
let pick: Node | null = on() ? mk(3) : null;

function main() {
    assert(g !== undefined && g !== null && g.tag == "t7", "an optional widened into a union");
    assert(h === undefined, "an empty optional widened into a union");
    assert(x == 1, "a conditional number");
    assert(label == "yes", "a conditional string");
    assert(pick !== null && pick.v == 3, "a conditional object");

    print("done.");
}
