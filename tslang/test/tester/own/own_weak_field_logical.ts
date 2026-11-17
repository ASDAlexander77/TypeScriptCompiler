// own spec 93: `??`, `??=`, `||=` and `&&=` on a @weak field: reads of it (checked), writes of it (counted),
// and the value of the expression is the field's declared type, as under gc and none
class Node {
    @weak parent: Node | null = null;
    constructor(public v: number) {}
}

function main() {
    const root = new Node(1);
    const other = new Node(5);
    const c = new Node(2);

    const a = c.parent ?? root;
    if (a.v != 1) throw "?? null";
    c.parent = other;
    const b = c.parent ?? root;
    if (b.v != 5) throw "?? set";

    c.parent = null;
    const d = (c.parent ??= root);
    if (d === null || d.v != 1 || c.parent !== root) throw "??=";

    c.parent = null;
    const e = (c.parent ||= other);
    if (e === null || e.v != 5 || c.parent !== other) throw "||=";

    const f = (c.parent &&= root);
    if (f === null || f.v != 1 || c.parent !== root) throw "&&=";
    print("done.");
}
