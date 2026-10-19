// A function-typed interface field is read-only: a read binds the interface's `this`, and a
// write could not be checked against the object behind the interface. For a class the slot
// points at a method cell shared by every instance (INTERFACE_METHOD_CELL_TAG), which a write
// must not reach.
interface MaybeFn { fn?: () => int; }
class O { fn?: () => int; }
function main() {
    const m: MaybeFn = new O();
    m.fn = () => 9;
}
