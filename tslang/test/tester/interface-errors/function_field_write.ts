// An optional function-typed interface field is not written through the interface: a write would
// have to set the instance's presence flag too, which the class's cell names only for a read
// (INTERFACE_METHOD_CELL_TAG). A non-optional one filled by a class field is written
// (00interface_function_field_write.ts).
interface MaybeFn { fn?: () => int; }
class O { fn?: () => int; }
function main() {
    const m: MaybeFn = new O();
    m.fn = () => 9;
}
