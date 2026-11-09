// A field a subclass declares again (export_redeclared_field.ts) is the base's, across a -shared
// boundary too: the importer reads it as the exporter laid it out, and through the subclass as its type

import './export_redeclared_field'

function main() {
    const n = new NumberHolder();
    const h: Holder = n;
    assert(n.id == 3 && h.id == 3, "the base's type");
    assert(n.v + 1 == 4 && h.v == 3, "a narrower type");

    n.v = 5;
    assert(h.v == 5, "a write through the subclass");

    print("done.");
}
