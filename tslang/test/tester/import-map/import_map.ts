// A module importing another, with a Map (or a dictionary, which holds one): the keys adapter of what the
// library boxes (a RangeError it throws) was made while the imported file was read as declarations - a
// declaration, made public - and the module failed: "'ts.Func' op symbol declaration cannot have public
// visibility". A dictionary keyed by an imported value is made here too.
import { prefix, count } from "./map_module";

function main() {
    const m = new Map<string, boolean>();
    m.set(prefix, true);
    assert(m.get("k") === true);

    const d: { [key: string]: number } = {};
    for (let i = 0; i < count; i++) {
        d[prefix + i] = i;
    }

    const e = { [prefix]: count };
    assert(d["k2"] == 2);
    assert(e["k"] == 3);
    assert(Object.keys(d).length == 3);
    print("done.");
}
