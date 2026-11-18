// A symbol made in another library (export_symbol.ts) is the same symbol here: `Symbol.for("x")` of
// either side, a well-known symbol, an exported symbol read back

import './export_symbol'

function main() {
    assert(k === getK(), "exported symbol");
    assert(Symbol.for("x") === forX(), "Symbol.for across libraries");
    assert(Symbol.iterator === iterator(), "well-known symbol across libraries");
    assert(Symbol.for("y") !== forX(), "another description");
    assert(k !== Symbol("k"), "same description, another symbol");

    print("done.");
}
