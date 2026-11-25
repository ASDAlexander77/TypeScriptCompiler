// TSLANG 191: `export =` and `import x = require("./m")`, each module compiled on its own. They were "'export =' is
// not supported" and "'import X = require(...)' is not supported".
import L = require("./export_equals_ns");
import Counter = require("./export_equals_class");

function main() {
    assert(L.twice(21) == 42, "a function of the namespace a module exports");
    assert(L.name == "lib", "a const of the namespace a module exports");

    const c = new Counter();
    c.inc();
    assert(c.inc() == 2, "the class a module exports");

    print("done.");
}
