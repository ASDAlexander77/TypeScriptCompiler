import { Base } from "./base_module";
import { Derived } from "./derived_module";

function main() {
    const b = new Base();
    b.add(1);
    b.add(2);
    assert(b.items.length == 2, "add");
    assert(b.count == 2, "count");
    assert(b.last() == 2, "last");
    b.first = 7;
    assert(b.items[0] == 7, "first");

    const d = new Derived();
    d.add(3);
    assert(d.count == 1, "derived count");
    assert(d.last() == 3, "derived last");
    assert(d.name == "derived", "derived name");

    print("done.");
}
