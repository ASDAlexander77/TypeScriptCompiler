import { Base } from "./base_module";
import { Derived } from "./derived_module";

function key(s: string) {
    return s;
}

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

    // a Derived of another module held as a Base, read by a key known at run time: its own field, through
    // the member table of its class (the library's)
    const asBase: Base = d;
    assert(<string>asBase[key("name")] == "derived", "actual class by key");
    assert(key("name") in asBase, "actual class in");

    print("done.");
}
