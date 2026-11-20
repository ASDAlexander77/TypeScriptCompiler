import { Base } from "./base_module";
import { Derived, Tagged, tagOf } from "./derived_module";

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

    // a Derived of another module held as a Base, read by a key known at run time: Derived is only exported,
    // so its module made it no keys adapter and it is read by Base's keys - Base's fields, not its own
    const asBase: Base = d;
    assert((<number[]>asBase[key("items")]).length == 1, "base field by key");
    assert(asBase[key("name")] == undefined, "exported-only class read by its base's keys");
    assert(!(key("name") in asBase), "exported-only class in");

    // a Tagged, which its own module reads by key: read by its own keys through its member table
    const t = new Tagged();
    assert(<string>tagOf(t, "tag") == "tagged", "tag in its module");
    const tAsBase: Base = t;
    assert(<string>tAsBase[key("tag")] == "tagged", "actual class by key");
    assert(key("tag") in tAsBase, "actual class in");

    print("done.");
}
