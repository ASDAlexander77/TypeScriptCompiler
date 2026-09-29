// Under the JIT cache each module is an object of its own: this program, sub/derived_module and
// other_module, which both import base_module - loaded once, and initialized before either.
import './sub/derived_module'
import './other_module'

assert(doubled == 30, "base_module initialized before derived_module");

const d = new Derived();
assert(d.hello() == "derived:base");
assert(Base.created == 1);

assert(bumpFromDerived() == 16);
assert(bumpFromOther() == 17);
assert(counter == 17, "one base_module, shared by both");

let caught = "";
try {
    throw "boom";
} catch (e) {
    caught = <string>e;
}

assert(caught == "boom");

print("done.");
