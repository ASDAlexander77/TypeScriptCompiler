// An object literal holding a fresh instance of a class another module defines, given to a call: the
// literal borrows it, as it does a class of its own module's. Under -mm=own it failed with "an object
// literal that holds a value it owns is moved only into a local or returned as it is" - `new Date(0)`
// in `JSON.stringify({ when: new Date(0) })`: the imported `..new` was known to make a fresh value
// only after the literals had been classified.
import './export_record_stamp'

function at(o: { when: Stamp }) {
    return o.when.at;
}

function name<T>(o: T) {
    return "" + typeof o;
}

function both(o: { first: Stamp, second: Stamp }) {
    return o.first.at + o.second.at;
}

function main() {
    assert(at({ when: new Stamp(3) }) == 3, "a literal argument holding an imported class");
    assert(name({ when: new Stamp(4) }) == "object" || name({ when: new Stamp(4) }) == "tuple", "to a generic");
    assert(both({ first: new Stamp(1), second: new Stamp(2) }) == 3, "two of them");
    const o = { when: new Stamp(5) };
    assert(at(o) == 5, "a local");
    print("done.");
}
