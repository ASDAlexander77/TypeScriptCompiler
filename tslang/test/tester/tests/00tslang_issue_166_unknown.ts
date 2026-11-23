// TSLANG_ISSUES 166: a module variable of type `unknown` keeps the box its initializer makes. Under -mm=rc the box was
// given back at the end of the statement and read as freed memory ("Can't cast from any type"). Not under -mm=own,
// which refuses such a global's initializer ("takes a second reference").
let u: unknown = 7;
let w: unknown = "a" + 1;

class K {
    n = 4;
}

const k: unknown = new K();

function main() {
    assert(<number>u == 7, "number");
    assert(<string>w == "a1", "string");
    assert((<K>k).n == 4, "object");
    u = "s";
    assert(<string>u == "s", "assigned");
    print("done.");
}
