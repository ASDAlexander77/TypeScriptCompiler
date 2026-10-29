// A function taking fewer parameters than the function type it is given as ignores the rest, as TypeScript lets
// it: a named function or an arrow that captures nothing is given as a function that takes them all and calls
// it with the first ones. It was an error, "it takes 1 argument(s), 2 expected".
type Visit = (name: string, index: number) => void;

function each(names: string[], visit: Visit) {
    for (let i = 0; i < names.length; i++) {
        visit(names[i], i);
    }
}

function apply(f: (a: number, b: number) => number) {
    return f(2, 3);
}

let seen = "";
function onlyName(name: string) { seen += name; }
function nothing() { seen += "."; }
function twice(a: number): number { return a * 2; }
function seven(): number { return 7; }

class Holder {
    constructor(public cb: (a: number, b: number) => number) {}
    run() { return this.cb(4, 5); }
}

function main() {
    assert(apply(twice) == 4, "one of two parameters");
    assert(apply(seven) == 7, "none of two");
    assert(apply((a) => a + 1) == 3, "an arrow with fewer parameters");

    each(["a", "b"], onlyName);
    each(["c"], nothing);
    assert(seen == "ab.", "a void callback, a string parameter");

    // the same function given twice is one wrapper
    each(["d"], onlyName);
    assert(seen == "ab.d", "again");

    const h = new Holder(twice);
    assert(h.run() == 8, "kept in a field");

    const f: (a: number, b: number) => number = (a) => a * 10;
    assert(f(1, 2) == 10, "assigned to a variable");

    print("done.");
}
