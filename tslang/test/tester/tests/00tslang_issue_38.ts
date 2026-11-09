// TSLANG_ISSUES 38: a class named Symbol leaves the well-known names working: in a computed name always,
// and as a value unless the class has that member.

class Symbol {
    static iterator = "mine";

    constructor(public description: string) {}

    toString() {
        return "Symbol(" + this.description + ")";
    }
}

class Numbers {
    *[Symbol.iterator]() {
        yield 3;
    }

    [Symbol.dispose]() {
        print("disposed");
    }
}

function main() {
    for (const v of new Numbers()) {
        assert(v == 3);
    }

    assert(Symbol.iterator == "mine");
    assert(typeof Symbol.asyncIterator == "symbol");
    assert(new Symbol("x").toString() == "Symbol(x)");

    print("done.");
}
