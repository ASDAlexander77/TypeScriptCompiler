// A local's retain at its declaration takes its initializer, whatever is stored into it later:
// MLIRGen puts the `ts.RetainSlot` right after the declaration, before anything else can write the
// slot. Where the initializer is data nothing owns - null in a union, a tuple of literals - the
// retain takes nothing, and each later store is decided on its own.
class C {
    constructor(public x: number) {}
}

function main() {
    let y: number | null = null;
    assert((y ?? 8) == 8);
    y = 0;
    assert((y ?? 9) == 0);

    let u: string | number | null = null;
    u = "s";
    assert(typeof u == "string");
    u = 3;
    assert(typeof u == "number");

    let c: C | number | null = null;
    c = new C(4);
    assert(c !== null && typeof c != "number");

    let pair: [name: string, age: number] = ["user", 10.0];
    assert(pair.name == "user" && pair.age == 10.0);
    pair.name = "other";
    assert(pair.name == "other");

    print("done.");
}
