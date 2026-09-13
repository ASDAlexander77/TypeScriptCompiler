import './export_generators'

// Iterating generators exported from another module. See export_generators.ts.

function main() {
    let a = 0;
    for (const v of functionGenerator()) a += v;
    assert(a == 3, "functionGenerator");

    let b = 0;
    for (const v of constGenerator()) b += v;
    assert(b == 7, "constGenerator");

    let c = 0;
    for (const v of new WithGenerator().items()) c += v;
    assert(c == 11, "WithGenerator.items");

    let d = 0;
    for (const v of NS.inNamespace()) d += v;
    assert(d == 15, "NS.inNamespace");

    const g = functionGenerator();
    assert(g.next().value == 1, "next 1");
    assert(g.next().value == 2, "next 2");
    assert(g.next().done, "next done");

    print("done.");
}
