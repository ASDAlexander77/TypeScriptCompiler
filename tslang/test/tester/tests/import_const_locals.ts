import './export_const_locals'

// A module imported as source must compile when its functions read `const` locals.
// See export_const_locals.ts.

function main() {
    assert(plainConst(1) == 2, "plainConst");
    assert(inferredFromConst(3) == 5, "inferredFromConst");
    assert(stringConst(7) == "v-7", "stringConst");
    assert(constInLoop(4) == 12, "constInLoop");
    assert(constInBlock(123) == 3, "constInBlock");
    assert(new WithConst().method(5) == 15, "WithConst.method");
    assert(arrayDestructuring(3) == 7, "arrayDestructuring");
    assert(objectDestructuring(4) == 9, "objectDestructuring");
    assert(NS.namespaceConst(12) == 5, "NS.namespaceConst");

    print("done.");
}
