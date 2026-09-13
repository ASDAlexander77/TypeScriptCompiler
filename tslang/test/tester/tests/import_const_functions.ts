import './export_const_functions'

// Calling exported const functions from another module. See export_const_functions.ts.

function main() {
    assert(arrowBlock(2) == 6, "arrowBlock");
    assert(arrowExpression(2) == 8, "arrowExpression");
    assert(functionExpression(2) == 10, "functionExpression");
    assert(arrowWithConstLocal(7) == "v-7", "arrowWithConstLocal");
    assert(usesNotExported(1) == 101, "usesNotExported");
    assert(NS.inNamespace(2) == 12, "NS.inNamespace");

    print("done.");
}
