// `ns.outer()` is an error when only the root namespace has an `outer`: the general lookup the
// access fell back to also searched the outer namespaces.

function outer(): int {
    return 1;
}

namespace ns {
    export function value(): int {
        return 42;
    }
}

function main() {
    print(ns.outer());
}
