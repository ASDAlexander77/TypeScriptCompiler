// `ns.missing` is an error when the namespace has no `missing`, even with a local `missing` in
// scope: it used to fall back to the general lookup, which found the local.

namespace ns {
    export function value(): int {
        return 42;
    }
}

function main() {
    const missing = 5;
    print(ns.missing);
}
