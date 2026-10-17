// regression test: `ns.name` is the namespace's member, whatever a local of the same name is.
//
// bug found and fixed: a local `const value = ns.value()`, then `ns.value()` further down the same
// function: the qualified name resolved to the local, which isn't callable ("not a function to
// call").

namespace ns {
    export function value(): int {
        return 42;
    }

    export const k = 7;

    export namespace inner {
        export function value(): int {
            return 9;
        }
    }
}

// a member declared after its use, and one in a second block of the same namespace
function early(): int {
    return ns2.later() + ns2.second();
}

namespace ns2 {
    export function later(): int {
        return 1;
    }
}

namespace ns2 {
    export function second(): int {
        return 2;
    }
}

function main() {
    assert(early() == 3);

    const value = ns.value();
    assert(value == 42);
    assert(ns.value() == 42);

    const k = ns.k + 1;
    assert(k == 8);
    assert(ns.k == 7);

    const inner = 3;
    assert(ns.inner.value() == 9);
    assert(inner == 3);

    {
        // a local named as the namespace is the local
        const ns = { value: () => 5 };
        assert(ns.value() == 5);
    }

    print("done.");
}
