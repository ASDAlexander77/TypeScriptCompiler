// `import b = a.member` of an enum, a function or a variable: b names that member, as a value and (an
// enum's) as a type ("this import-equals target is not supported" before). A parameter written `never`
// takes what it is given ("'Never' can't be used as parameter type" before).
namespace a {
    export enum weekend { Friday, Saturday, Sunday }
    export function hello(n: number) { return n + 1; }
    export var count = 5;
}

namespace c {
    import b = a.weekend;
    export var bVal: b = b.Sunday;
    export import h = a.hello;
}

import cnt = a.count;

type Shape = { kind: "circle", r: number } | { kind: "square", s: number };
function assertNever(x: never): never { throw "unexpected shape"; }
function area(s: Shape) {
    switch (s.kind) {
        case "circle": return 3 * s.r * s.r;
        case "square": return s.s * s.s;
        default: return assertNever(s);
    }
}

function main() {
    assert(c.bVal == a.weekend.Sunday);
    assert(c.h(1) == 2);
    assert(cnt == 5);
    assert(area({ kind: "square", s: 2 }) == 4);
    print("done.");
}
