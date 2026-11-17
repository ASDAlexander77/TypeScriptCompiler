// `new c()` through a class value, given where a union or an optional is wanted: the instance moves
// into it. The `.new` call's mark that its result is fresh was lost when it was lowered, so -mm=own
// took the cast for a second reference. Values are made at run time, in a loop, with strings that
// own memory, so a double release or a leak shows.
class A {
    constructor(public s: string) {}
    name() {
        return "A" + this.s;
    }
}

class B {
    constructor(public s: string) {}
    name() {
        return "B" + this.s;
    }
}

function asUnion(c: typeof A, s: string): A | B {
    return new c(s);
}

function asOptional(c: typeof A, s: string): A | null {
    return new c(s);
}

function viaLocal(c: typeof A, s: string): A | B {
    const o = new c(s);
    const u: A | B = o;
    return u;
}

function eitherOf(c: typeof A | typeof B, s: string): A | B {
    return new c(s);
}

function main() {
    let total = 0;
    const kept: (A | B)[] = [];
    for (let i = 0; i < 100; i++) {
        const s = "" + i;
        total += asUnion(A, s).name().length;
        const o = asOptional(A, s);
        if (o) {
            total += o.name().length;
        }

        total += viaLocal(A, s).name().length;
        kept.push(eitherOf(i % 2 == 0 ? A : B, s));
    }

    assert(total == 3 * (10 * 2 + 90 * 3), "made each");
    assert(kept.length == 100 && kept[0].name() == "A0" && kept[1].name() == "B1", "kept, each its class");

    print("done.");
}
