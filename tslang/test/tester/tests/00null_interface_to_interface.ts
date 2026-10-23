// @strict-null false
// A null interface cast to an interface it does not extend, whose members are elsewhere (#666 builds
// that interface's vtable out of the value's own one) is null still, and its methods throw: the null
// vtable's slots, stubs that throw, are what the new vtable is made of.
interface Hit {
    thing: number;
    dist: number;
    get(): number;
}

interface HasDist {
    dist: number;
    get(): number;
}

class E {
    constructor(public thing: number, public dist: number) {}

    get(): number {
        return this.dist * 10;
    }
}

@noinline function mkE(flag: boolean, thing: number, dist: number): E {
    return flag ? new E(thing, dist) : null;
}

// what a method of the interface does: -2 when it throws
function call(h: HasDist): number {
    try {
        return h.get();
    } catch (e) {
        return -2;
    }
}

function main() {
    // a field first, and no method: a method called through a wrong vtable runs whatever code it lands on
    const liveHit: Hit = mkE(true, 1, 2);
    const liveDist: HasDist = liveHit;
    assert(liveDist.dist == 2, "a field read first");
    assert(call(liveDist) == 20, "a class object seen through another interface");

    const nullHit: Hit = mkE(false, 1, 2);
    const nullDist: HasDist = nullHit;
    assert(nullDist === null, "a null class object seen through another interface");
    assert(call(nullDist) == -2, "its method throws");

    const nullLit: Hit = null;
    const nullLitDist: HasDist = nullLit;
    assert(nullLitDist === null, "null seen through another interface");
    assert(call(nullLitDist) == -2, "its method throws too");

    print("done.");
}
