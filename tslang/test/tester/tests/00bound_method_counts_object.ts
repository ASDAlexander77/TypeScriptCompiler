// A method bound to an object keeps that object alive while it is held: bound through an
// interface (`c = ig.get`), or held by a global from its initializer (`let c = mv(3).get`,
// `let c = G.v.get`). Under rc a bound value counts its object through the object's type tag, and
// only its receivers take a count. Through an interface it had no tag and counted nothing, and a
// global's initializer took no count, so the object was given back while the bound value still
// ran on it. Each read follows allocations of the same size, so a freed block is reused.
class V {
    constructor(public x: number) {}

    get(): number {
        return this.x;
    }
}

class N {
    v = new V(4);
}

interface IG {
    get(): number;
}

function mv(x: number) {
    return new V(x);
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new V(777));
    }
}

let G = new N();
let fromTemporary: () => number = mv(3).get;
let fromField: () => number = G.v.get;

function boundThroughInterface(x: number) {
    let c: () => number = mv(0).get;
    {
        const ig: IG = mv(x);
        c = ig.get;
    }

    churn();
    return c();
}

function globalBoundThroughInterface(x: number) {
    {
        let ig: IG = mv(x);
        fromTemporary = ig.get;
    }

    churn();
    return fromTemporary();
}

function main() {
    churn();
    assert(fromTemporary() == 3, "a global bound to a temporary object by its initializer");
    G.v = mv(9);
    churn();
    assert(fromField() == 4, "a global bound to a global's field by its initializer");
    fromField = mv(6).get;
    churn();
    assert(fromField() == 6, "a global bound again");

    for (let i = 0; i < 3; i++) {
        assert(boundThroughInterface(i + 10) == i + 10, "a local bound through an interface");
        assert(globalBoundThroughInterface(i + 20) == i + 20, "a global bound through an interface");
    }

    print("done.");
}
