// An argument read out of a global and given to a call as a view - an interface, a base class, an
// object literal holding it - stays alive when a later argument gives the global a new value:
// `check(G, (G = mk(2), 0))` with an interface parameter, `check({ c: G.c }, (G = mk(2), 0))`.
// The view is made at the call, after every argument is evaluated, so under rc a hold taken on the
// view (00global_argument_reassigned.ts) came after the later argument had freed the block, and an
// object literal held what it was given with no count until then. Each read follows allocations of
// the same size, so a freed block is reused.
class Base {
    tag = "";
    constructor(public x: number) {
        this.tag = "c" + x;
    }
}

class C extends Base {
    v: number[] = [];
}

class H {
    c: C = mk(3);
}

interface IC {
    x: number;
    tag: string;
}

interface IHolder {
    c: C;
}

function mk(x: number): C {
    const r = new C(x);
    r.v.push(x);
    return r;
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.tag = "junk";
        c.v.push(99);
        junk.push(c);
    }
}

let G: C = mk(1);
const GH = new H();

function thrower(): number {
    throw "thrown";
}

function viaInterface(c: IC, n: number, x: number) {
    churn();
    return c.x == x && c.tag == "c" + x;
}

function viaBase(c: Base, n: number, x: number) {
    churn();
    return c.x == x && c.tag == "c" + x;
}

function viaHolder(h: IHolder, n: number, x: number) {
    churn();
    return h.c.x == x && h.c.tag == "c" + x && h.c.v[0] == x;
}

function viaRecord(h: { c: C }, n: number, x: number) {
    churn();
    return h.c.x == x && h.c.tag == "c" + x && h.c.v[0] == x;
}

function inLoop() {
    for (let i = 0; i < 3; i++) {
        const x = G.x;
        assert(viaInterface(G, (G = mk(10 + i), 0), x), "an interface over a global, in a loop");
        const y = G.x;
        assert(viaHolder({ c: G }, (G = mk(20 + i), 0), y), "an object literal holding a global, in a loop");
    }
}

function inTryCatch() {
    G = mk(30);
    let caught = false;
    try {
        viaInterface(G, (G = mk(31), thrower()), 30);
    } catch (e) {
        caught = true;
    }

    churn();
    assert(caught && G.x == 31 && G.tag == "c31", "a later argument throws after the assignment");
}

function main() {
    assert(viaInterface(G, (G = mk(2), 0), 1), "an interface over a global");
    assert(viaBase(G, (G = mk(3), 0), 2), "a global as its base class");
    assert(viaInterface(GH.c, (GH.c = mk(4), 0), 3), "an interface over a global's field");
    assert(viaHolder({ c: G }, (G = mk(5), 0), 3), "an object literal holding a global");
    assert(viaHolder({ c: GH.c }, (GH.c = mk(6), 0), 4), "an object literal holding a global's field");
    assert(viaRecord({ c: G }, (G = mk(7), 0), 5), "an object literal given as a record");
    inLoop();
    inTryCatch();
    churn();
    assert(G.x == 31 && GH.c.x == 6, "the globals' last values");
    print("done.");
}
