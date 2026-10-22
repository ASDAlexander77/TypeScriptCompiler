// An argument read out of a global, or out of a global's field, stays alive when a later argument
// of the same call gives that global a new value: `check(G, (G = mk(2), G), 1)`. Under rc the
// caller holds a count across a call that may assign the global (00global_argument_held.ts), and
// took it at the call - after the later argument had already freed the block, so the hold
// retained and released a freed block. Each read follows allocations of the same size, so a freed
// block is reused.
// @strict-null false
class C {
    x = 5;
    tag = "";
    v: number[] = [];
}

class H {
    c: C = mk(3);
}

function mk(x: number): C {
    const r = new C();
    r.x = x;
    r.tag = "c" + x;
    r.v.push(x);
    return r;
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C();
        c.x = 999;
        c.tag = "junk";
        c.v.push(99);
        junk.push(c);
    }
}

let G: C = mk(1);
const GH = new H();

function thrower(): C {
    throw "thrown";
}

// assigns both globals, so a caller holds what it reads out of them
function check(c: C, other: C, x: number) {
    G = mk(100 + x);
    GH.c = mk(200 + x);
    churn();
    return c.x == x && c.tag == "c" + x && c.v[0] == x && other.x > 0;
}

function inLoop() {
    for (let i = 0; i < 3; i++) {
        const x = G.x;
        assert(check(G, (G = mk(10 + i), G), x), "a global, in a loop");
    }
}

function inTryCatch() {
    G = mk(30);
    let caught = false;
    try {
        check(G, (G = mk(31), thrower()), 30);
    } catch (e) {
        caught = true;
    }

    churn();
    assert(caught && G.x == 31 && G.tag == "c31", "a later argument throws after the assignment");
}

function main() {
    assert(check(G, (G = mk(2), G), 1), "a global given before an argument that assigns it");
    const fx = GH.c.x;
    assert(check(GH.c, (GH.c = mk(4), GH.c), fx), "a global's field given before an argument that assigns it");
    inLoop();
    inTryCatch();
    churn();
    assert(G.x == 31 && GH.c.x == 701, "the globals' last values");
    print("done.");
}
