// An argument read out of a global stays alive while the callee gives the global a new value. The
// argument is borrowed, so under rc the global's release freed the block the parameter still read.
// Each read follows allocations of the same size, so a freed block is reused.
// @strict-null false
class C {
    x = 5;
    tag = "";
    v: number[] = [];

    m() {
        G = mk(7);
        churn();
        return this.x;
    }
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

function run(h: C) {
    G = mk(2);
    churn();
    return h.x == 1 && h.tag == "c1" && h.v[0] == 1;
}

function runField(c: C) {
    GH.c = mk(4);
    churn();
    return c.x == 3 && c.tag == "c3";
}

function main() {
    assert(run(G), "a global given to a function that assigns it");
    assert(runField(GH.c), "a global's field given to a function that assigns it");
    G = mk(6);
    assert(G.m() == 6, "a method of a global that assigns it");
    print("done.");
}
