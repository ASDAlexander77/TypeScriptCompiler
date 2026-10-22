// @strict-null false
// -mm=own rejects: a value moved into a field of a parameter or of `this` is read after a global
// is assigned - here, or in a callee. The parameter or `this` may be the global's old value, which
// the assignment frees, and the moved value with it (spec 29).
class C {
    x: number = 5;
    v: number[] = [];
}

class H {
    c: C | null = null;
}

function mkH(): H {
    const r = new H();
    return r;
}

function mk(): C {
    const r = new C();
    r.v.push(1);
    return r;
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = mk();
        c.x = 999;
        c.v.push(99);
        junk.push(c);
    }
}

class K {
    f: C | null = null;
    run() {
        const c = mk();
        this.f = c;
        GK = new K();
        churn();
        print(c.x);
    }
}

let G: H = mkH();
let GK: K = new K();

function reset() {
    G = mkH();
}

function run(h: H) {
    const c = mk();
    h.c = c;
    G = mkH();
    churn();
    print(c.x);
}

function runCallee(h: H) {
    const c = mk();
    h.c = c;
    reset();
    churn();
    print(c.x);
}

// The rule is the callees': each is given a fresh value here, since a written global's value lent
// to a call that may drop is rejected at the call already (#639).
function main() {
    run(mkH());
    new K().run();
    runCallee(mkH());
}
