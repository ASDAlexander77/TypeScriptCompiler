// @strict-null false
// -mm=own rejects: `x` borrows the field `h.c` of a parameter, and a global is assigned before `x`
// is read. The parameter may be the global's old value, which the assignment frees, and its field
// with it (spec 29).
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

let G: H = mkH();

function run(h: H) {
    h.c = mk();
    const x = h.c;
    G = mkH();
    churn();
    print(x.x);
}

// The rule is the callee's: it is given a fresh value here, since a written global's value lent to
// a call that may drop is rejected at the call already (#639).
function main() {
    run(mkH());
}
