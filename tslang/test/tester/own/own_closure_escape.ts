// -mm=own, phase 5b: a closure that escapes owns what it captures. Each cell moves into its box
// where the box takes it, the frame's release of the cell goes, and the box's release destroys the
// cell and what is in it. Each case reads through a churn() after the frame that made the closure
// is gone.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

// over an object local
function makeAdder(k: number) {
    let bump = new C(k);
    bump.v.push(k);
    return (v: number) => v + bump.x + bump.v.length;
}

// over a number parameter: a cell that owns nothing but itself
function makeScaler(k: number) {
    return (v: number) => v * k;
}

// over a number local the closure assigns
function makeCounter() {
    let n = 0;
    return () => ++n;
}

// a copy beside a cell: the copy moves in as well
function makeWithCopy(k: number) {
    const h = new C(k);
    let m = 1;
    return () => h.x * m;
}

// called through its box before it is returned
function usedThenReturned(k: number) {
    let a = new C(k);
    const f = () => a.x;
    churn();
    f();
    return f;
}

// kept by a constructor
class Holder {
    constructor(public f: () => number) {}
}

function intoField(k: number) {
    let a = new C(k);
    return new Holder(() => a.x);
}

// pushed into a global
let handlers: (() => number)[] = [];

function register(k: number) {
    let c = new C(k);
    handlers.push(() => c.x);
}

function main() {
    let t = 0;
    for (let i = 0; i < 20000; i++) {
        const n = i % 10;
        const add = makeAdder(n);
        const scale = makeScaler(n);
        const count = makeCounter();
        count();
        const copy = makeWithCopy(n);
        const used = usedThenReturned(n);
        const held = intoField(n);
        if (i < 10) {
            register(n);
        }

        churn();
        t += add(1) + scale(2) + count() + copy() + used() + held.f();
    }

    churn();
    for (const h of handlers) {
        t += h();
    }

    assert(t == 620045);
    print("done.");
}
