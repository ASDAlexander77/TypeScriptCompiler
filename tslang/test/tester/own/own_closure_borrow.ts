// -mm=own, phase 5: a closure that does not escape borrows what it captures. The frame keeps the
// cells of its variables, the box frees itself only, and a captured parameter's cell frees the
// cell and not the caller's value. Each case reads through a churn() after the point where a wrong
// release would free.
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

function apply(f: () => number) {
    churn();
    return f();
}

// a local, called through a folded `const`; the frame still owns it afterwards
function local(n: number) {
    let a = new C(n);
    const f = () => a.x + a.v.length;
    churn();
    const r = f();
    churn();
    return r + a.x;
}

// the closure goes at the end of its block, before the frame's last reads: a box that destroyed
// what it borrows would leave `a` freed here
function innerBlock(n: number) {
    let a = new C(n);
    a.v.push(n);
    let r = 0;
    {
        let f = () => a.x;
        r = f();
    }

    churn();
    return r + a.x + a.v.length - 1;
}

// a copy of a `const` beside a cell: the box borrows it too
function byValue(n: number) {
    const h = new C(n);
    let k = 0;
    const f = () => h.x + k;
    churn();
    return f() + h.x;
}

// passed as an argument
function asArgument(n: number) {
    let a = new C(n);
    const r = apply(() => a.x);
    churn();
    return r + a.x;
}

// a box per iteration; the cells go after the loop
function inLoop(n: number) {
    let a = new C(n);
    let t = 0;
    for (let k = 0; k < 3; k++) {
        const f = () => a.x + k;
        t += f();
    }

    churn();
    return t + a.x;
}

// the inner closure borrows the cell the outer one borrows
function nested(n: number) {
    let a = new C(n);
    const g = () => {
        const h = () => a.x;
        churn();
        return h();
    };
    const r = g();
    churn();
    return r + a.x;
}

// `this` and a parameter: cells that borrow their values from the caller
class D {
    constructor(public c: C) {}

    read() {
        const f = () => this.c.x;
        churn();
        return f();
    }
}

function param(c: C) {
    const f = () => c.x;
    churn();
    return f() + c.x;
}

// the frame assigns the captured local between calls, and the closure sees the new value
function frameAssigns(n: number) {
    let a = new C(n);
    const f = () => a.x;
    const first = f();
    a = new C(n + 1);
    churn();
    return first + f();
}

// the closure assigns the captured local
function closureAssigns(n: number) {
    let a = new C(n);
    const f = () => {
        a = new C(n + 2);
    };
    f();
    churn();
    return a.x;
}

// a `let` holds the closure
function letHolds(n: number) {
    let a = new C(n);
    let f = () => a.x;
    churn();
    return f();
}

// a `let` with no initializer owns nothing: it is an alias, and the closure keeps its release
function letAliases(n: number) {
    let a = new C(n);
    let f: () => number;
    f = () => a.x;
    churn();
    return f();
}

function main() {
    let t = 0;
    for (let i = 0; i < 20000; i++) {
        const n = i % 10;
        const d = new D(new C(n));
        const c = new C(n);
        t += innerBlock(n) + local(n) + byValue(n) + asArgument(n) + inLoop(n) + nested(n) + d.read() + param(c) +
             frameAssigns(n) + closureAssigns(n) + letHolds(n) + letAliases(n);
        // the captured parameters' values are still the caller's
        churn();
        t += c.x + d.c.x - 2 * n;
    }

    assert(t == 2100000);
    print("done.");
}
