// A `for...of` over an iterator read out of a place - a parameter, a field, a global - lends it to the
// loop, which never assigns its hidden cursor. Under -mm=own the cursor was a `let`, a second owner, so
// `for (const v of iter)` over a parameter was "a parameter, kept here". Each case churns between
// steps, so an iterator freed early reads garbage.
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

function* gen() {
    yield 1;
    yield 2;
    yield 3;
    yield 4;
}

type It = { next(): { value: any, done: boolean } };

// a parameter
function sum<TIter extends It>(iter: TIter) {
    let n = 0;
    for (const v of iter) {
        churn();
        n += v;
    }
    return n;
}

// a parameter, through `continue` and `break` of a label
function labelled<TIter extends It>(iter: TIter) {
    let s = "";
    outer: for (const v of iter) {
        for (let j = 0; j < 3; j++) {
            churn();
            if (v == 2) continue outer;
            if (v == 4) break outer;
            if (j == 1) break;
            s += v;
        }
        s += ";";
    }
    return s;
}

// a parameter the loop assigns: the cursor keeps the first, which is still the caller's
function reassigns<TIter extends It>(iter: TIter, other: TIter) {
    let n = 0;
    for (const v of iter) {
        iter = other;
        churn();
        n += v;
    }
    return n;
}

// a field
class Holder<T extends It> {
    constructor(public it: T) {}

    sum() {
        let n = 0;
        for (const v of this.it) {
            churn();
            n += v;
        }
        return n;
    }
}

// a global
const g = gen();

function sumGlobal() {
    let n = 0;
    for (const v of g) {
        churn();
        n += v;
    }
    return n;
}

function main() {
    let t = 0;
    for (let i = 0; i < 1000; i++) {
        t += sum(gen());
        const it = gen();
        t += sum(it);
        assert(labelled(gen()) == "1;3;", "labelled");
        t += reassigns(gen(), gen());
        t += new Holder(gen()).sum();
    }

    t += sumGlobal();
    assert(t == 40010, "t");
    print("done.");
}
