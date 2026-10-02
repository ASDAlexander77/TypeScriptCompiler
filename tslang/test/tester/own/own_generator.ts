// -mm=own, phase 7: a generator's state object is an ordinary owned block. Its maker makes it and
// returns it, the caller owns it, and its release destroys whatever the generator's locals hold.
// Each case runs many times, so a state object that is never freed shows in measure.ps1, and
// reads through a churn() between yields, so a local freed too early reads garbage.
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

function* evens() {
    yield 2;
    yield 4;
}

// what the generators below start from: a global, which a generator reads without capturing it
let seed = 0;

// a local that owns a block: a field of the state object, destroyed with it
function* fromLocal() {
    const k = seed;
    const c = new C(k);
    c.v.push(k);
    yield c.x;
    yield c.x + c.v.length;
}

// a local assigned between yields: the old value is released, the state keeps the new one
function* reassigned() {
    const k = seed;
    let c = new C(k);
    yield c.x;
    c = new C(k + 1);
    yield c.x;
}

// yields fresh blocks: each moves into the `{value, done}` result, which the caller owns
function* objects() {
    const k = seed;
    yield new C(k);
    yield new C(k + 1);
}

// a string made in another function: a temporary made in front of a `yield` in the generator's
// own body is never released, under rc as well (OwnedReturnConsumptionPass does not release past a
// resume point)
function label(i: number) {
    return "s" + i;
}

function* strings() {
    for (let i = 0; i < 3; i++) {
        yield label(i);
    }
}

// parameters holding numbers: their cells move into the box the state object owns
function* range(from: number, to: number) {
    for (let i = from; i < to; i++) {
        yield i;
    }
}

// an object literal with a method: the same made block, seen as its object
function makeCounter(start: number) {
    return {
        n: start,
        next() {
            this.n++;
            return this.n;
        },
    };
}

function main() {
    let t = 0;
    for (let i = 0; i < 20000; i++) {
        // through for...of
        for (const v of evens()) {
            t += v;
        }

        // by hand, the generator kept in a local
        const it = evens();
        let r = it.next();
        while (!r.done) {
            t += r.value;
            r = it.next();
        }

        const n = i % 10;
        seed = n;
        for (const v of fromLocal()) {
            churn();
            t += v;
        }

        for (const v of reassigned()) {
            churn();
            t += v;
        }

        for (const v of objects()) {
            churn();
            t += v.x + v.v.length;
        }

        let text = "";
        for (const s of strings()) {
            churn();
            text = text + s;
        }

        t += text.length;

        // by hand, each result kept across a churn() before it is read
        const os = objects();
        let o = os.next();
        churn();
        t += o.value.x;
        o = os.next();
        churn();
        t += o.value.x;

        for (const v of range(n, n + 3)) {
            churn();
            t += v;
        }

        // given up after the first value: the state object still destroys its local
        const half = fromLocal();
        t += half.next().value;

        const c = makeCounter(i % 10);
        t += c.next() + c.next();
    }

    assert(t == 1820000, "t");
    print("done.");
}
