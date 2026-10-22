// `reduce` takes its accumulator (spec 33): the initial value moves into the generated
// `___array_reduce`, each callback is given the accumulator to keep, and its result is the next
// one. A function that returns a parameter on some paths and a fresh value on others keeps that
// parameter: the path that returns it moves it out, every other path releases it exactly once -
// an early return, a `break` out of a loop, a return from inside a loop or a `try` with a
// `finally`, a `throw`. A callback that only borrows the accumulator releases it.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) {
        return new V(a.x + b.x);
    }
}

class K {
    static zero = new V(0);
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new V(999));
    }
}

// returns acc on one path
function pick(acc: V, v: V) {
    return v.x > 2 ? acc : V.plus(acc, v);
}

function early(a: V, n: number) {
    if (n > 5) {
        return a;
    }

    for (let i = 0; i < n; i++) {
        if (i == 3) {
            break;
        }
    }

    return new V(a.x + n);
}

function fromLoop(a: V, n: number) {
    for (let i = 0; i < n; i++) {
        if (i == 2) {
            return a;
        }
    }

    return new V(a.x * 10);
}

let finallies: number = 0;

function guarded(a: V, n: number) {
    try {
        if (n > 1) {
            return a;
        }
    } finally {
        finallies++;
    }

    return new V(a.x + 100);
}

function thrower(a: V, n: number) {
    if (n < 0) {
        throw "negative";
    }

    if (n == 0) {
        return a;
    }

    return new V(a.x + 1);
}

function main() {
    const a = [new V(1), new V(2), new V(3)];
    const r = a.reduce((acc: V, v: V) => V.plus(acc, v), new V(0));
    const q = a.reduce(pick, K.zero);
    churn();
    print(r.x, q.x); // 6 3

    // every element above 2: pick returns the immortal start value itself, which no release frees
    const z = [new V(5), new V(7)].reduce(pick, K.zero);
    churn();
    print(z.x, K.zero.x); // 0 0

    let total: number = 0;
    let caught = 0;
    for (let i = 0; i < 20000; i++) {
        const n = i % 9;
        const e = early(new V(i % 10), n);
        const f = fromLoop(new V(1), n);
        const g = guarded(new V(2), n);
        let t: number = 0;
        try {
            const h = thrower(new V(3), n - 1);
            t = h.x;
        } catch (err) {
            caught++;
        }

        churn();
        total += e.x + f.x + g.x + t;
    }

    print(total, caught, finallies);
    assert(r.x == 6 && q.x == 3 && z.x == 0 && K.zero.x == 0);
    print("done.");
}
