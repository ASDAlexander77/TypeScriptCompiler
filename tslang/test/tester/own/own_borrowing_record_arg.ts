// @strict-null false
// -mm=own: object literals that borrow some fields and own others (spec 32) - a parameter, a field
// read, an owning local's value, a captured parameter, beside fresh values - as call arguments, as
// `const` and `let` locals, and returned from a method. The box is a temporary or a local's like any
// fresh block; its release frees the fresh fields and leaves the borrowed ones to their owners, which
// are read again after it is gone. churn() keeps what it allocates alive, so a block freed too early
// (or freed twice) is reused before it is read.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 200; i++) {
        keep.push(new V(77777));
    }
    return keep.length;
}

interface Ray { start: V; dir: V; }

function len(r: Ray): number { return r.start.x + r.dir.x; }

// two parameters: the ray tracer's `this.traceRay({ start: pos, dir: rd }, scene, depth + 1)`
function viaParams(pos: V, rd: V) { return len({ start: pos, dir: rd }); }

// a parameter and a fresh value
function viaParamFresh(pos: V) { return len({ start: pos, dir: new V(1) }); }

class Cam {
    pos: V = new V(3);

    // a field read and a fresh value
    go() { return len({ start: this.pos, dir: new V(1) }); }

    // a `const` holding the literal, read while its sources live
    held() {
        const r: Ray = { start: this.pos, dir: new V(2) };
        churn();
        return len(r) + r.start.x;
    }

    // a `let` holding it
    heldLet() {
        let r: Ray = { start: this.pos, dir: new V(4) };
        churn();
        const n = len(r);
        return n + r.dir.x;
    }

    // returned: the result borrows `this`
    mk(): Ray { return { start: this.pos, dir: new V(5) }; }
}

// an owning local's value and a captured parameter: the ray tracer's addLight
function natural(pos: V) {
    const f = (k: number) => {
        let d = V.plus(pos, new V(k));
        const n = len({ start: pos, dir: d });
        churn();
        return n + d.x;
    };
    return f(1) + f(2);
}

function main() {
    const p = new V(10);
    const q = new V(20);
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        s += viaParams(p, q);
        s += viaParamFresh(p);
        churn();
    }

    assert(s == 41000);

    const c = new Cam();
    for (let i = 0; i < 1000; i++) {
        assert(c.go() == 4);
        assert(c.held() == 8);
        assert(c.heldLet() == 11);
        const r = c.mk();
        churn();
        assert(r.start.x == 3 && r.dir.x == 5);
    }

    assert(c.pos.x == 3);

    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        t += natural(p);
    }

    assert(t == 66000);

    // a `let` read into a field (once own_err_record_literal_let): the record borrows its value, and
    // the `let` keeps its own release
    let lp = new V(1);
    for (let i = 0; i < 1000; i++) {
        assert(len({ start: lp, dir: new V(2) }) == 3);
        churn();
    }

    assert(lp.x == 1);
    assert(p.x == 10 && q.x == 20);
    print(s, t);
    print("done.");
}
