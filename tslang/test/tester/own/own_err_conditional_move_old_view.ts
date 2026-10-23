// @strict-null false
// -mm=own rejects: a bound method of `v` and an interface view of it are used after `v`, moved into
// `b` on one path only, is given a new value. The old value, released where the paths join on the
// path that did not move it, is what they read; the joined point is long past (spec 30).
class W {
    constructor(public x: number) {}
}

class V {
    constructor(public w: W) {}
    get() {
        return this.w.x;
    }
}

interface I {
    w: W;
}

function mkv(n: number) {
    return new V(new W(n));
}

function churn() {
    let s: number = 0;
    const a: V[] = [];
    for (let i = 0; i < 64; i++) {
        a.push(mkv(1000 + i));
    }

    for (const e of a) {
        s += e.w.x;
    }

    return s;
}

function bound(c: boolean) {
    let b: V = undefined;
    let v = mkv(1);
    const m = v.get;
    if (c) {
        b = v;
    }

    v = mkv(2);
    churn();
    return m() + v.w.x;
}

function view(c: boolean) {
    let b: V = undefined;
    let v = mkv(1);
    const i: I = v;
    if (c) {
        b = v;
    }

    v = mkv(2);
    churn();
    return i.w.x + v.w.x;
}

function main() {
    print(bound(true), bound(false), view(true), view(false));
}
