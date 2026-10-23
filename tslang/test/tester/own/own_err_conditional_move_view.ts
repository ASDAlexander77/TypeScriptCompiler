// @strict-null false
// -mm=own rejects: `i` is `v` seen through an interface, `v` moves into `b` on one path only, and
// `i` is read after the paths join. Freeing `v` where the paths join, on the path that did not move
// it, would free what `i` reads (spec 30). Only the walk of what is read out of `v` sees it.
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

function test(c: boolean) {
    let b: V = undefined;
    let v = mkv(1);
    const i: I = v;
    if (c) {
        b = v;
    }

    churn();
    return i.w.x;
}

function main() {
    print(test(true), test(false));
}
