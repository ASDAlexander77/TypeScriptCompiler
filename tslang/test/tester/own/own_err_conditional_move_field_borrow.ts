// @strict-null false
// -mm=own rejects: `f` borrows a field of `v`, `v` moves into `b` on one path only, and `f` is
// read after the paths join. Freeing `v` where the paths join, on the path that did not move it,
// would free what `f` reads (spec 30).
class W {
    constructor(public x: number) {}
}

class V {
    constructor(public w: W) {}
}

function mkv(n: number) {
    return new V(new W(n));
}

function test(c: boolean) {
    let v = mkv(1);
    const f = v.w;
    let b: V = undefined;
    if (c) {
        b = v;
    }

    print(f.x);
    if (b != undefined) {
        print(b.w.x);
    }
}

function main() {
    test(true);
    test(false);
}
