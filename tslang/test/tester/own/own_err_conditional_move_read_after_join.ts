// @strict-null false
// -mm=own rejects: `v` moves into `b` on one path only and is read after the paths join. On the
// path that moved it the read goes through `b` (spec 29), but on the other `v` still owns the
// value and must free it after the read: only a run-time record of the move could tell the two
// apart there (spec 30).
class V {
    constructor(public x: number) {}
}

function mkv(n: number) {
    return new V(n);
}

function test(c: boolean) {
    let v = mkv(1);
    let b: V = undefined;
    if (c) {
        b = v;
    }

    print(v.x);
    if (b != undefined) {
        print(b.x);
    }
}

function main() {
    test(true);
    test(false);
}
