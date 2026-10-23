// @strict-null false
// -mm=own rejects: `v` moves into `b` on one path only, `b` is overwritten after the paths join,
// and `v` is read after that - through a freed block on the path that moved it (spec 30).
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

    b = undefined;
    print(v.x);
}

function main() {
    test(true);
    test(false);
}
