// @strict-null false
// -mm=own rejects: `v` moves into `best` on one path only inside a catch clause. The edge that
// did not move it is inside the exception handler, where -mm=own places no release of its own
// (spec 30).
class V {
    constructor(public x: number) {}
}

function mkt(n: number) {
    if (n > 5) {
        throw n;
    }

    return new V(n);
}

function test(c: boolean) {
    let best: V = undefined;
    let v = mkt(1);
    try {
        mkt(9);
    } catch {
        if (c) {
            best = v;
        }
    }

    print(best != undefined);
}

function main() {
    test(true);
    test(false);
}
