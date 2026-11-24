// -mm=own: an array literal given straight to a call that holds no value at all - `po([undefined])`, `pn([null])`,
// `ln([])` - was never given back: it neither owned nor borrowed anything, so own left it to nobody (about 50 bytes a
// call; 3M calls: 110-190 MB on main, flat now). It is owned like a literal of new values, and given back after the
// call. The same program under gc and rc.
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function v(o: N | undefined | null): number {
    return o === undefined ? 1 : o === null ? 2 : o.x;
}

function po([a]: (N | undefined)[], i: number): number {
    return v(a) + i;
}

function pn(xs: (N | null)[]): number {
    return v(xs[0]);
}

function ln(xs: N[]): number {
    return xs.length + 1;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        t += po([undefined], i) + pn([null]) + ln([]) + pn([null, null]);
    }

    print(t);
    assert(t == 505500, "total");
    print("done.");
}
