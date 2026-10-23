// @strict-null false
// -mm=own rejects: `v` is declared outside the loop and moved inside it on some iterations, and
// the next iteration reads it without giving it a new value (spec 30).
class V {
    constructor(public x: number) {}
}

function mkv(n: number) {
    return new V(n);
}

function main() {
    let v = mkv(1);
    let best: V = undefined;
    let s: number = 0;
    for (let i = 0; i < 4; i++) {
        s += v.x;
        if (i % 2 == 0) {
            best = v;
        }
    }

    print(s, best.x);
}
