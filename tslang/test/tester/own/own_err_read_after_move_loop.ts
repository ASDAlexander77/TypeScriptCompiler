// @strict-null false
// -mm=own rejects: `v` moved into `best` before the loop, and the read at the top of each iteration
// is reached around the back edge after the iteration's overwrite of `best` freed it (spec 29).
class V {
    constructor(public x: number) {}
}
function main() {
    let best: V = undefined;
    let v = new V(1);
    best = v;
    let s: number = 0;
    for (let i = 0; i < 3; i++) {
        s += v.x;
        best = new V(i);
    }
    print(s, best.x);
}
