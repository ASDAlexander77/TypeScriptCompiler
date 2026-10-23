// @strict-null false
// -mm=own rejects: `v` moved into `best`, and a later loop overwrites `best` before `v` is read
// after it (spec 29).
class V {
    constructor(public x: number) {}
}
function main() {
    let best: V = undefined;
    let v = new V(1);
    best = v;
    for (let i = 0; i < 3; i++) {
        best = new V(i);
    }
    print(v.x, best.x);
}
