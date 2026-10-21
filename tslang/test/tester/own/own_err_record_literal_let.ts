// -mm=own rejects: a `let` read into an object literal's field. Its slot keeps its own release
// at the end of its scope, which nothing cancels, and the record's would free the block a second
// time; only a fresh value, or a `const` folded into the value it holds, moves in (spec 27).
class V {
    constructor(public x: number) {}
}
interface Ray { start: V; dir: V; }
function len(r: Ray) { return r.start.x + r.dir.x; }
function main() {
    let p = new V(1);
    print(len({ start: p, dir: new V(2) }));
}
