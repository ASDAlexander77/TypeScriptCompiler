// -mm=own rejects: the returned literal borrows the value of a local (spec 32), which the function
// frees as it returns; only what a parameter or `this` holds may be lent to the caller.
class V {
    constructor(public x: number) {}
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 200; i++) {
        keep.push(new V(77777));
    }
    return keep.length;
}

interface Ray { start: V; dir: V; }

function mk(): Ray {
    let d = new V(1);
    return { start: d, dir: new V(2) };
}

function main() {
    const r = mk();
    churn();
    print(r.start.x);
}
