// -mm=own rejects: `mk` is exported, so a caller in another module, which cannot see that its result
// borrows `p` (spec 32), would not bound it by `p`; it has to say so with `@lends`.
export class V {
    constructor(public x: number) {}
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 200; i++) {
        keep.push(new V(77777));
    }
    return keep.length;
}

export interface Ray { start: V; dir: V; }

export function mk(p: V): Ray {
    return { start: p, dir: new V(2) };
}

function main() {
    const r = mk(new V(1));
    churn();
    print(r.start.x);
}
