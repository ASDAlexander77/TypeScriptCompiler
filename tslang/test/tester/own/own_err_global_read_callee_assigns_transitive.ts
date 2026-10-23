// -mm=own rejects: `c` borrows what `G` holds, and `step` calls `reset`, which gives `G` a new value
// and frees that object while `c` still reads it.
class V {
    constructor(public x: number) {}
}

let G = new V(0);

function reset() {
    G = new V(1);
}

function step() {
    reset();
}

function main() {
    const c = G;
    step();
    print(c.x);
}
