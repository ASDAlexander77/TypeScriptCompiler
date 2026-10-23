// -mm=own rejects: `c` borrows what `G` holds, and `reset` gives `G` a new value, which frees that
// object while `c` still reads it.
class V {
    constructor(public x: number) {}
}

let G = new V(0);

function reset() {
    G = new V(1);
}

function main() {
    const c = G;
    reset();
    print(c.x);
}
