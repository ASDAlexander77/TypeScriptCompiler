// An exported global is mortal under -mm=own (spec 25), even when this module never writes it:
// another module may, and its store would free the block a read here was taken to own nothing of.
// `M` (a module-level `export let`) and `MK.m` (a static of an exported class) stay mortal; `I`, the
// same `let` not exported, is immortal (test-own-global-mortal-export reads the IR).
class V {
    constructor(public x: number) {}
}

export let M = new V(2);

export class MK {
    static m = new V(3);
}

let I = new V(1);

function main() {
    print(I.x, M.x, MK.m.x);   // 1 2 3
}
