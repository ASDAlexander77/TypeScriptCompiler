// -mm=own rejects: `K.w`'s one store is in a loop of top-level script code, which runs it once per
// round, so `K.w` is not immortal (spec 25): each store frees the value before it. `e` cannot be
// given the global's value, which `K.w` keeps.
class V {
    constructor(public x: number) {}
}

class K {
    static w: V;
}

for (let i = 0; i < 10; i++) {
    K.w = new V(i + 1);
}

let e = K.w;

function main() {
    print(e.x);
}
