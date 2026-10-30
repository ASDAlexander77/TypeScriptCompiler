// -mm=own rejects: an array literal that borrows its element (`[HG.v]`) given to a function that
// overwrites an element (`p[0] = ...`). The overwrite would free the global's value, which the array
// never owned, and the new element would be lost with the array.
class V {
    constructor(public x: number) {}
}

class H {
    v: V = new V(1);
}

const HG = new H();

function replaceFirst(p: V[]) {
    p[0] = new V(3);
    return p[0].x;
}

function main() {
    print(replaceFirst([HG.v]), HG.v.x);
}
