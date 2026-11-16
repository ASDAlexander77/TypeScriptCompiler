// -mm=own rejects: `v` moves into the first method value stored in `h`, and is given to a second.
class V {
    constructor(public x: number) {}

    get() {
        return this.x;
    }
}

class H {
    f: () => number = () => 0;
    g: () => number = () => 0;
}

function store(h: H, i: number) {
    const v = new V(i);
    h.f = v.get;
    h.g = v.get;
}

function main() {
    const h = new H();
    store(h, 3);
    print(h.f() + h.g());
}
