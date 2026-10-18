// Seen through an interface: version 1 treats the conversion as an escape.
interface HasX {
    x: number;
}

class P implements HasX {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function f(i: number) {
    const p = new P(i);
    const h: HasX = p;
    return h.x;
}

function main() {
    assert(f(4) == 4);
    print("done.");
}
