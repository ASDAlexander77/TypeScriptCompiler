// Converted to any: version 1 treats the conversion as an escape.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function f(i: number) {
    const p = new P(i);
    const a: any = p;
    return typeof a == "class" ? 1 : 0;
}

function main() {
    assert(f(4) == 1);
    print("done.");
}
