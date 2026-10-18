// A function with a try region promotes nothing in version 1.
class Holder {
    s: string;
    constructor(s: string) {
        this.s = s;
    }
}

function f(i: number) {
    const p = new Holder("v" + i);
    try {
        if (i > 100) {
            throw "big";
        }
    } catch (e) {
        return 0;
    }
    return p.s.length;
}

function main() {
    assert(f(1) == 2);
    assert(f(1000) == 0);
    print("done.");
}
