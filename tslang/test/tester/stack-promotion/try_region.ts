// A function with a try region promotes nothing under rc and own in version 1, where the unwind
// path would have to give back what the object owns; under gc and none it promotes.
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
