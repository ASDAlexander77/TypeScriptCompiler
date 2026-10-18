// Returning a field of a stack object: the string must survive the object's contents being given
// back at the return.
class Holder {
    s: string;
    constructor(s: string) {
        this.s = s;
    }
}

function f(i: number) {
    const p = new Holder("v" + i);
    return p.s;
}

function main() {
    let all = "";
    for (let i = 0; i < 1000; i++) {
        const s = f(i);
        all = all + s;
    }
    assert(all.length == 3890);
    assert(f(42) == "v42");
    print("done.");
}
