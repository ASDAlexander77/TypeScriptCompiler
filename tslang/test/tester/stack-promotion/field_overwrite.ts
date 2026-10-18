// A field overwritten several times: every old value is given back once, the last one at exit.
class Holder {
    s: string;
    constructor(s: string) {
        this.s = s;
    }
}

function f(i: number) {
    const p = new Holder("a" + i);
    print(p.s);
    p.s = "b" + i;
    print(p.s);
    p.s = "c" + i;
    print(p.s);
    return p.s.length;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 3; i++) {
        t += f(i);
        const after = "x" + i + "y" + i;
        print(after);
    }
    assert(t == 6);
    print("done.");
}
