// An allocation on a path that is not taken: the slot is still given back at the return, so it must
// hold nothing then.
class Holder {
    s: string;
    constructor(s: string) {
        this.s = s;
    }
}

function f(c: boolean) {
    if (c) {
        const p = new Holder("s" + 1);
        assert(p.s == "s1");
    }
    return 1;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        t += f(false);
        t += f(true);
    }
    assert(t == 2000);
    print("done.");
}
