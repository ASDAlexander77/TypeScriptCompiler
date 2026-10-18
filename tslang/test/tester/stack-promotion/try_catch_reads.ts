// Under gc and none a stack object is read in a catch - a funclet of its own on Windows - and
// survives a throw out of a loop iteration.
class Holder {
    s: string;
    constructor(s: string) {
        this.s = s;
    }
}

function check(i: number) {
    if (i % 3 == 0) {
        throw "bad";
    }
}

function f(n: number) {
    let total = 0;
    for (let i = 0; i < n; i++) {
        const p = new Holder("v" + i);
        try {
            check(i);
            total += p.s.length;
        } catch (e) {
            total += p.s.length * 100;
            continue;
        }
    }
    return total;
}

function main() {
    assert(f(7) == 3 * 200 + 4 * 2);
    print("done.");
}
