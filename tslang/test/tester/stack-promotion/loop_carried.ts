// The object of iteration 3 is still used after the loop: the one stack slot would be reused, so
// it goes on the heap.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function f() {
    let keep: P = new P(-1);
    for (let i = 0; i < 10; i++) {
        const p = new P(i);
        if (i == 3) {
            keep = p;
        }
    }
    return keep.x;
}

function main() {
    assert(f() == 3);
    print("done.");
}
