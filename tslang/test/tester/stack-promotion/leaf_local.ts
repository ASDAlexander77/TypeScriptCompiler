// A class with no constructor, built and dropped every iteration: its fields are only read and
// written, so it goes on the stack.
class P {
    x: number;
    y: number;
}

function sum(n: number) {
    let t: number = 0;
    for (let i = 0; i < n; i++) {
        const p = new P();
        p.x = i;
        p.y = i + 1;
        t += p.x + p.y;
    }
    return t;
}

function main() {
    assert(sum(1000) == 1000000);
    print("done.");
}
