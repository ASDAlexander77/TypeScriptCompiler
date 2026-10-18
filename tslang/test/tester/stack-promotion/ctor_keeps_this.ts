// The constructor stores `this` in a global: the object escapes while it is being built.
let all: P[] = [];

class P {
    x: number;
    constructor(x: number) {
        this.x = x;
        all.push(this);
    }
}

function build(i: number) {
    const p = new P(i);
    return p.x;
}

function main() {
    for (let i = 0; i < 3; i++) {
        build(i);
    }
    assert(all.length == 3);
    assert(all[1].x == 1);
    print("done.");
}
