// A function that keeps its parameter: the object outlives the call, so it goes on the heap.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

let kept: P[] = [];

function keep(p: P) {
    kept.push(p);
}

function build(i: number) {
    const p = new P(i);
    keep(p);
}

function overwrite(i: number) {
    const a = new P(i * 100);
    const b = new P(i * 1000);
    return a.x + b.x;
}

function main() {
    for (let i = 0; i < 3; i++) {
        build(i);
        overwrite(i + 7);
    }
    assert(kept.length == 3);
    assert(kept[0].x == 0);
    assert(kept[1].x == 1);
    assert(kept[2].x == 2);
    print("done.");
}
