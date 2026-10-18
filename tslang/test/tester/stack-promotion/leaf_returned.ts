// Returned, so it outlives the function: heap.
class P {
    x: number;
    y: number;
}

function make(i: number) {
    const p = new P();
    p.x = i;
    return p;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        t += make(i).x;
    }
    assert(t == 499500);
    print("done.");
}
