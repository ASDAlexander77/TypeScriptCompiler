// A method that stores `this` somewhere: calling it lets the object out.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
    register() {
        registry.push(this);
    }
}

let registry: P[] = [];

function build(i: number) {
    const p = new P(i);
    p.register();
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
    assert(registry.length == 3);
    assert(registry[2].x == 2);
    print("done.");
}
