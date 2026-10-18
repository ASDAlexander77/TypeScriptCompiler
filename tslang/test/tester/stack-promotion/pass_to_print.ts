// print(p) converts p to a string through its own toString, which borrows `this`.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
    toString() {
        return "P";
    }
}

function show(i: number) {
    const p = new P(i);
    print(p);
}

function main() {
    show(1);
    print("done.");
}
