// A class with a string field built every iteration: on the stack, and the string is given back
// each time under rc and own.
class Named {
    name: string;
    n: number;
    constructor(name: string, n: number) {
        this.name = name;
        this.n = n;
    }
}

function total(count: number) {
    let t: number = 0;
    for (let i = 0; i < count; i++) {
        const p = new Named("item" + i, i);
        t += p.name.length + p.n;
    }
    return t;
}

function main() {
    assert(total(10000) == 78890 + 49995000);
    print("done.");
}
