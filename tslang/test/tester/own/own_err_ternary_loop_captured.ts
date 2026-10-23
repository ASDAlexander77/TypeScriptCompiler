// -mm=own rejects: `c ? h : h2` inside a loop is a borrow of `h` or `h2` only while it is read;
// captured by a closure that escapes, it would move one of them into another closure each turn.
interface HasDist {
    dist: number;
}

class C implements HasDist {
    constructor(public dist: number) {}
}

let readers: (() => number)[] = [];

function register() {
    const h: HasDist = new C(3);
    const h2: HasDist = new C(30);
    for (let i = 0; i < 4; i++) {
        const each: HasDist = i % 2 == 0 ? h : h2;
        readers.push(() => each.dist);
    }
}

function main() {
    register();
    print(readers.length);
}
