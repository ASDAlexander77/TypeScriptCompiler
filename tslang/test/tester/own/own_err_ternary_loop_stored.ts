// -mm=own rejects: `c ? h : h2` inside a loop is a borrow of `h` or `h2` only while it is read;
// pushed, it would give one of them a new owner each iteration.
class C {
    constructor(public dist: number) {}
}

function main() {
    const h = new C(3);
    const h2 = new C(30);
    const kept: C[] = [];
    for (let i = 0; i < 4; i++) {
        const each: C = i % 2 == 0 ? h : h2;
        kept.push(each);
    }

    print(kept.length);
}
