// -mm=own rejects: `c ? h : h2` inside a loop is a borrow of `h` or `h2` only while it is read;
// returned, it would hand the caller a local the function still owns.
class C {
    constructor(public dist: number) {}
}

function pick(n: number) {
    const h = new C(3);
    const h2 = new C(30);
    for (let i = 0; i < 4; i++) {
        let each: C = i % 2 == 0 ? h : h2;
        if (i == n) {
            return each;
        }
    }

    return new C(0);
}

function main() {
    print(pick(1).dist);
}
