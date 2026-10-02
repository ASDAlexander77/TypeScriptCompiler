// -mm=own, phase 7 rejects: each call is given `o` through an interface of its own, and rc counts
// each one; with no count, both would destroy `o`'s block.
interface I {
    x: number;
    get(): number;
}

function use(i: I) {
    return i.get();
}

function main() {
    const o = {
        x: 4.0,
        get(): number {
            return this.x;
        },
    };
    print(use(o));
    print(use(o));
}
