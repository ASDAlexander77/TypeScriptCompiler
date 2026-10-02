// -mm=own, phase 7 rejects: `i` is `raw`'s block seen through an interface, and both would own it.
interface I {
    x: number;
    get(): number;
}

function main() {
    let raw = {
        x: 4.0,
        get(): number {
            return this.x;
        },
    };
    let i = <I>raw;
    print(i.get());
}
