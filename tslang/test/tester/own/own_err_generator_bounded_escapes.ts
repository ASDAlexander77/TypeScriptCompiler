// -mm=own, phase 7b rejects: the generator borrows `a`, which dies with `make`'s frame.
function* each(a: number[]) {
    for (const v of a) {
        yield v;
    }
}

function make() {
    const a: number[] = [1, 2];
    return each(a);
}

function main() {
    for (const v of make()) {
        print(v);
    }
}
