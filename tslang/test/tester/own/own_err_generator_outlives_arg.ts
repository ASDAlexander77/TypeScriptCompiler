// -mm=own, phase 7b rejects: `it` borrows `a`, and `a` is assigned while `it` is still used.
function* each(a: number[]) {
    for (const v of a) {
        yield v;
    }
}

function main() {
    let a: number[] = [1, 2];
    const it = each(a);
    a = [3.5];
    print(it.next().value);
}
