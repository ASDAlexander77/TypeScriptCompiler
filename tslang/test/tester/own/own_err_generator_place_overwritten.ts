// -mm=own, phase 7b rejects: the generator borrows the array `h.items` holds, and the field is
// assigned while the generator is still used.
class H {
    items: number[] = [1, 2, 3];
}

function* each(a: number[]) {
    for (const v of a) {
        yield v;
    }
}

function main() {
    const h = new H();
    const it = each(h.items);
    h.items = [9];
    print(it.next().value);
}
