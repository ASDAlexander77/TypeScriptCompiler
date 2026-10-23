// `a[i] = f()` stores into the array once `f()` has run, even when `f` grows the array. The
// element's address was taken before the right side ran, and growing moved the elements to a new
// block: the store went into the one given back. The value was lost under every model, and under
// rc a freed block was written to. The array and the index are still the ones the left side
// evaluated first, as in JavaScript.
class V {
    constructor(public x: number) {}
}

class H {
    a: number[] = [1];
    items: V[] = [new V(1)];
}

function growNumbers(a: number[]) {
    for (let i = 0; i < 16; i++) a.push(i);
    return 7;
}

function growItems(items: V[]) {
    for (let i = 0; i < 16; i++) items.push(new V(100 + i));
    return new V(8);
}

function numbers() {
    const a: number[] = [1];
    a[0] = growNumbers(a);
    return a.length == 17 && a[0] == 7 && a[1] == 0;
}

function objects() {
    const items = [new V(1)];
    items[0] = growItems(items);
    return items.length == 17 && items[0].x == 8 && items[16].x == 115;
}

function compound() {
    const a: number[] = [1];
    a[0] += growNumbers(a);
    return a[0] == 8;
}

function field() {
    const h = new H();
    h.a[0] = growNumbers(h.a);
    h.items[0] = growItems(h.items);
    return h.a[0] == 7 && h.items[0].x == 8;
}

function otherArray() {
    let a: number[] = [1, 2];
    const old = a;
    a[1] = (a = [5], 9);
    return old[1] == 9 && a.length == 1 && a[0] == 5;
}

function main() {
    for (let i = 0; i < 3; i++) {
        assert(numbers(), "a number stored after the call grows the array");
        assert(objects(), "an object stored after the call grows the array");
        assert(compound(), "a compound assignment after the call grows the array");
        assert(field(), "an element of a field stored after the call grows the array");
        assert(otherArray(), "the left side's array, though the right side assigns the variable");
    }

    print("done.");
}
