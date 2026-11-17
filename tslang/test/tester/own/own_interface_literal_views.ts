// -mm=own: an object literal given to interface parameters several times (`use(o); use(o)`), and
// called through an interface view of it (`(<Something>o).toString()`): each argument is a view
// made for its call, which only borrows the literal's block; the block is freed once, after the last.
interface Something {
    r: number;
    toString: () => string;
}

let total: number = 0;

function use(iface: Something) {
    total += iface.r;
}

function twice(i: number) {
    const something = { r: i, toString() { return "Hello " + this.r; } };
    use(something);
    use(something);
}

function viewAndCalls(i: number) {
    const something = { r: i, toString() { return "Hello " + this.r; } };
    const iface = <Something>something;
    assert(iface.r == i);
    use(something);
    use(something);
    use(something);
}

function main() {
    for (let i = 0; i < 1000; i++) {
        twice(1);
        viewAndCalls(2);
    }

    assert(total == 1000 * (2 + 6), "total");
    print("done.");
}
