// -mm=own rejects: the view borrows the elements of `a`, and `a.pop()` frees the last of them while
// the view still reads it (spec 91). A view is only a borrow while nothing that may free an element
// runs between it and its uses.
class Item {
    constructor(public v: number) {}
}

function last(a: Item[]) {
    const w = a.view(0, 2);
    a.pop();
    return w[2].v;
}

function main() {
    const items = [new Item(1), new Item(2), new Item(3)];
    print(last(items));
}
