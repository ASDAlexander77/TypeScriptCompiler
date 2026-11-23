// -mm=own --own-fallback=count: a view of an array read after something that may free one of its
// elements stays an error with the flag. The view's report commits rc's reference for the local
// as a borrow (fieldBorrowRetains), so the value it is about cannot keep rc's ops whole: it is not
// counted (spec 91; own rc fallback design, section 3).
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
    print("done.");
}
