// -mm=own: a field moves out of the object an owning `let` holds only when the local was given `new`'s
// or a call's own result and nothing sees the object after the move (spec 65); given a global, which it does not own: the global would be left with an empty field.
class Item {
    constructor(public n: number) {}
}

class Holder {
    value = new Item(3);
}

const shared = new Holder();

function take() {
    let r = shared;
    return r.value;
}

function main() {
    print(take().n, shared.value.n);
}
