// -mm=own: a field moves out of the object an owning `let` holds only when the local was given `new`'s
// or a call's own result and nothing sees the object after the move (spec 65); a closure that reads the local, called after the take.
class Item {
    constructor(public n: number) {}
}

class Holder {
    value = new Item(3);
}

const shared = new Holder();

function take() {
    let r = new Holder();
    const read = () => r.value.n;
    const v = r.value;
    print(read());
    return v;
}

function main() {
    print(take().n);
}
