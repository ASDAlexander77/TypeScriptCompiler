// -mm=own: a field moves out of the object an owning `let` holds only when the local was given `new`'s
// or a call's own result and nothing sees the object after the move (spec 65); taken in a loop that does not give the local anew: the second turn finds the field empty.
class Item {
    constructor(public n: number) {}
}

class Holder {
    value = new Item(3);
}

const shared = new Holder();

function take() {
    const out: Item[] = [];
    let r = new Holder();
    for (let i = 0; i < 3; i++) {
        out.push(r.value);
    }

    return out;
}

function main() {
    print(take().length);
}
