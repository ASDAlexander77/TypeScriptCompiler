// -mm=own: a field moves out of an object a `const` names only when nothing sees the object after
// the move but its release (spec 64); a method of the object called after its field is taken.
class Item {
    constructor(public n: number) {}
}

class Holder {
    value = new Item(3);
    show() {
        return this.value.n;
    }
}

const kept: Holder[] = [];

function keep(r: Holder) {
    kept.push(r);
}

function same(r: Holder) {
    return r;
}

function take() {
    const r = new Holder();
    const v = r.value;
    print(r.show());
    return v;
}

function main() {
    print(take().n);
}
