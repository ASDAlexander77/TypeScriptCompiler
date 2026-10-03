// -mm=own rejects (spec 23.3): a borrow read through one handle, used after a write through a
// handle of another static type that reaches the same block - a Shared<Derived> converts to a
// Shared<Base> - so any write through any handle ends it.
class Base {
    v = 0;
}

class Derived extends Base {
    w = 1;
}

class Box {
    items: Derived[] = [];
}

function main() {
    const a = new Shared(new Derived());
    const b: Shared<Base> = a;
    const n = a.value;
    b.value = new Base();
    const box = new Box();
    for (let i = 0; i < 100; i++) {
        const d = new Derived();
        d.w = 77;
        d.v = 66;
        box.items.push(d);
    }

    print(n.v, n.w, Shared.count(a), box.items.length);
}
