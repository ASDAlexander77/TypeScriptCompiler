// `<Opaque>x` under -mm=own borrows x (spec 92, user ruling 2026-11-16): the raw pointer owns nothing,
// and the object stays its owner's - a local, a `const`, a field. Values are made at run time, in a
// loop, so a double release or a leak shows.
class Item {
    constructor(public id: number, public name: string) {}
}

class State {
    current: Opaque | null = null;

    set(next: Opaque) {
        if (next != this.current) {
            this.current = next;
            return true;
        }

        return false;
    }
}

class Holder {
    item = new Item(0, "h");
}

function main() {
    const s = new State();
    const h = new Holder();
    let changes = 0;
    for (let i = 0; i < 3000; i++) {
        const item = new Item(i, "i" + i);
        const p = <Opaque>item;
        if (s.set(p)) {
            changes++;
        }

        assert(!s.set(p), "the same object again");
        h.item = new Item(i, "f" + i);
        if (s.set(<Opaque>h.item)) {
            changes++;
        }

        assert(item.name == "i" + i && h.item.name == "f" + i, "the objects are still their owners'");
        s.current = null;
    }

    assert(changes == 6000, "changes");
    print("done.");
}
