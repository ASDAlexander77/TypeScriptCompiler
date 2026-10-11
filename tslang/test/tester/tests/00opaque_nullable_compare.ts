// Comparing an Opaque with an `Opaque | null` overflowed the compiler's stack (0xC00000FD, no
// message): the comparison's lowering kept calling itself with the same operand types.

class State {
    current: Opaque | null;

    constructor() {
        this.current = null;
    }

    set(next: Opaque): boolean {
        if (next != this.current) {
            this.current = next;
            return true;
        }

        return false;
    }
}

class Item {
    id = 1;
}

function sameOpaque(a: Opaque, b: Opaque | null) {
    return a == b;
}

function sameOpaqueReversed(a: Opaque | null, b: Opaque) {
    return a == b;
}

function sameItem(a: Item, b: Item | null) {
    return a === b;
}

function differentItem(a: Item, b: Item | null) {
    return a !== b;
}

function main() {
    const s = new State();
    const h = <Opaque>new Item();
    const g = <Opaque>new Item();

    assert(s.set(h), "first set binds");
    assert(!s.set(h), "same handle again is skipped");
    assert(s.set(g), "another handle binds");

    assert(sameOpaque(h, h), "h == h");
    assert(!sameOpaque(h, g), "h == g");
    assert(!sameOpaque(h, null), "h == null");
    assert(sameOpaqueReversed(h, h), "h == h (union on the left)");

    const item = new Item();
    assert(sameItem(item, item), "item === item");
    assert(!sameItem(item, null), "item === null");
    assert(differentItem(item, new Item()), "item !== other");

    print("done.");
}
