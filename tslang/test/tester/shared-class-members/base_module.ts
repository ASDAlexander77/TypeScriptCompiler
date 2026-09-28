// An accessor written before a method: the importer of a -shared library read the class's members
// with the accessors last, built its vtable in that order, and its `add` call ran the getter.
export class Base {
    items: number[] = [];

    get count(): number {
        return this.items.length;
    }

    add(v: number) {
        this.items.push(v);
    }

    set first(v: number) {
        this.items[0] = v;
    }

    last() {
        return this.items[this.items.length - 1];
    }
}
