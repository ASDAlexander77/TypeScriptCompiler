// -mm=own rejects: `add` returns its `this`, here the object `SHARED` owns, so `ALIAS` would be a
// second owner of it. A chain on a new object (`new Bag().add("q")`) is the global's own (TSLANG 85).
class Bag {
    items: string[] = [];

    add(s: string): this {
        this.items.push(s);
        return this;
    }
}

const SHARED = new Bag();
const ALIAS = SHARED.add("q");

function main() {
    print(ALIAS.items.length);
}
