// -mm=own: a call's drop facts name the argument whose object owns what it overwrites (TSLANG 216), but two values the
// function does not own may be one - `fill(b, b.names)` - so a borrow of an element of `items` still ends at
// `bag.clear()`. An error.
class Bag {
    names: string[] = [];
    add(name: string) { this.names.push("" + name); }
    clear() { this.names.splice(0, this.names.length); }
}

function fill(bag: Bag, items: string[]) {
    for (const it of items) {
        bag.clear();
        bag.add(it);
    }
}

function main() {
    const b = new Bag();
    fill(b, ["a" + 1]);
}
