// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own: the loop reads the very array the call empties - the bag the call is given is the chain's root (TSLANG
// 216). An error.
class Bag {
    names: string[] = [];
    add(name: string) { this.names.push("" + name); }
    clear() { this.names.splice(0, this.names.length); }
}

function main() {
    const bag = new Bag();
    bag.add("a" + 1);
    for (const it of bag.names) {
        bag.clear();
        print(it);
    }
    print("done.");
}
