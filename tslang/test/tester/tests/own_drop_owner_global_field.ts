// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own: a field given a value it does not make (`set(a)`, here an immortal `const` global) may be another's: what
// the call empties may be `G` itself, a bag made here or not (TSLANG 216). An error.
const G: string[] = ["a", "b"];

class Bag {
    names: string[] = [];
    set(a: string[]) { this.names = a; }
    clear() { this.names.splice(0, this.names.length); }
}

function main() {
    const b = new Bag();
    b.set(G);
    for (const it of G) {
        b.clear();
        print(it);
    }
    print("done.");
}
