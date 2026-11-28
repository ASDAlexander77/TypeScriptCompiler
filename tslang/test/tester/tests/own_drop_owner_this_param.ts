// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own: `this` and a parameter may be one (`b.fill(b.names)`): a borrow of an element of `items` ends at
// `this.clear()` (TSLANG 216). An error.
class Bag {
    names: string[] = [];
    add(name: string) { this.names.push("" + name); }
    clear() { this.names.splice(0, this.names.length); }
    fill(items: string[]) {
        for (const it of items) {
            this.clear();
            this.add(it);
        }
    }
}

function main() {
    const b = new Bag();
    b.fill(["a" + 1]);
    print("done.");
}
