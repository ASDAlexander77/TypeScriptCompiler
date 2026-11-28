// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// TSLANG_ISSUES 48: a borrow of a field held across `this.factor()`, a method of an exported class
// another module may override. The error says why the call may release the field, and that
// `@final` says it does not.
export class Item {
    constructor(public n: number) {}
}

export class Base {
    items: Item[] = [new Item(1), new Item(2)];
    scale: number = 2.0;
    factor(): number { return this.scale; }
}

export class App extends Base {
    total(): number {
        const item = this.items[1];
        const f = this.factor();
        return item.n * f;
    }
}

function main() { print(new App().total()); }
