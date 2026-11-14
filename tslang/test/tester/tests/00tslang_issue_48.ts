// TSLANG_ISSUES 48 (-mm=own): in an exported class that isn't @final, any `this.method()` may be
// overridden in another module by a method that overwrites any field, so a borrow of a field held
// across it was refused. `@final` on the method says no subclass overrides it: its calls run it, and
// -mm=own goes by its body.

export class Item {
    constructor(public n: number) {}
}

export class Base {
    items: Item[] = [new Item(1), new Item(2)];
    scale: number = 2.0;

    @final factor(): number {
        return this.scale;
    }

    // a method that is not final, overridden below: still called through the vtable
    bonus(): number {
        return 0;
    }
}

export class App extends Base {
    total(): number {
        const item = this.items[1];
        const f = this.factor();
        return item.n * f + this.bonus();
    }

    bonus(): number {
        return 1;
    }
}

function main() {
    const app = new App();
    print(app.total());
    assert(app.total() === 5);
    const base: Base = app;
    assert(base.factor() === 2 && base.bonus() === 1);
    print("done.");
}
