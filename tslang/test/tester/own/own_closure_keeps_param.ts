// A closure the function keeps - stored into a field, a global, an array - owns the parameters it
// captures, so the function keeps them and its callers give them up (spec 76): a callback captured
// in its cell, a class instance captured by copy, with `@keeps` saying so and without. A closure
// stored into a field of the object it captures, from an arrow that captures something of its own,
// borrows that object as one from an arrow that captures nothing does (spec 75). Each step churns, so
// anything freed too early reads garbage; every model runs it.
class Junk {
    constructor(public n: number) {}
}

function churn() {
    const junk: Junk[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Junk(999));
}

class Item {
    constructor(public n: number) {}
}

// the callback's cell moves into the closure
class Holder {
    run: () => int;

    constructor(body: (n: int) => int) {
        this.run = (): int => body(2);
    }
}

// the same, as another module would be told it
@final
export class Declared {
    run: () => int;

    @keeps("body")
    constructor(body: (n: int) => int) {
        this.run = (): int => body(3);
    }
}

// a class instance captured by copy, by a closure kept in a field of `this`
class Counter {
    n: number = 1;
    get = () => 0;

    setup(item: Item) {
        this.get = () => item.n * 10;
    }
}

// ...and by one pushed into an array
const tasks: (() => number)[] = [];

function schedule(item: Item) {
    tasks.push(() => item.n + 100);
}

class Scope {
    handler: () => void;
    count: int;

    constructor() {
        this.handler = () => {};
        this.count = 0;
    }
}

function run(body: (self: Scope) => void): int {
    const scope = new Scope();
    body(scope);
    churn();
    scope.handler();
    scope.handler();
    return scope.count;
}

function main() {
    const h = new Holder((n: int): int => n * 21);
    churn();
    assert(h.run() == 42, "callback kept by a closure");

    const d = new Declared((n: int): int => n * 14);
    churn();
    assert(d.run() == 42, "@keeps of a callback a closure keeps");

    const c = new Counter();
    c.setup(new Item(4));
    churn();
    assert(c.get() == 40, "instance kept by a closure in a field");

    schedule(new Item(5));
    churn();
    assert(tasks[0]() == 105, "instance kept by a closure in an array");

    // an arrow that captures nothing fills a field of what it is given
    assert(run((self: Scope) => {
        self.handler = () => {
            self.count++;
        };
    }) == 2, "back-reference from an arrow capturing nothing");

    // ...and one that captures a variable of its own
    let seen = 0;
    const step = new Item(3);
    assert(run((self: Scope) => {
        seen += step.n;
        self.handler = () => {
            self.count++;
        };
    }) == 2, "back-reference from an arrow that captures");
    assert(seen == 3, "the capturing arrow ran");

    print("done.");
}
