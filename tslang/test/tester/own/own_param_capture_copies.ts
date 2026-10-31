// A parameter nothing assigns is captured by copy, not moved into a heap cell: the ray tracer's
// `addLight` shape, a callback over the method's parameters given to `reduce`, called with values
// read out of fields. A parameter the body or a closure assigns stays in a cell, so every reader
// sees the assignment. Each call churns, so a parameter freed too early reads garbage; every model
// runs it.
class Item {
    constructor(public n: number) {}
}

interface Thing {
    n: number;
}

class Light implements Thing {
    constructor(public n: number) {}
}

interface Scene {
    lights: Light[];
    thing: Thing;
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

class Tracer {
    k = 2;

    // `this`, an interface, a class and a number, all captured by a closure that does not escape
    natural(thing: Thing, item: Item, scene: Scene, scale: number) {
        const add = (sum: number, light: Light) => {
            churn();
            return sum + light.n * thing.n + item.n * scale + scene.lights.length + this.k;
        };
        return scene.lights.reduce(add, 0);
    }

    // a closure in a closure over the same parameter
    nested(item: Item, xs: number[]) {
        const outer = (x: number) => {
            const inner = (y: number) => y + item.n;
            return inner(x) * 2;
        };
        let s = 0;
        for (const x of xs) s += outer(x);
        return s;
    }

    trace(scene: Scene, item: Item) {
        // the arguments are read out of fields: the callee borrows them
        return this.natural(scene.thing, item, scene, 3) + this.natural(scene.lights[0], item, scene, 1);
    }
}

// assigned by the closure: the frame sees it
function assignedInClosure(n: number, xs: number[]) {
    const bump = (x: number) => {
        n = n + x;
    };
    for (const x of xs) bump(x);
    return n;
}

// assigned by the body after the closure is made: the closure sees it
function assignedAfter(n: number) {
    const read = () => n;
    n++;
    const first = read();
    --n;
    n += 10;
    return first * 100 + read();
}

function main() {
    const scene: Scene = { lights: [new Light(1), new Light(2), new Light(3)], thing: new Light(5) };
    const tracer = new Tracer();
    const item = new Item(7);

    // natural(thing 5) = (1+2+3)*5 + 3*(7*3 + 3 + 2) = 30 + 78 = 108
    // natural(thing 1) = (1+2+3)*1 + 3*(7*1 + 3 + 2) = 6 + 36 = 42
    const traced = tracer.trace(scene, item);
    assert(traced == 150, "captured parameters read after churn");
    assert(item.n == 7 && scene.thing.n == 5, "the caller's values are still there");

    // (1+7)*2 + (2+7)*2 + (3+7)*2 = 54
    assert(tracer.nested(item, [1, 2, 3]) == 54, "a closure in a closure");

    assert(assignedInClosure(1, [1, 2, 3]) == 7, "assigned by the closure");
    assert(assignedAfter(4) == 514, "assigned by the body");

    print("done.");
}
