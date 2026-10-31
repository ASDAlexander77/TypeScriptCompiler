// -mm=own rejects: `bag` is lent to `forEach`, which walks the array `bag` holds while it runs a callback
// that gives `bag` a new value - freeing the array under the walk.
class Box {
    constructor(public n: number) {}
}

class Bag {
    items: Box[] = [new Box(1), new Box(2)];

    forEach(f: (b: Box) => void) {
        for (const b of this.items) f(b);
    }
}

function main() {
    let bag = new Bag();
    bag.forEach(b => { bag = new Bag(); });
    print(bag.items.length);
}
