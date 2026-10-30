// A method that gives an element a second place says `@sharesElements` (spec 73). Made for an element
// type that owns a block of its own (an object), -mm=own stubs it and the rest of the class works; made
// for numbers, it works as it is. (A generator a stubbed function makes goes with it: the default
// library's Array<T>.values.)
class Item {
    constructor(public name: string) {}
}

class Bag<T> {
    items: T[] = [];

    add(item: T) {
        this.items.push(item);
    }

    get(i: int) {
        return this.items[i];
    }

    @sharesElements
    twice() {
        const both: T[] = [];
        for (const v of this.items) both.push(v);
        for (const v of this.items) both.push(v);
        return both;
    }
}

function main() {
    const bag = new Bag<Item>();
    bag.add(new Item("a"));
    bag.add(new Item("b"));
    assert(bag.get(1).name == "b", "objects without the sharing methods");

    const numbers = new Bag<number>();
    numbers.add(1);
    numbers.add(2);
    assert(numbers.twice().length == 4, "numbers share");

    print("done.");
}
