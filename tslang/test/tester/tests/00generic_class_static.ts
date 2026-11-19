// A static member of a generic class through the class itself, as TypeScript allows (`Box.of(1)`):
// the generic class had no members to find ("Can't resolve property 'of' of type Box")
class Box<T> {
    constructor(public value: T) {}

    public static of(value: number): Box<number> {
        return new Box<number>(value + 1);
    }

    public static label = "box";

    public static describe(): string {
        return "a " + Box.label;
    }

    // an instance method of the same name, and one a T without its arguments would not compile
    public of(value: T): Box<T> {
        return new Box<T>(value);
    }

    public get(): T {
        return this.value;
    }
}

class Pair<K = string, V = number> {
    constructor(public key: K, public value: V) {}

    public static make(key: string, value: number): Pair<string, number> {
        return new Pair<string, number>(key, value);
    }
}

interface Named {
    name: string;
}

class Item implements Named {
    constructor(public name: string) {}
}

class Registry<T extends Named> {
    private items: T[] = [];

    public add(item: T) {
        this.items.push(item);
    }

    public static empty(): number {
        return 0;
    }

    public get count(): number {
        return this.items.length;
    }
}

function main() {
    assert(Box.of(1).value == 2, "a static method");
    assert(Box.of(41).get() == 42, "its result's instance method");
    assert(Box.label == "box" && Box.describe() == "a box", "a static field and a static method using it");
    assert(new Box<string>("x").of("y").value == "y", "the instance twin");

    const p = Pair.make("a", 1);
    assert(p.key == "a" && p.value == 1, "type parameters with defaults");

    assert(Registry.empty() == 0, "a constrained type parameter");
    const r = new Registry<Item>();
    r.add(new Item("n"));
    assert(r.count == 1, "the instantiated class still works");

    print("done.");
}
