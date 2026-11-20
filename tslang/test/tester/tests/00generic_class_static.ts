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

// A static field of a generic class is one field for every instantiation, as in TypeScript: each had
// its own (`Counter<number>.made` beside `Counter<string>.made`), so a count kept by the constructor
// was split between them
class Counter<T> {
    public static made = 0;
    public static log: string[] = [];

    constructor(public value: T) {
        Counter.made++;
        Counter.log.push("made");
    }

    public seen(): number {
        return Counter.made;
    }

    public entries(): number {
        return Counter.log.length;
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

// a static field the first use of its generic class, in a const's initializer
class Tally<T> {
    public static total = 0;
    public static names: string[] = [];
}

// a private static member of a generic class, used inside it through the class: the access level was
// the generic class's, not the class being compiled ("Class member 'encode' is not accessable")
class Codec<S> {
    private static offset = 1;

    private static encode<E>(value: number): number {
        return value + Codec.offset;
    }

    private static decode(value: number): number {
        return value - Codec.offset;
    }

    public roundTrip(value: number): number {
        return Codec.decode(Codec.encode<S>(value) * 2);
    }
}

function main() {
    const total = Tally.total;
    const named = Tally.names.length == 0 && total == 0;
    assert(named, "a static field read before the class is used otherwise");
    Tally.total = 4;
    assert(Tally.total == 4, "and written");

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

    const n = new Counter<number>(1);
    const s = new Counter<string>("a");
    new Counter<number>(2);
    assert(Counter.made == 3 && n.seen() == 3 && s.seen() == 3, "one static field for every instantiation");
    assert(Counter.log.length == 3 && n.entries() == 3 && s.entries() == 3, "an initialized array, made once");
    Counter.made = 10;
    assert(n.seen() == 10 && s.seen() == 10, "written through the class, read through each");

    assert(new Codec<string>().roundTrip(5) == 11 && new Codec<number>().roundTrip(0) == 1, "private statics inside the class");

    print("done.");
}
