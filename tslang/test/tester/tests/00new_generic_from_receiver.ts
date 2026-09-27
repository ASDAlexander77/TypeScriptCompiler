// `new C()` of a generic class takes its type arguments from the type it is given to, as in
// `const m: Map<string, number[]> = new Map()`. Without them a plain generic was an error ("missing
// type arguments"), and one with defaults (`Map<K = any, V = any>`) became C<any, any>, whose cast
// to the declared type crashed at run time (#231).

class Box<T> {
    items: T[] = [];

    add(v: T) {
        this.items.push(v);
    }

    get(i: number): T {
        return this.items[i];
    }
}

class Pair<K = any, V = any> {
    keys: K[] = [];
    values: V[] = [];

    set(k: K, v: V) {
        this.keys.push(k);
        this.values.push(v);
    }
}

class Holder {
    private readonly boxes: Box<string> = new Box();
    pairs: Pair<string, number[]>;

    constructor() {
        this.pairs = new Pair();
    }

    fill() {
        this.boxes.add("x");
        this.pairs.set("a", [1, 2, 3]);
        return this.boxes.get(0) + this.pairs.values[0][2];
    }
}

function main() {
    const b: Box<number> = new Box();
    b.add(5);
    assert(b.get(0) == 5);

    const p: Pair<string, number> = new Pair();
    p.set("k", 7);
    assert(p.keys[0] == "k" && p.values[0] == 7);

    let q = new Pair<number, string>();
    q.set(0, "zero");
    q = new Pair();
    q.set(1, "one");
    assert(q.values[0] == "one");

    // explicit type arguments still win
    const e: Box<number> = new Box<number>();
    e.add(1);
    assert(e.get(0) == 1);

    assert(new Holder().fill() == "x3");

    print("done.");
}
