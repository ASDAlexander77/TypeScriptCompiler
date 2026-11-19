// `x[k]` and `k in x` of a value of type `any` (user ruling 11-19): looked up at run time through the
// adapter made for the type of the value when it was boxed (TYPE_DESCR_KEYS). A key the value does not
// have reads undefined; a method is in it, but reading one is a TypeError.
class Point {
    constructor(public x: number, protected y: number) {
    }

    get sum() { return this.x + this.y; }

    norm() { return this.x * this.x + this.y * this.y; }
}

class Point3 extends Point {
    constructor(x: number, y: number, public z: number) {
        super(x, y);
    }
}

interface Named {
    name: string;
    title?: string;
}

class Node1 {
    next: Node1 | null = null;
    constructor(public value: number) {
    }
}

function get(o: any, k: any): any {
    return o[k];
}

function has(o: any, k: any): boolean {
    return k in o;
}

function main() {
    // an object literal
    const o: any = { a: 1, b: "two" };
    assert(<number>o["a"] == 1);
    assert(<string>o["b"] == "two");
    assert(typeof o["c"] == "undefined");
    const ka = "a";
    assert(<number>get(o, ka) == 1);
    assert(has(o, "b"));
    assert(!has(o, "c"));
    assert("a" in o);

    // a tuple: its positions
    const t: any = <[number, string]>[5, "five"];
    assert(<number>t[0] == 5);
    assert(<string>get(t, 1) == "five");
    assert(has(t, 0));
    assert(!has(t, 2));

    // a class: its public fields and accessors, a base's too, and methods only for `in`; a protected field is
    // not read from outside the class, through `any` either
    const p: any = new Point3(3, 4, 5);
    assert(<number>p["x"] == 3);
    assert(typeof get(p, "y") == "undefined");
    assert(!has(p, "y"));
    assert(<number>get(p, "z") == 5);
    assert(<number>get(p, "sum") == 7);
    assert(has(p, "norm"));
    assert(!has(p, "w"));
    assert(typeof get(p, "w") == "undefined");
    let threw = false;
    try {
        get(p, "norm");
    } catch (e) {
        threw = true;
    }
    assert(threw);

    // a class with a field of its own type
    const n = new Node1(1);
    n.next = new Node1(2);
    const an: any = n;
    const next: Node1 | null = <Node1 | null>an["next"];
    assert(next != null && next.value == 2);

    // an interface: an optional field not set is not there, and reads undefined
    const named: Named = { name: "n" };
    const ai: any = named;
    assert(<string>ai["name"] == "n");
    assert(!has(ai, "title"));
    assert(typeof get(ai, "title") == "undefined");

    // an array (a constant one, boxed as the array it makes): its elements in range, and its length
    const a: any = [10, 20, 30];
    assert(<number>a[1] == 20);
    assert(<number>get(a, "length") == 3);
    assert(typeof get(a, 3) == "undefined");
    assert(has(a, 2));
    assert(!has(a, 3));
    const nums: number[] = [1.5, 2.5];
    const an2: any = nums;
    assert(<number>an2[1] == 2.5);
    const back = <number[]>an2;
    assert(back.length == 2 && back[0] == 1.5);

    // a string: its characters and its length
    const s: any = "abc";
    assert(<string>s[1] == "b");
    assert(<number>get(s, "length") == 3);

    // a number has no keys: undefined, and `in` of it a TypeError
    const num: any = 5;
    assert(typeof get(num, "x") == "undefined");

    // null: a TypeError
    let nullThrew = false;
    try {
        get(null, "x");
    } catch (e) {
        nullThrew = true;
    }
    assert(nullThrew);

    // a key known only at run time, of a value whose type knows its keys
    const plain = { u: 1, v: 2 };
    let key = "v";
    assert(key in plain);
    key = "w";
    assert(!(key in plain));

    print("done.");
}
