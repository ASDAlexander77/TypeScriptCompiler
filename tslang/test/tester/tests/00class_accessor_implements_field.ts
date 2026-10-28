// A getter/setter filling a field of a base class or of an interface (docs/not_impl/not_impl15.ts).
// `abstract prop: T` is an abstract get/set pair, so a read or write through the base reaches what a
// subclass fills it with: accessors of its own, or a field (given accessors over it). Through an interface,
// a field filled by accessors has a slot naming a cell of them (INTERFACE_ACCESSOR_CELL_TAG, Defines.h).
// A read through the base used to read a slot of the base's own that nothing wrote, and print null.

interface A {
    prop: string;
    raw: string;
    m(): string;
}

abstract class B implements A {
    abstract prop: string;
    abstract raw: string;
    abstract readonly ro: string;
    abstract get readonlyProp(): string;
    abstract set readonlyProp(val: string);
    abstract m(): string;

    describe() {
        return this.prop + "/" + this.raw + "/" + this.ro + "/" + this.readonlyProp;
    }
}

class C extends B {
    written = "";
    get prop() { return "foo"; }
    set prop(v: string) { this.written = v; }
    raw = "edge";
    readonly ro = "readonly please";
    readonlyProp: string = "rp";
    m() { return "m"; }
}

interface Named {
    name: string;
    count: number;
}

class Accessors implements Named {
    _name = "n";
    sets = 0;
    get name() { return this._name + "!"; }
    set name(v: string) { this._name = v; this.sets++; }
    count: number = 3;
}

class Fields implements Named {
    name = "plain";
    count: number = 4;
}

class GetterOnly implements Named {
    get name() { return "fixed"; }
    get count(): number { return 5; }
}

function readName(n: Named) {
    return n.name;
}

function main() {
    const c = new C();

    // through the abstract base
    const b: B = c;
    assert(b.prop == "foo");
    b.prop = "x";
    assert(c.written == "x");
    assert(b.raw == "edge");
    b.raw = "y";
    assert(c.raw == "y");
    assert(b.ro == "readonly please");
    assert(b.readonlyProp == "rp");
    b.readonlyProp = "rp2";
    assert(c.readonlyProp == "rp2");
    assert(b.describe() == "foo/y/readonly please/rp2");

    // through an interface the base implements
    const a: A = c;
    assert(a.prop == "foo");
    a.prop = "z";
    assert(c.written == "z");
    assert(a.raw == "y");
    a.raw = "w";
    assert(c.raw == "w");
    assert(a.m() == "m");

    // through an interface the class implements, with accessors or with fields
    const acc = new Accessors();
    const n: Named = acc;
    assert(n.name == "n!");
    n.name = "v";
    assert(acc.sets == 1);
    assert(n.name == "v!");
    assert(n.count == 3);
    n.count = 6;
    assert(acc.count == 6);

    assert(readName(acc) == "v!");
    assert(readName(new Fields()) == "plain");

    // a getter only: a write through the interface has nothing to call
    const g: Named = new GetterOnly();
    assert(g.name == "fixed");
    assert(g.count == 5);
    g.name = "ignored";
    assert(g.name == "fixed");

    print("done.");
}
