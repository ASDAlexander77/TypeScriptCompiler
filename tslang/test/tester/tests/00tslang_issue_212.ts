// TSLANG 212: leftovers of 211's triage.

// 132: a generic class given for a constructor type takes its type arguments from the type's parameters - a rest's
// element type for each constructor parameter it reaches (`any` makes T `any`, as TypeScript infers `Banana<any>`)
class Banana<T extends string> {
    constructor(public a: string, public property: T) { }
}
function fruitFactory1<TFruit>(Fruit: new (...args: any[]) => TFruit): TFruit { return new Fruit("x1", "foo"); }
function fruitFactory2<TFruit>(Fruit: new (a: string, ...args: any[]) => TFruit): TFruit { return new Fruit("x2", "foo"); }
function fruitFactory3<TFruit>(Fruit: new (a: string, s: "foo", ...args: any[]) => TFruit): TFruit { return new Fruit("x3", "foo"); }
function fruitFactory4<TFruit>(Fruit: new (a: string, ...args: "foo"[]) => TFruit): TFruit { return new Fruit("x4", "foo"); }
function fruitFactory5<TFruit>(Fruit: new (...args: "foo"[]) => TFruit): TFruit { return new Fruit("foo", "foo"); }

function testGenericClassForConstructorType() {
    const b1 = fruitFactory1(Banana);
    assert(b1.a == "x1" && b1.property == "foo", "132 rest of any");
    const b2 = fruitFactory2(Banana);
    assert(b2.a == "x2" && b2.property == "foo", "132 a, rest of any");
    const b3 = fruitFactory3(Banana);
    assert(b3.a == "x3" && b3.property == "foo", "132 a, s, rest");
    const b4 = fruitFactory4(Banana);
    assert(b4.a == "x4" && b4.property == "foo", "132 a, rest of \"foo\"");
    const b5 = fruitFactory5(Banana);
    assert(b5.a == "foo" && b5.property == "foo", "132 rest of \"foo\"");
}

// 164: an interface narrowed to a class by `instanceof` is the object it holds - under -mm=own a borrow, as the
// interface is (it was taken for a reference of its own: "'v' is a parameter, kept here")
interface IA { a: number; }
interface IB { b: string; }
class A implements IA { a: number = 1; s: string; constructor(s: string) { this.s = s; } }
class B implements IB { b = "x"; }

function narrowed(v: IA): string {
    if (v instanceof A) return "A" + v.a + v.s;
    return "?";
}

function narrowedFromUnion(v: IA | IB): string {
    if (!(v instanceof A)) return "?";
    return v.s;
}

// ...and by a generic class: true of every G<T>, it narrows a union to its members that are one, else not at all (an
// interface crashed the compiler, a union read "Can't resolve property")
class G<T> implements IA { a: number = 2; field: T; constructor(f: T) { this.field = f; } }
class K extends G<string> { k = "k"; }

function genericOfInterface(v: IA): string {
    if (v instanceof G) return "G" + v.a;
    return "?";
}

function genericOfUnion(v: G<number> | B): string {
    if (v instanceof G) return "G" + v.field;
    return "B" + v.b;
}

function derivedOfUnion(v: K | B): string {
    if (!(v instanceof G)) return "B" + v.b;
    return "K" + v.k;
}

function testInstanceOfNarrowing() {
    for (let i = 0; i < 3; i++) {
        const a: IA = new A("s" + i);
        const u: IA | IB = new A("t" + i);
        const b: IA | IB = new B();
        assert(narrowed(a) == "A1s" + i, "164 interface");
        assert(narrowedFromUnion(u) == "t" + i && narrowedFromUnion(b) == "?", "164 union of interfaces");
    }

    assert(genericOfInterface(new G<number>(5)) == "G2", "generic of an interface");
    assert(genericOfInterface(new A("z")) == "?", "generic of an interface, another class");
    assert(genericOfUnion(new G<number>(7)) == "G7" && genericOfUnion(new B()) == "Bx", "generic of a union");
    assert(derivedOfUnion(new K("s")) == "Kk" && derivedOfUnion(new B()) == "Bx", "derived of a union");
}

// 144: a const a function before it reads, holding a closure or an object, owns it under -mm=own ("captured by a
// closure, but -mm=own cannot tell who owns its value")
class C { v: string; constructor(v: string) { this.v = v; } }

function forwardClosure(k: number): number {
    const f = () => g() + k;
    const g = () => 10 + k;
    return f();
}

function forwardObjects(i: number): string {
    const f = () => o.x + c.v;
    const o = { x: "a" + i };
    const c = new C("c" + i);
    const h = () => c.v + "!";
    return f() + h();
}

function forwardEarlyExit(i: number): string {
    const f = () => k.v;
    if (i % 2 == 0) return "early";
    const k = new C("k" + i);
    return f();
}

function testForwardCaptures() {
    for (let i = 0; i < 3; i++) {
        assert(forwardClosure(i) == 10 + 2 * i, "144 closure");
        assert(forwardObjects(i) == "a" + i + "c" + i + "c" + i + "!", "144 objects");
        assert(forwardEarlyExit(i) == (i % 2 == 0 ? "early" : "k" + i), "144 early exit");
    }
}

// 191: an array of a union said to be an array of members of it is read element by element (its elements are unions:
// "cast from array type ... is not supported" when it was lowered; tsc's `x.slice(1) as readonly string[]` of
// `[number, string] | [number, string, string]`, which needs the default library's slice)
function membersOf(i: number): number {
    const m: (number | string)[] = ["a" + i, "bb"];
    const r = m as readonly string[];
    return r.length * 10 + r[0].length + r[1].length;
}

function testArrayOfUnionAsMembers() {
    for (let i = 0; i < 3; i++) {
        assert(membersOf(i) == 24, "191 array of a union as readonly string[]");
    }
}

// a function an object literal holds, called (`o.make(i)`): its result is given back (it was never, under rc and
// -mm=own, wherever the program links a library function whose result is not classified)
class Made { s: string; constructor(public n: number) { this.s = "m" + n; } }

function testObjectLiteralFunction() {
    const o = { make: (n: number): Made => new Made(n) };
    let total = 0;
    for (let i = 0; i < 3; i++) {
        const m = o.make(i);
        total += m.n + m.s.length;
    }

    assert(total == 9, "object literal function");
}

function main() {
    testGenericClassForConstructorType();
    testInstanceOfNarrowing();
    testForwardCaptures();
    testArrayOfUnionAsMembers();
    testObjectLiteralFunction();
    print("done.");
}
