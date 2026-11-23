// TSLANG_ISSUES 108: `this` in a static method is the class it is called through. A base class's static
// method that reads `this` was compiled once, with `this` the base, so `Derived.me()` returned Base and
// `Derived.create()` made a Base. Each subclass now has a copy of its own of such a method.
class Base {
    static label = "base";
    static me() { return this; }
    static name2() { return this.label; }
    static create() { return new this(); }
    static viaArrow() { const f = () => this.label; return f(); }
    static plain() { return 1; }
    kind() { return "base"; }
}
class Derived extends Base {
    static label = "derived";
    kind() { return "derived"; }
}
class Leaf extends Derived {
    static label = "leaf";
    kind() { return "leaf"; }
}
class Own extends Base {
    static label = "own";
    static name2() { return "declared"; }
    static create() { return new Own(); }
}

function main() {
    assert(Base.me() === Base && Derived.me() === Derived && Leaf.me() === Leaf, "me");
    assert(Base.name2() == "base" && Derived.name2() == "derived" && Leaf.name2() == "leaf", "this.label");
    assert(Base.create().kind() == "base" && Derived.create().kind() == "derived" && Leaf.create().kind() == "leaf", "new this()");
    assert(Derived.viaArrow() == "derived", "an arrow reads the method's this");
    assert(Derived.plain() == 1, "a method that does not read this");
    assert(Own.name2() == "declared" && Own.me() === Own, "a method the subclass declares itself");

    let k: typeof Base = Base;
    assert(k.name2() == "base", "through a class value");
    k = Derived;
    assert(k.name2() == "derived" && k.create().kind() == "derived", "through a class value holding a subclass");
    print("done.");
}
