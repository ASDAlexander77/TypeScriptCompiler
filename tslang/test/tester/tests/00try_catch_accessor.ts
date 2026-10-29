// An accessor that throws inside `try` is caught there: its call is lowered after the `try`, which recorded only
// the calls and throws already in its body, so a get or set reached the caller uncaught. Through an interface's
// field the call is made by the load or the store of a slot naming a cell of accessors, as late as LLVM: the access
// ends its block with an edge to the handler, and calls the accessor by an invoke unwinding there.
class Fail {
    constructor(public where: string) {}
}

class A {
    private n = 1;
    get v() { if (this.n < 0) throw new Fail("get"); return this.n; }
    set v(x: number) { if (x < 0) throw new Fail("set"); this.n = x; }

    static get s() { throw new Fail("static get"); return 0; }
    static set s(x: number) { throw new Fail("static set"); }
}

class B extends A {
    override get v() { return super.v + 100; }
    override set v(x: number) { super.v = x; }
}

interface Named {
    name: string;
}

class Box implements Named {
    private n = "box";
    get name() { if (this.n == "bad") throw new Fail("interface get"); return this.n; }
    set name(v: string) { if (v == "") throw new Fail("interface set"); this.n = v; }
}

class Plain implements Named {
    name = "plain";
}

abstract class Shape {
    abstract size: number;
}

class Square extends Shape {
    get size() { throw new Fail("abstract get"); return 0; }
    set size(v: number) { throw new Fail("abstract set"); }
}

function caught(f: () => void) {
    try {
        f();
    } catch (e: Fail) {
        return e.where;
    }

    return "";
}

function main() {
    const a = new A();
    let where = "";
    try { a.v = -1; } catch (e: Fail) { where = e.where; }
    assert(where == "set", "set");

    a.v = 5;
    assert(a.v == 5, "a set that does not throw");

    where = "";
    try { a.v = -2; print("not reached"); } catch (e: Fail) { where = e.where; }
    assert(where == "set" && a.v == 5, "set, the statement after it skipped");

    where = "";
    try { A.s = 1; } catch (e: Fail) { where = e.where; }
    assert(where == "static set", "static set");

    where = "";
    try { print(A.s); } catch (e: Fail) { where = e.where; }
    assert(where == "static get", "static get");

    const b: A = new B();
    where = "";
    try { b.v = -1; } catch (e: Fail) { where = e.where; }
    assert(where == "set", "virtual set");

    const box: Named = new Box();
    where = "";
    try { box.name = ""; } catch (e: Fail) { where = e.where; }
    assert(where == "interface set", "set through an interface");

    box.name = "bad";
    where = "";
    try { print(box.name); } catch (e: Fail) { where = e.where; }
    assert(where == "interface get", "get through an interface");

    box.name = "ok";
    where = "";
    try { where = box.name; } catch (e: Fail) { where = "wrong"; }
    assert(where == "ok", "get through an interface that does not throw");

    let boxFinally = false;
    where = "";
    try {
        try { box.name = ""; } finally { boxFinally = true; }
    } catch (e: Fail) {
        where = e.where;
    }
    assert(boxFinally && where == "interface set", "set through an interface, finally");

    // a plain field behind the interface: the slot is no cell
    const plain: Named = new Plain();
    where = "";
    try { plain.name = "plain2"; where = plain.name; } catch (e: Fail) { where = "wrong"; }
    assert(where == "plain2", "a field through an interface");

    const shape: Shape = new Square();
    where = "";
    try { print(shape.size); } catch (e: Fail) { where = e.where; }
    assert(where == "abstract get", "get through an abstract field");
    where = "";
    try { shape.size = 1; } catch (e: Fail) { where = e.where; }
    assert(where == "abstract set", "set through an abstract field");

    // a finally still runs
    let finallyRan = false;
    try {
        try { a.v = -3; } finally { finallyRan = true; }
    } catch (e: Fail) {
        where = e.where;
    }
    assert(finallyRan && where == "set", "finally");

    assert(caught(() => { a.v = -4; }) == "set", "in a function");

    print("done.");
}
