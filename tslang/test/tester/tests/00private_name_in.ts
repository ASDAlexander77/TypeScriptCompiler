// `#name in obj`: true for an object of the class that declares #name or of a subclass of it
class C {
    #f = 1;
    static has(o: any) {
        return #f in o;
    }
    hasThis() {
        return #f in this;
    }
    get f() { return this.#f; }
}

class D extends C { }

class E {
    #f = 2;
    static has(o: E | C) {
        return #f in o;
    }
}

function main() {
    assert(C.has(new C()));
    assert(C.has(new D()));
    assert(!C.has(new E()));
    assert(!C.has({ f: 1 }));
    assert(new D().hasThis());
    const e = new E();
    const c = new C();
    assert(E.has(e));
    assert(!E.has(c));
    print("done.");
}
