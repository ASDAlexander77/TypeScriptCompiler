// `new (...) => A` with `A` a class is the class value `typeof A`: a class given to it, `A` or a subclass, is read at
// run time, and `new c()` makes one of that class. Its constructor is called as `A`'s is, so a subclass given there
// takes the same parameters (a constructor taking others is called with what `A`'s takes: wasm traps on that).
class A { n = 1; constructor(public tag: string = "a") {} }
class B extends A { constructor(tag: string = "b") { super(tag); this.n = 2; } }

function make(c: new () => A): A { return new c(); }

class Registry {
    items: A[] = [];
    register(c: new (...args: any[]) => A) { this.items.push(new c()); }
}

function main() {
    const a = make(B);
    assert(a.n == 2 && a.tag == "b");
    const b = make(A);
    assert(b.n == 1 && b.tag == "a");
    const r = new Registry();
    r.register(A);
    r.register(B);
    assert(r.items.length == 2 && r.items[1].n == 2);
    let ctor: new () => A = B;
    assert(new ctor().n == 2);
    print("done.");
}
