// A parameter with a default (`d: C[] = []`) is kept as any parameter is: its slot holds the
// argument given, moved in by the caller, or the default made here when none was given, and a store
// of it into a field moves it out on either path. A path that does not keep it releases what the
// slot holds, whichever it is. The argument left out is `undefined`, which holds no block. Before,
// the merge of the argument and the default was "takes a second reference". Each step churns, so a
// value freed too early reads garbage; the same program runs under every model.
class C { constructor(public x: number) {} }
function label(i: number) { return "s" + i; }
function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) junk.push(new C(999));
}
class A {
    d: C[] = [];
    constructor(d: C[] = [new C(7)], keep = true) {
        if (keep) this.d = d;
    }
}
class B {
    constructor(private items: C[] = [], public name = label(3)) {}
    first() { return this.items.length > 0 ? this.items[0].x : -1; }
}
function main() {
    let ok = 0;
    for (let i = 0; i < 1000; i++) {
        const a = new A();
        churn();
        if (a.d[0].x == 7) ok++;
        const b = new A([new C(i)]);
        churn();
        if (b.d[0].x == i) ok++;
        const c = new A([new C(i)], false);
        const e = new A(undefined, false);
        churn();
        if (c.d.length == 0 && e.d.length == 0) ok++;
        const f = new B();
        const g = new B([new C(5)], label(i));
        churn();
        if (f.first() == -1 && g.first() == 5 && f.name == "s3" && g.name == label(i)) ok++;
    }
    assert(ok == 1000 * 4, "ok");
    print("done.");
}
