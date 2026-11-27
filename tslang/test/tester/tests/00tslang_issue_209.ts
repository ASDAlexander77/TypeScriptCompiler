// TSLANG 209, every model: under -mm=own a method that keeps its parameter (pushes it) and throws before keeping it - by
// a `throw` of its own or a call that throws - leaked the parameter: 1M throws grew to about 60 MB.

class C {
    s: string;
    constructor(public n: number) { this.s = "x" + n; }
}
function check(n: number) {
    if (n % 3 == 0) throw "no";
}
function checkC(c: C) {
    if (c.n % 3 == 0) throw "no";
}
// (each method keeps `c` on every way it returns by)
class K {
    items: C[] = [];
    keep(c: C) {
        check(c.n);
        this.items.push(c);
    }
    keepCaught(c: C) {
        try {
            check(c.n);
        } catch (e) {}
        this.items.push(c);
    }
    keepPassed(c: C) {
        checkC(c);
        this.items.push(c);
    }
    keepThrow(c: C) {
        if (c.n % 3 == 0) throw "no";
        this.items.push(c);
    }
}
function main() {
    const k = new K();
    let thrown = 0;
    for (let i = 0; i < 6; i++) {
        try { k.keep(new C(i)); } catch (e) { thrown++; }
        try { k.keepCaught(new C(i)); } catch (e) { thrown++; }
        try { k.keepPassed(new C(i)); } catch (e) { thrown++; }
        try { k.keepThrow(new C(i)); } catch (e) { thrown++; }
    }
    assert(thrown == 6 && k.items.length == 18, "counts");
    assert(k.items[17].s == "x5");
    print("done.");
}
