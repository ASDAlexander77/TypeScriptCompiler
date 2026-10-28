// A union value narrowed by `instanceof` is a view of the union's value: the test hands the object to
// its class's `.instanceOf` and reads its vtable, which keeps nothing (spec 61), so the narrowed name
// can be called again. Each step churns, so an object freed too early reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

class A {
    constructor(public k: number) {}

    walk(a: Item[]): number {
        let s: number = 0;
        for (let i = 0; i < a.length; i++) s += a[i].n * this.k;
        return s;
    }
}

class B {
    walk(a: Item[]): number {
        return -1;
    }
}

function pick(i: number): A | B {
    return i < 0 ? new B() : new A(2);
}

function test(a: Item[], i: number) {
    const u = pick(i);
    let r: number = 0;
    if (u instanceof A) {
        r += u.walk(a);
        churn();
        r += u.walk(a) + u.k;
    } else {
        r += u.walk(a);
    }

    return r;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = [new Item(1), new Item(5), new Item(2)];
        if (test(a, i) == 34 && test(a, -1) == -1) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
