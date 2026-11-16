// An object of a subclass kept in a slot of its base's type (`h.v: Base = new Mid()`), an element of a
// `Base[]`, or the object of a method value, is released by the base's routine: that routine tells the
// object's class by its vtable and gives back the fields the subclass adds too (they leaked under rc
// and own before). Three levels; the arrays are made at run time, so a double release shows.
class Base {
    items: number[] = [];
    constructor(public x: number) {
        this.items.push(x);
    }

    get() {
        return this.x + this.items.length;
    }
}

class Mid extends Base {
    more: number[] = [];
    constructor(x: number) {
        super(x);
        this.more.push(x);
    }

    get() {
        return this.x * 2 + this.more.length;
    }
}

class Top extends Mid {
    names: string[] = [];
    constructor(x: number) {
        super(x);
        this.names.push("t" + x);
    }

    get() {
        return this.x * 3 + this.names.length;
    }
}

class Holder {
    v: Base = new Base(0);
    m: Mid = new Mid(0);
    f: () => number = () => 0;
}

function make(i: number): Base {
    const k = i % 3;
    if (k == 0) {
        return new Base(i);
    }

    if (k == 1) {
        return new Mid(i);
    }

    return new Top(i);
}

function makeMid(i: number): Mid {
    if (i % 2 == 0) {
        return new Mid(i);
    }

    return new Top(i);
}

function churn() {
    const junk: Base[] = [];
    for (let i = 0; i < 32; i++) junk.push(new Top(999));
    return junk.length;
}

function main() {
    const h = new Holder();
    const all: Base[] = [];
    let t: number = 0;
    for (let i = 0; i < 3000; i++) {
        h.v = make(i);
        h.m = makeMid(i);
        const b: Base = new Top(i);
        h.f = b.get;
        all.push(make(i + 1));
        if (all.length > 10) {
            all.shift();
        }

        churn();
        t += h.v.get() + h.m.get() + h.f() + all[all.length - 1].get();
    }

    assert(t > 0, "t");
    print("done.");
}
