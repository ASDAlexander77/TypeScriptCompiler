// What a subclass's own field holds is freed when an object of the subclass is released through a
// slot of its base's type (spec 89): a WeakRef to it gives undefined afterwards (spec 84). Before the
// base's release routine told the object's class by its vtable, the field's object leaked and the
// WeakRef still found it - which a test that only runs to the end cannot see.
class Part {
    constructor(public v: number) {}
}

class Base {
    constructor(public x: number) {}
}

class Mid extends Base {
    part: Part;
    constructor(x: number) {
        super(x);
        this.part = new Part(x);
    }
}

class Top extends Mid {
    extra: Part;
    constructor(x: number) {
        super(x);
        this.extra = new Part(x + 1);
    }
}

class Holder {
    v: Base = new Base(0);
}

function main() {
    const h = new Holder();
    for (let i = 0; i < 100; i++) {
        const m = new Mid(i);
        const midPart = new WeakRef(m.part);
        h.v = m;
        h.v = new Base(i);
        assert(midPart.deref() === undefined, "a Mid's part, released through Base");

        const t = new Top(i);
        const topPart = new WeakRef(t.part);
        const topExtra = new WeakRef(t.extra);
        h.v = t;
        h.v = new Base(i);
        assert(topPart.deref() === undefined, "a Top's part, released through Base");
        assert(topExtra.deref() === undefined, "a Top's extra, released through Base");
    }

    print("done.");
}
