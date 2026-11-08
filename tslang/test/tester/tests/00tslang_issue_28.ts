// TSLANG_ISSUES 28: TypeScript makes a parameter property of a constructor parameter with `public`,
// `private`, `protected` or `readonly`. A parameter with `readonly` alone was a plain parameter, so
// `t.width` didn't resolve ("can't resolve property/field/base 'width' of class 'Targets'").

class Targets {
    readonly tag: int = 7;
    constructor(readonly width: int, readonly height: int) {
    }
}

// a readonly parameter property with a default, a string one, and one mixed with a plain parameter
class Named {
    constructor(readonly name: string, scale: int, readonly count: int = 2) {
        this.scaled = scale * count;
    }
    scaled: int = 0;
}

// `private readonly` and `protected readonly` stay what they were
class Base {
    constructor(protected readonly id: int, private readonly secret: string) {
    }
    secretLength() {
        return this.secret.length;
    }
}

class Derived extends Base {
    constructor(readonly extra: int) {
        super(extra * 10, "abc");
    }
    total() {
        return this.id + this.extra;
    }
}

function main() {
    const t = new Targets(3, 4);
    print(`tag ${t.tag} size ${t.width} x ${t.height}`);
    assert(t.tag == 7 && t.width == 3 && t.height == 4, "readonly parameter properties");

    const n = new Named("plane", 5);
    assert(n.name == "plane", "a readonly string parameter property");
    assert(n.count == 2, "a readonly parameter property takes its default");
    assert(n.scaled == 10, "a plain parameter beside readonly ones");

    const m = new Named("cube", 3, 4);
    assert(m.name == "cube" && m.count == 4 && m.scaled == 12, "a readonly parameter property given a value");

    const d = new Derived(2);
    assert(d.extra == 2, "a readonly parameter property of a derived class");
    assert(d.total() == 22, "a protected readonly parameter property");
    assert(d.secretLength() == 3, "a private readonly parameter property");

    print("done.");
}
