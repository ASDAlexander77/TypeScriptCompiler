// An arrow without parameter types, `(v) => ...`, is generic until something gives it a type. As a
// `new` argument it gets one when it is cast to the constructor parameter's type, and the cast
// instantiated it there - but when the arrow captured a local, the instantiation built the closure
// over the captured variables and the cast went on with the bare function instead. The constructor
// stored a function whose captures were a null pointer, and the first call crashed - under every
// memory model, so it is a front-end bug. A non-capturing arrow, or one with typed parameters,
// never went that way.

class ParamProperty {
    constructor(private f: (v: number) => void) {}

    run(v: number) {
        this.f(v);
    }
}

class StoredField {
    f: (v: number) => void;

    constructor(f: (v: number) => void) {
        this.f = f;
    }

    run(v: number) {
        this.f(v);
    }
}

class Mapper {
    constructor(private f: (v: number) => number) {}

    apply(v: number) {
        return this.f(v);
    }
}

let globalSeen: number = 0;

function main() {
    let seen: number = 0;
    const p = new ParamProperty((v) => {
        seen += v;
    });
    p.run(1);
    p.run(2);
    assert(seen == 3, "a capturing untyped arrow in a parameter property");

    let total: number = 0;
    const s = new StoredField((v) => {
        total += v;
    });
    s.run(5);
    assert(total == 5, "a capturing untyped arrow stored in a field");

    const k = 10;
    const m = new Mapper((v) => v * k);
    assert(m.apply(4) == 40, "a capturing untyped arrow returning a value");

    const g = new ParamProperty((v) => {
        globalSeen += v;
    });
    g.run(7);
    assert(globalSeen == 7, "a non-capturing untyped arrow");

    print("done.");
}
