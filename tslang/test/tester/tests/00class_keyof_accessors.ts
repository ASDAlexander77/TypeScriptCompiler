// `keyof D` of a class with an accessor and a base class is its fields and accessors as tsc has them: not the
// accessor's functions (`get_g`), nor the base's storage the class begins with (its id is the base's name). So a
// key set of fields and accessors reads and writes, under every model -mm=own included (an accessor is no
// method: reading it moves nothing).
class B {
    private _g: number = 1;

    get g(): number {
        return this._g;
    }

    set g(v: number) {
        this._g = v;
    }
}

class D extends B {
    a: number = 2;
}

function pick(i: number): keyof D {
    return i == 0 ? "a" : "g";
}

function main() {
    const d = new D();
    assert(d[pick(1)] == 1);
    assert(d[pick(0)] == 2);

    d[pick(0)] = 5;
    assert(d.a == 5);
    d[pick(1)] = 7;
    assert(d.g == 7);

    print("done.");
}
