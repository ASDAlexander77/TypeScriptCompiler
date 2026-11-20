// A class boxed into any whose members take a union (a field, a setter's value, a method's parameter): `any` is not
// cast to a union, so such a field or setter is read through any but not written, and such a method is read bound
// but not called through any. Before, their adapters did not compile, and the whole module with them (the default
// library, once its exported classes had keys adapters).
class U {
    v: number | string = "s";
    private _s: number | string = 1;

    get s(): number | string {
        return this._s;
    }

    set s(x: number | string) {
        this._s = x;
    }

    f(p: number | string): string {
        return "f:" + p;
    }
}

function key(s: string) {
    return s;
}

function main() {
    const u = new U();
    const a: any = u;

    assert(<string>a[key("v")] == "s");
    assert(typeof a[key("s")] == "number");
    assert(typeof a[key("f")] == "function");

    let threw = false;
    try {
        a[key("v")] = 5;
    } catch (e) {
        threw = true;
    }
    assert(threw);

    // read through the class's own type, as ever
    u.s = "t";
    assert(u.f(u.s) == "f:t");

    print("done.");
}
