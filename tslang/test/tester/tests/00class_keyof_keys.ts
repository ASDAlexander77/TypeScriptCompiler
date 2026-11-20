// A class read by a `keyof` key (or a union of its names) known only at run time: a switch over the names,
// each read as `c.name` is - the result is the union of the members' types, a method bound to the object
// (its override, through the vtable). A write needs a value every named field can hold. `C[k]` with
// `k: keyof typeof C` reads a static.
class C {
    a: number = 1;
    b = "x";
    m(): number {
        return 3;
    }
    static s: number = 4;
    static t = "y";
}

class E extends C {
    override m(): number {
        return 30;
    }
}

class W {
    a: number = 1;
    a2: number = 2;
}

function pick(i: number): keyof C {
    return i == 0 ? "a" : i == 1 ? "b" : "m";
}

function main() {
    const c = new C();
    const v = c[pick(0)];
    assert(typeof v == "number" && v == 1);
    const s = c[pick(1)];
    assert(typeof s == "string" && s == "x");
    const f = c[pick(2)];
    assert(typeof f == "function" && f() == 3);

    // the override
    const e: C = new E();
    const g = e[pick(2)];
    assert(typeof g == "function" && g() == 30);

    // a write through a union of names whose fields all hold the value
    const w = new W();
    const wk: "a" | "a2" = pick(0) == "a" ? "a2" : "a";
    w[wk] = 7;
    assert(w.a2 == 7 && w.a == 1);

    // `in` with a keyof key
    assert(pick(1) in c);

    // a static
    const sk: keyof typeof C = pick(0) == "a" ? "t" : "s";
    const st = C[sk];
    assert(typeof st == "string" && st == "y");

    print("done.");
}
