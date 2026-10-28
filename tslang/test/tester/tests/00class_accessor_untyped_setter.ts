// a set accessor's parameter without a type takes the type its get accessor returns
class GetFirst {
    f = 1;
    get prop() { return this.f; }
    set prop(v) { print("set", v); this.f = v; }
}

// the get accessor after the set accessor
class SetFirst {
    f = 1;
    set prop(v) { this.f = v + 1; }
    get prop() { return this.f; }
}

// a get accessor with a written type
class Typed {
    s = "";
    get prop(): string { return this.s; }
    set prop(v) { this.s = v + "!"; }
}

class Static {
    static f = 1;
    static get prop() { return Static.f; }
    static set prop(v) { Static.f = v * 2; }
}

// the value is given to a set accessor as the type it takes, not as the one its get accessor returns
class AnySetter {
    f = 1;
    get prop() { return this.f; }
    set prop(v: any) { this.f = v; }
}

function main() {
    const c = new GetFirst();
    c.prop = 5;
    print(c.prop);
    assert(c.prop == 5);

    const s = new SetFirst();
    s.prop = 5;
    assert(s.prop == 6);

    const t = new Typed();
    t.prop = "hi";
    assert(t.prop == "hi!");

    Static.prop = 4;
    assert(Static.prop == 8);

    const a = new AnySetter();
    a.prop = 7;
    assert(a.prop == 7);

    print("done.");
}
